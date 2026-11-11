#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from random import randint
import math
import json

import numpy as np
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

def distCUDA2(*args, **kwargs):
    from simple_knn._C import distCUDA2 as operation
    return operation(*args, **kwargs)
def sort_with_plas(*args, **kwargs):
    from plas import sort_with_plas as operation
    return operation(*args, **kwargs)

from .TimeFormer import TimeFormer

from ..utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import strip_symmetric, build_scaling_rotation

from ..utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
def compute_relocation_cuda(*args, **kwargs):
    from utils.relocation_utils import compute_relocation_cuda as operation
    return operation(*args, **kwargs)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=1, include_input=True):
    if i == -1:
        return torch.nn.Identity(), 3

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(y):
    assert y.max() < 20, "Probably mixed up linear and log values for xyz. These going in here are supposed to be quite small (log scale)"
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.disable_xyz_log_activation:
            self.xyz_activation = lambda x: x
            self.inverse_xyz_activation = lambda x: x
        else:
            self.xyz_activation = inverse_log_transform
            self.inverse_xyz_activation = log_transform

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, disable_xyz_log_activation, eval=False):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.disable_xyz_log_activation = disable_xyz_log_activation

        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.time_line = 60  # 60
        self.feat_dim = 12
        self.time_batch = 3
        # self.time_per_feat = self.feat_dim // self.time_batch
        self.total_feat_dim = self.feat_dim * self.time_batch
        self._point_feats = []
        self.time_embedding_num = self.time_line + self.time_batch - 1

        self.time_pe_fn, time_pe_ch = get_embedder(7, 1, include_input=False)
        self.xyz_pe_fn, xyz_pe_ch = get_embedder(3, 3, include_input=False)
        # print(f"time pe dim = {time_pe_ch}, position pe dim = {xyz_pe_ch}")

        hidden = 64
        bias = False

        if not eval:
            self.timeFormer = TimeFormer(input_dims=hidden+time_pe_ch+xyz_pe_ch, output_dims=hidden, multires=4, nhead=2, hidden_dims=hidden, layers=2).to('cuda')
        else:
            self.timeFormer = None

        self.mlp_time = nn.Sequential(
            nn.Linear(self.total_feat_dim+time_pe_ch, hidden, bias=bias),
            nn.Tanh()
        ).cuda()

        self.mlp_deform = nn.Sequential(
            nn.Linear(hidden, hidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(hidden, 3, bias=bias),
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(hidden, hidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(hidden, 7, bias=bias),
        ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(3+1+hidden, hidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(hidden, 1, bias=bias),
            nn.Tanh()
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(3+1+hidden, hidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(hidden, 3, bias=bias),
            # nn.Sigmoid()
        ).cuda()

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def eval(self):
        if self.timeFormer is not None:
            self.timeFormer.eval()
        self.mlp_time.eval()
        self.mlp_deform.eval()
        self.mlp_cov.eval()
        self.mlp_opacity.eval()
        self.mlp_color.eval()

    def train(self):
        if self.timeFormer is not None:
            self.timeFormer.train()
        self.mlp_time.train()
        self.mlp_deform.train()
        self.mlp_cov.train()
        self.mlp_opacity.train()
        self.mlp_color.train()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self.xyz_activation(self._xyz)

    def get_time_features(self, timestamp, following=False, noise=0.):
        start_ = int(timestamp * self.time_line)
        # start_ = int((timestamp * 300) % 60)

        if not following or start_ > 1:
            feat = torch.cat(self._point_feats[start_:start_ + self.time_batch], dim=1)
        else:
            feat = [
                self._point_feats[start_].detach() if (start_ == 0 and i < 2) or (start_ == 1 and i == 0)
                else self._point_feats[start_ + i]
                for i in range(self.time_batch)
            ]
            feat = torch.cat(feat, dim=1)
        if noise:
            added_noise = torch.empty_like(feat).uniform_(-0.5, 0.5) * noise
            feat += added_noise
            return feat, added_noise
        return feat

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @staticmethod
    def get_quant_attr(attr_tensor, quantization):
        min_val = attr_tensor.min()
        max_val = attr_tensor.max()
        val_range = max_val - min_val
        # no division by zero
        if val_range == 0:
            val_range = 1
        attr_tensor_norm = (attr_tensor - min_val) / val_range
        qpow = 2 ** quantization
        attr_tensor_quantized = torch.round(attr_tensor_norm * qpow) / qpow
        attr_tensor = attr_tensor_quantized * val_range + min_val
        return attr_tensor.float()

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_time_mlp(self):
        return self.mlp_time

    @property
    def get_time_pe(self):
        return self.time_pe_fn

    @property
    def get_xyz_pe(self):
        return self.xyz_pe_fn

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # features_dc = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_dc = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(self.inverse_xyz_activation(fused_point_cloud).requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.timeFormer = None
        self.time_line = time_line
        self.time_embedding_num = time_line + self.time_batch - 1
        for i in range(3):
            point_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
            point_feat_param = nn.Parameter(point_feat.requires_grad_(True))
            self._point_feats.append(point_feat_param)
            self.register_parameter(f'_point_feat_{i}', point_feat_param)

    def create_from_ckpt(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = checkpoint['_xyz'].float().cuda()
        features_dc = checkpoint['_features_dc'].float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = checkpoint['_scaling'].float().cuda()
        rots = checkpoint['_rotation'].float().cuda()

        opacities = checkpoint['_opacity'].float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.time_line = time_line
        self.time_embedding_num = time_line + self.time_batch - 1
        for i in range(self.time_embedding_num):
            if i < (self.time_batch-1):
                point_feat = checkpoint[f'_point_feat_{i}'].float().cuda()
                point_feat_param = nn.Parameter(point_feat.requires_grad_(False))
            else:
                point_feat = checkpoint[f'_point_feat_{i}'].float().cuda() + torch.empty_like(self._point_feats[0]).uniform_(-0.5, 0.5) * 0.1
                point_feat_param = nn.Parameter(point_feat.requires_grad_(True))
            self._point_feats.append(point_feat_param)
            self.register_parameter(f'_point_feat_{i}', point_feat_param)

    def create_from_coarse(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = checkpoint['_xyz'].float().cuda()
        features_dc = checkpoint['_features_dc'].float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = checkpoint['_scaling'].float().cuda()
        rots = checkpoint['_rotation'].float().cuda()

        opacities = checkpoint['_opacity'].float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.time_line = time_line
        self.time_embedding_num = time_line + self.time_batch - 1
        for i in range(self.time_embedding_num):
            point_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
            point_feat_param = nn.Parameter(point_feat.requires_grad_(True))
            self._point_feats.append(point_feat_param)
            self.register_parameter(f'_point_feat_{i}', point_feat_param)

    def update_bound(self, verbose=None):
        x_bound_min = (torch.min(self.get_xyz, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self.get_xyz, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        # verbose.write(f'bound_updated, x_bound_min={x_bound_min}, x_bound_max={x_bound_max}')

    def print_gaussian_bound(self, verbose):
        xyz_bound_min = (torch.min(self._xyz, dim=0, keepdim=True)[0]).detach()
        xyz_bound_max = (torch.max(self._xyz, dim=0, keepdim=True)[0]).detach()

        scale_bound_min = (torch.min(self._scaling, dim=0, keepdim=True)[0]).detach()
        scale_bound_max = (torch.max(self._scaling, dim=0, keepdim=True)[0]).detach()

        rotation_bound_min = (torch.min(self._rotation, dim=0, keepdim=True)[0]).detach()
        rotation_bound_max = (torch.max(self._rotation, dim=0, keepdim=True)[0]).detach()

        opacity_bound_min = (torch.min(self._opacity, dim=0, keepdim=True)[0]).detach()
        opacity_bound_max = (torch.max(self._opacity, dim=0, keepdim=True)[0]).detach()

        fdc_bound_min = (torch.min(self._features_dc, dim=0, keepdim=True)[0]).detach()
        fdc_bound_max = (torch.max(self._features_dc, dim=0, keepdim=True)[0]).detach()

        stacked_point_feats = torch.stack(self._point_feats).detach()
        time_bound_min = torch.min(stacked_point_feats)
        time_bound_max = torch.max(stacked_point_feats)

        verbose.write(f'xyz_min={xyz_bound_min}, xyz_max={xyz_bound_max}')
        verbose.write(f'scale_min={scale_bound_min}, scale_max={scale_bound_max}')
        verbose.write(f'rotation_min={rotation_bound_min}, rotation_max={rotation_bound_max}')
        verbose.write(f'opacity_min={opacity_bound_min}, opacity_max={opacity_bound_max}')
        verbose.write(f'fdc_min={fdc_bound_min}, fdc_max={fdc_bound_max}')
        verbose.write(f'time_min={time_bound_min}, time_max={time_bound_max}')

    def query_encoded_xyz(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        return self.encoding_xyz(x)

    def query_time_feats(self, time_batch, following):
        feats = []
        feats_pe = []
        added_noises = []
        for timestamp in time_batch:
            p_time = torch.ones((self._xyz.shape[0], 1), device='cuda') * timestamp
            time_pe = self.get_time_pe(p_time)
            position_pe = self.get_xyz_pe(self._xyz.detach())
            feat, added_noise = self.get_time_features(timestamp, following, noise=0.001)
            feat = torch.cat([feat, time_pe], 1)
            feat = self.mlp_time(feat)
            # feat = torch.cat([feat, time_pe], 1)
            feats.append(feat)
            feat_pe = torch.cat([feat, time_pe, position_pe], 1)
            feats_pe.append(feat_pe)
            added_noises.append(added_noise)
        feats = torch.stack(feats)
        # feats = self.timeFormer(feats)
        feats_pe = torch.stack(feats_pe)
        delta_feats = self.timeFormer(feats_pe)
        # delta_feats = self.timeFormer(feats.clone())
        feats += delta_feats
        return feats, added_noises

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': self.mlp_time.parameters(), 'lr': training_args.offset_lr_init, "name": "mlp_time"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
        ]

        if self.timeFormer is not None:
            l.append({'params': self.timeFormer.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_time"},)

        for i in range(len(self._point_feats)):
            l.append({'params': [self._point_feats[i]], 'lr': training_args.feature_lr, "name": f"point_feat_{i}"})  # used in densification

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.temporal_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.temporal_feature_lr_init,
                                                           lr_final=training_args.temporal_feature_lr_final,
                                                           lr_delay_mult=training_args.temporal_feature_lr_delay_mult,
                                                           max_steps=training_args.temporal_feature_lr_steps)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                             lr_final=training_args.encoding_xyz_lr_final,
                                                             lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                             max_steps=training_args.encoding_xyz_lr_max_steps)

    def refine_training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        factor = 0.1
        l = [
            {'params': self.mlp_time.parameters(), 'lr': training_args.offset_lr_init*factor, "name": "mlp_time"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init*factor, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init*factor, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init*factor, "name": "mlp_color"},
            {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init*factor, "name": "mlp_deform"},
            # {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
            # {'params': self.timeFormer.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_time"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init*factor,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init*factor,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init*factor,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init*factor,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr_dict = {}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                lr_dict['xyz'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_time":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                lr_dict['transformer'] = lr
            # if "point_feat" in param_group["name"]:
            #     lr = self.temporal_feature_scheduler_args(iteration)
            #     param_group['lr'] = lr

        return lr_dict

    def save(self, path, name='model.pth', save_all=False, save_geometry_only=False):
        if save_all:
            torch.save(self.state_dict(), os.path.join(path, name))
        elif save_geometry_only:
            state_dict = self.state_dict()
            new_state_dict = {k: v for k, v in state_dict.items() if ('timeFormer' not in k) and ('mlp' not in k) and ('point_feat' not in k)}
            torch.save(new_state_dict, os.path.join(path, name))
        else:
            state_dict = self.state_dict()
            wo_timeformer = {k: v for k, v in state_dict.items() if 'timeFormer' not in k}
            torch.save(wo_timeformer, os.path.join(path, name))

    def save_mlps(self, path, name=''):
        torch.save(self.mlp_time.state_dict(), os.path.join(path, name + "mlp_time.pth"))
        torch.save(self.mlp_deform.state_dict(), os.path.join(path, name + "mlp_deform.pth"))
        torch.save(self.mlp_cov.state_dict(), os.path.join(path, name + "mlp_cov.pth"))
        torch.save(self.mlp_opacity.state_dict(), os.path.join(path, name + "mlp_opacity.pth"))
        torch.save(self.mlp_color.state_dict(), os.path.join(path, name + "mlp_color.pth"))

    def save_mlps_as_json(self, path, name):
        mlp_time_params = dict(self.mlp_time.named_parameters())
        mlp_deform_params = dict(self.mlp_deform.named_parameters())
        mlp_cov_params = dict(self.mlp_cov.named_parameters())
        mlp_opacity_params = dict(self.mlp_opacity.named_parameters())
        mlp_color_params = dict(self.mlp_color.named_parameters())

        mlp = {}
        for k, p in mlp_time_params.items():
            p_np = p.detach().cpu().numpy().T
            mlp[f'time.{k}'] = p_np.tolist()
        for k, p in mlp_deform_params.items():
            p_np = p.detach().cpu().numpy().T
            mlp[f'deform.{k}'] = p_np.tolist()
        for k, p in mlp_cov_params.items():
            p_np = p.detach().cpu().numpy().T
            mlp[f'cov.{k}'] = p_np.tolist()
        for k, p in mlp_opacity_params.items():
            p_np = p.detach().cpu().numpy().T
            mlp[f'opacity.{k}'] = p_np.tolist()
        for k, p in mlp_color_params.items():
            p_np = p.detach().cpu().numpy().T
            mlp[f'color.{k}'] = p_np.tolist()

        mlp_file = os.path.join(path, name)
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)

    def replace_tensors_to_optimizer(self, inds=None):
        """
        Error: TypeError: 'NoneType' object is not subscriptable
        camera.time有问题，检查__init__.py / neural_3D_dataset_NDC.py / dataset.py
        """
        tensors_dict = {"xyz": self._xyz,
                        "f_dc": self._features_dc,
                        "opacity": self._opacity,
                        "scaling" : self._scaling,
                        "rotation" : self._rotation}
        for i in range(self.time_embedding_num):
            tensors_dict[f"point_feat_{i}"] = self._point_feats[i]

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            if len(group["params"]) > 1 : continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.opacity_accum = self.opacity_accum[valid_points_mask]

        point_feats = []
        for i in range(len(self._point_feats)):
            param = optimizable_tensors[f"point_feat_{i}"]
            point_feats.append(param)
            self.register_parameter(f'_point_feat_{i}', param)
        self._point_feats = point_feats

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_point_feats):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation
             }

        for i in range(len(self._point_feats)):
            d[f"point_feat_{i}"] = new_point_feats[i]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        point_feats = []
        for i in range(len(self._point_feats)):
            param = optimizable_tensors[f"point_feat_{i}"]
            point_feats.append(param)
            self.register_parameter(f'_point_feat_{i}', param)
        self._point_feats = point_feats

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        num_preserved = -1
        num_removed = -1
        tmp = 0
        while num_preserved < 0 or num_removed < 0:
            selected_pts_mask = torch.where(padded_grad >= (grad_threshold - tmp * 0.000005), True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            num_gaussians = n_init_points + torch.sum(selected_pts_mask).detach().cpu()
            if tmp == 0:
                self.grid_sidelen = math.ceil(np.sqrt(num_gaussians))
            num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen
            num_preserved = torch.sum(selected_pts_mask) - num_removed
            tmp += 1
        assert num_preserved >= 0, "The newly added Gaussian is not enough to be removed"

        selected_padded_grad = padded_grad[selected_pts_mask]
        sorted_indices = torch.argsort(selected_padded_grad, descending=True)
        top_indices = sorted_indices[:num_preserved]
        selected_global_indices = torch.nonzero(selected_pts_mask, as_tuple=True)[0][top_indices]
        selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        selected_pts_mask[selected_global_indices] = True

        # if not selected_pts_mask.any():
        #     return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = self.inverse_xyz_activation(torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)  # (N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features_rest = None
        new_point_feats = []
        for i in range(len(self._point_feats)):
            new_point_feats.append(self._point_feats[i][selected_pts_mask].repeat(N,1))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_point_feats)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        assert self._xyz.shape[0] - self.grid_sidelen * self.grid_sidelen == 0, f"The number of gaussian is {self._xyz.shape[0]} while grid_size is {self.grid_sidelen*self.grid_sidelen}"

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_features_rest = None
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_point_feats = []
        for i in range(len(self._point_feats)):
            new_point_feats.append(self._point_feats[i][selected_pts_mask])

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_point_feats)

    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]

        return final_point, mask_d    

    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)

        new_features_dc = self._features_dc[selected_pts_mask][mask]
        # new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_features_rest = None
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_bias = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return selected_xyz, new_xyz

    @staticmethod
    def _sample_alives(probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return new_opacity, new_scaling

    def relocate_gs(self, max_grad, min_opacity, extent, logger=None):
        dead_mask = (self.opacity_accum < min_opacity * self.denom).squeeze()
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # logger.write(f"relocate gs number = {dead_indices.shape[0]}")

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0])
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._opacity[dead_indices],
            self._scaling[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)
        self._xyz[dead_indices] = self._xyz[reinit_idx]
        self._features_dc[dead_indices] = self._features_dc[reinit_idx]
        self._rotation[dead_indices] = self._rotation[reinit_idx]
        for i in range(self.time_embedding_num):
            self._point_feats[i][dead_indices] = self._point_feats[i][reinit_idx]

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        optimizable_tensors = self.replace_tensors_to_optimizer(inds=reinit_idx)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        point_feats = []
        for i in range(len(self._point_feats)):
            param = optimizable_tensors[f"point_feat_{i}"]
            point_feats.append(param)
            self.register_parameter(f'_point_feat_{i}', param)
        self._point_feats = point_feats

        torch.cuda.empty_cache()
        return dead_mask.sum()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = (self.opacity_accum < min_opacity * self.denom).squeeze()
        self.prune_points(prune_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_opacity_stats(self, opacity, update_filter, min_opacity):
        temp_opacity = opacity.detach().clone()
        temp_opacity[temp_opacity < min_opacity] = 0
        self.opacity_accum[update_filter] += temp_opacity[update_filter]

    def add_noise(self, noise_lr):
        L = build_scaling_rotation(self.get_scaling, self.get_rotation)
        actual_cov = L @ L.transpose(1, 2)

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = torch.randn_like(self._xyz * (op_sigmoid(1 - self.get_opacity)) * noise_lr)
        noise = torch.bmm(actual_cov, noise.unsqueeze(-1)).squeeze(-1)
        self._xyz.add_(noise)

    ###  self organizing gaussian
    def prune_all_but_these_indices(self, indices):
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(indices)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[indices]
            self.denom = self.denom[indices]
            self.max_radii2D = self.max_radii2D[indices]
            self.opacity_accum = self.opacity_accum[indices]

            point_feats = []
            for i in range(len(self._point_feats)):
                param = optimizable_tensors[f"point_feat_{i}"]
                point_feats.append(param)
                self.register_parameter(f'_point_feat_{i}', param)
            self._point_feats = point_feats
        else:
            self._xyz = self._xyz[indices]
            self._features_dc = self._features_dc[indices]
            self._opacity = self._opacity[indices]
            self._scaling = self._scaling[indices]
            self._rotation = self._rotation[indices]

            point_feats = []
            for i in range(len(self._point_feats)):
                param = self._point_feats[i][indices]
                point_feats.append(param)
                self.register_parameter(f'_point_feat_{i}', param)
            self._point_feats = point_feats

    def prune_to_square_shape(self, sort_by_opacity=True, verbose=None):
        num_gaussians = self._xyz.shape[0]

        self.grid_sidelen = int(np.sqrt(num_gaussians))
        num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen

        if verbose is not None:
            verbose.write(f"Removing {num_removed}/{num_gaussians} gaussians to fit the grid. ({100 * num_removed / num_gaussians:.4f}%)")
        if self.grid_sidelen * self.grid_sidelen < num_gaussians:
            if sort_by_opacity:
                alpha = self.get_opacity[:, 0]
                _, keep_indices = torch.topk(alpha, k=self.grid_sidelen * self.grid_sidelen)
            else:
                shuffled_indices = torch.randperm(num_gaussians)
                keep_indices = shuffled_indices[:self.grid_sidelen * self.grid_sidelen]
            sorted_keep_indices = torch.sort(keep_indices)[0]
            self.prune_all_but_these_indices(sorted_keep_indices)

    def as_grid_img(self, gs_attr):
        if not hasattr(self, "grid_sidelen"):
            raise "Gaussians not pruned yet!"

        if self.grid_sidelen * self.grid_sidelen != gs_attr.shape[0]:
            raise "Tensor shape does not match img sidelen, needs pruning?"

        img = gs_attr.reshape((self.grid_sidelen, self.grid_sidelen, -1))
        return img

    def reshape_as_grid_img(self, gs_attr, grid_sidelen):
        self.grid_sidelen = grid_sidelen
        return gs_attr.reshape((grid_sidelen, grid_sidelen, -1))

    def sort_into_grid(self, sorting_cfg, verbose):
        normalization_fn = self.normalize if sorting_cfg.sorting_normalize else lambda x: x

        params_to_sort = []
        for attr_name, attr_weight in sorting_cfg.sorting_weights.items():
            if attr_weight > 0:
                if attr_name == 'point_feat':
                    gs_attr = getattr(self, f"_{attr_name}_0")
                else:
                    gs_attr = getattr(self, f"_{attr_name}")
                params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
        params_to_sort = torch.cat(params_to_sort, dim=1)

        if sorting_cfg.sorting_shuffle:
            shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)
            params_to_sort = params_to_sort[shuffled_indices]

        grid_to_sort = self.as_grid_img(params_to_sort).permute(2, 0, 1)
        _, sorted_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
        sorted_indices = sorted_indices.squeeze().flatten()

        if sorting_cfg.sorting_shuffle:
            sorted_indices = shuffled_indices[sorted_indices.long()]

        self.prune_all_but_these_indices(sorted_indices)

    @staticmethod
    def normalize(tensor):
        tensor = tensor - tensor.mean()
        if tensor.std() > 0:
            tensor = tensor / tensor.std()
        return tensor

    def neighborloss_2d(self, attr_name, neighbor_cfg):
        if attr_name == 'point_feat':
            gs_attr = getattr(self, f"_{attr_name}_0")
        else:
            gs_attr = getattr(self, f"_{attr_name}")

        if neighbor_cfg.normalize:
            gs_attr = self.normalize(gs_attr)

        attr_img = self.as_grid_img(gs_attr)
        attr_img = attr_img.permute(2, 0, 1).unsqueeze(0)

        blurred_x = kornia.filters.gaussian_blur2d(
            attr_img.detach(),
            kernel_size=(1, neighbor_cfg.kernel_size),
            sigma=(neighbor_cfg.sigma, neighbor_cfg.sigma),
            border_type="circular",
        )

        blurred_xy = kornia.filters.gaussian_blur2d(
            blurred_x,
            kernel_size=(neighbor_cfg.kernel_size, 1),
            sigma=(neighbor_cfg.sigma, neighbor_cfg.sigma),
            border_type="reflect",
        )

        return F.huber_loss(blurred_xy, attr_img)

    def frame_maker(self, img):
        xn = 4
        yn = self.feat_dim // xn
        frame = torch.zeros((self.grid_sidelen*xn, self.grid_sidelen*yn, 1), device='cuda')
        x, y = 0, 0
        for i in range(img.shape[2]):
            if y >= frame.shape[1]:
                y = 0
                x += self.grid_sidelen
            assert x + self.grid_sidelen <= frame.shape[0], f"Frame_maker: not enough space, needs {x + self.grid_sidelen}x{y} but only have {self.grid_sidelen*4}"

            frame[x:x+self.grid_sidelen, y:y+self.grid_sidelen, 0] = img[:, :, i]
            y += self.grid_sidelen
        return frame

    def attr_as_grid_img(self, attr_name):
        if '_' not in attr_name:
            attr_name = '_' + attr_name
        if 'point_feat' not in attr_name:
            gs_attr = getattr(self, f"{attr_name}")
            return [self.as_grid_img(gs_attr)]
        else:
            imgs = []
            for i in range(len(self._point_feats)):
                # gs_attr = getattr(self, f"{attr_name}_{i}")
                gs_attr = self._point_feats[i]
                img = self.as_grid_img(gs_attr)  # grid_sidelen, grid_sidelen, 16
                img = self.frame_maker(img)  # grid_sidelen * 4, grid_sidelen * 4
                imgs.append(img)
            return imgs

    def set_attr_from_grid_img(self, attr_name, img, original_gaussian=None):
        if self.optimizer is not None:
            raise "Overwriting Gaussians during training not implemented yet! - Consider pruning method implementations"

        attr_shapes = {
            "_xyz": (3,),
            "_features_dc": (3,),
            "_rotation": (4,),
            "_scaling": (3,),
            "_opacity": (1,),
        }

        target_shape = attr_shapes[attr_name]
        img_shaped = img.reshape(-1, *target_shape)
        tensor = torch.tensor(img_shaped, dtype=torch.float, device="cuda")
        # if original_gaussian is not None:
        #     mse = ((tensor - getattr(original_gaussian, f"{attr_name}")) ** 2).mean()
        #     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
        #     print(f'Compressed {attr_name} loss is {mse}, PSNR is {psnr} \n')
        setattr(self, attr_name, tensor)

    def set_point_feat_from_grid_img(self, idx, img, original_gaussian):
        # if self.optimizer is not None:
        #     raise "Overwriting Gaussians during training not implemented yet! - Consider pruning method implementations"
        self.grid_sidelen = img.shape[1] // (self.feat_dim // 4)
        feats = np.zeros((self.grid_sidelen, self.grid_sidelen, self.feat_dim))
        x, y = 0, 0
        for i in range(self.feat_dim):
            if y >= img.shape[1]:
                y = 0
                x += self.grid_sidelen
            assert x + self.grid_sidelen <= img.shape[0], f"Tile_maker: not enough space, needs {x + self.grid_sidelen}x{y} but only have {self.grid_sidelen * 4}"
            feats[:, :, i] = img[x:x+self.grid_sidelen, y:y+self.grid_sidelen]
            y += self.grid_sidelen
        feats = feats.reshape(-1, self.feat_dim)
        tensor = torch.tensor(feats, dtype=torch.float, device="cuda")
        # if original_gaussian is not None and idx == 1:
        #     mse = ((tensor - original_gaussian._point_feats[idx]) ** 2).mean()
        #     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
        #     print(f'Compressed feature map loss is {mse}, PSNR is {psnr} \n')
        setattr(self, f'_point_feat_{idx}', tensor)
        self._point_feats.append(tensor)

    @staticmethod
    def get_geo_attr_from_grid_img(attr_name, img):
        attr_shapes = {
            "_xyz": (3,),
            "_features_dc": (3,),
            "_rotation": (4,),
            "_scaling": (3,),
            "_opacity": (1,),
        }

        target_shape = attr_shapes[attr_name]
        img_shaped = img.reshape(-1, *target_shape)
        tensor = torch.tensor(img_shaped, dtype=torch.float, device="cuda")
        return tensor

    def set_geo_attr_from_grid_img(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'point_feat' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(False))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.empty_cache()

