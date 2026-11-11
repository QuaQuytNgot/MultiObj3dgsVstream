import torch
from torch import nn


class TimeFormer(nn.Module):

    def __init__(self, input_dims=48, output_dims=48, multires=4, nhead=6, hidden_dims=48, layers=1):
        super(TimeFormer, self).__init__()
        # self.embed_fn, self.embed_ch = get_embedder(multires=multires, input_dims=input_dims)
        self.layers = layers
        self.input_ch = input_dims

        # self.encoder_layer_0 = nn.TransformerEncoderLayer(d_model=self.input_ch, nhead=nhead,
        #                                                   dim_feedforward=hidden_dims,
        #                                                   activation=nn.functional.tanh)
        # self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.input_ch, nhead=nhead,
        #                                                   dim_feedforward=hidden_dims,
        #                                                   activation=nn.functional.tanh)
        # self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.input_ch, nhead=nhead,
        #                                                   dim_feedforward=hidden_dims,
        #                                                   activation=nn.functional.tanh)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_ch, nhead=nhead, dropout=0.1,
                                                   dim_feedforward=hidden_dims,
                                                   activation=nn.functional.tanh)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        self.output_layer = nn.Linear(self.input_ch, output_dims)
        self.act = nn.functional.tanh

    def forward(self, h):
        # src_embed = self.embed_fn(src)
        # h = self.encoder_layer_0(h)
        # if self.layers > 1:
        #     h = self.encoder_layer_1(h)
        # if self.layers > 2:
        #     h = self.encoder_layer_2(h)
        h = self.transformer_encoder(h)
        h = self.output_layer(h)
        h = self.act(h)
        return h