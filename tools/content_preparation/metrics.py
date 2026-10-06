"""Image metrics and a separate planner-distortion aggregation strategy.

RGB inputs are float [0,1]; LPIPS receives the standard [-1,1] input convention.
MSE is the default planner signal. PSNR is reported as its logarithmic transform;
LPIPS is a reporting signal unless the caller explicitly chooses another strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .paths import configure_upstream_imports


@dataclass(frozen=True)
class MetricResult:
    mse: float
    psnr: float
    lpips: float | None
    ssim: float | None = None

    def to_dict(self):
        result = {"mse": float(self.mse), "psnr": "inf" if math.isinf(self.psnr) else float(self.psnr),
                  "lpips": None if self.lpips is None else float(self.lpips)}
        if self.ssim is not None:
            result["ssim"] = float(self.ssim)
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(float(value["mse"]), float(value["psnr"]),
                   None if value.get("lpips") is None else float(value["lpips"]),
                   None if value.get("ssim") is None else float(value["ssim"]))


def _image(image):
    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3 or not value.size:
        raise ValueError("Metric images require nonempty H x W x 3 RGB arrays.")
    if not np.all(np.isfinite(value)) or np.min(value) < 0 or np.max(value) > 1:
        raise ValueError("Metric RGB images must be finite floats in [0,1].")
    return value


class MetricEvaluator:
    """Sequential LPIPS evaluation with pretrained weights and batch size one.

    CPU is the default so a VGG model never competes with a Gaussian model for
    the 6 GB GPU. Explicit CUDA LPIPS moves the model to GPU for one comparison
    and back to CPU before returning. Missing weights raise an actionable error.
    """
    def __init__(self, config: Mapping | None = None):
        self.config = dict(config or {})
        lpips_config = self.config.get("lpips", {})
        if isinstance(lpips_config, bool):
            lpips_config = {"enabled": lpips_config}
        self.lpips_config = dict(lpips_config)
        self.enabled = self.lpips_config.get("enabled", self.config.get("lpips_enabled", True))
        self.net = self.lpips_config.get("net", self.config.get("lpips_net", "vgg"))
        self.device = self.lpips_config.get("device", self.config.get("device", "cpu"))
        batch = self.lpips_config.get("batch_size", self.config.get("batch_size", 1))
        if batch != 1:
            raise ValueError("Content preparation requires LPIPS batch_size=1 for sequential execution.")
        if self.net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("LPIPS net must be alex, vgg, or squeeze.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("LPIPS device must be cpu or cuda.")
        self._criterion = None
        self.backend = None

    def _load_lpips(self):
        if self._criterion is not None:
            return self._criterion
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("LPIPS requires PyTorch in the Dynamic-LapisGS environment.") from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LPIPS device=cuda requires accessible CUDA; set metrics.lpips.device=cpu to evaluate serially on CPU.")
        allow_download = bool(self.lpips_config.get("allow_download", False))
        cache = Path(torch.hub.get_dir())/"checkpoints"
        backbone_names = {"vgg": "vgg16-397923af.pth", "alex": "alexnet-owt-7be5be79.pth", "squeeze": "squeezenet1_1-b8a52dc0.pth"}
        # Repo lpipsPyTorch uses torchvision pretrained backbones plus the official
        # learned linear weights cached under <net>.pth. Check first to avoid an
        # accidental network download in a reproducible offline preparation job.
        required = [cache/backbone_names[self.net], cache/f"{self.net}.pth"]
        if not allow_download and any(not path.is_file() for path in required):
            missing = ", ".join(str(path) for path in required if not path.is_file())
            raise RuntimeError(f"Pretrained LPIPS weights are missing: {missing}. Populate the torchvision/official LPIPS cache, or explicitly enable metrics.lpips.allow_download. LPIPS is never replaced by a synthetic score.")
        try:
            configure_upstream_imports()
            from lpipsPyTorch.modules.lpips import LPIPS
            self._criterion = LPIPS(net_type=self.net, version="0.1").cpu().eval()
            self.backend = "repo_lpipsPyTorch_pretrained_0.1"
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError("Cannot initialize pretrained repo LPIPS. Use the project environment with torchvision and cached official backbone/linear weights.") from exc
        return self._criterion

    def compare(self, image, reference) -> MetricResult:
        image, reference = _image(image), _image(reference)
        if image.shape != reference.shape:
            raise ValueError("Metric image/reference dimensions must match.")
        mse = float(np.mean((image.astype(np.float64)-reference.astype(np.float64))**2))
        psnr = math.inf if mse == 0 else -10*math.log10(mse)
        lpips_value = None
        ssim_value = None
        if self.enabled or self.config.get("ssim", False):
            import torch
            # VGG/AlexNet require a minimum spatial extent. Resizing is opt-in:
            # smaller renders fail rather than silently alter the methodology.
            minimum = 32 if self.net in {"vgg", "squeeze"} else 64
            if self.enabled and min(image.shape[:2]) < minimum:
                raise ValueError(f"LPIPS {self.net} requires render dimensions >= {minimum}; increase configured resolution.")
            a = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)
            b = torch.from_numpy(reference.transpose(2, 0, 1).copy()).unsqueeze(0)
            if self.enabled:
                criterion = self._load_lpips()
                if self.device == "cuda":
                    from .renderer_adapter import GPUExecutionGuard, cleanup_gpu
                    with GPUExecutionGuard():
                        try:
                            criterion = criterion.cuda()
                            with torch.no_grad():
                                lpips_value = float(criterion(a.cuda()*2-1, b.cuda()*2-1).item())
                        except torch.cuda.OutOfMemoryError as exc:
                            raise RuntimeError("LPIPS CUDA out of memory at batch_size=1. Set metrics.lpips.device=cpu or reduce configured render resolution; no automatic change was made.") from exc
                        finally:
                            self._criterion = criterion.cpu()
                            cleanup_gpu()
                else:
                    with torch.no_grad():
                        lpips_value = float(criterion(a*2-1, b*2-1).item())
            if self.config.get("ssim", False):
                configure_upstream_imports()
                from utils.loss_utils import ssim
                with torch.no_grad():
                    ssim_value = float(ssim(a[0], b[0]).item())
            del a, b
        return MetricResult(mse, psnr, lpips_value, ssim_value)

    def metadata(self):
        return {"rgb_range": [0, 1], "mse_range": "full_image_RGB_mean", "psnr_peak": 1,
                "lpips_enabled": bool(self.enabled), "lpips_net": self.net, "lpips_version": "0.1",
                "lpips_input_range": [-1, 1], "lpips_batch_size": 1, "lpips_device": self.device,
                "lpips_backend": self.backend, "planner_default_distortion": "mse"}

    def close(self):
        self._criterion = None
        from .renderer_adapter import cleanup_gpu
        cleanup_gpu()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def aggregate(values: Sequence[float], strategy="uniform_mean", weights: Sequence[float] | None = None) -> float:
    """Aggregate a reporting signal without discarding the raw view records."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(np.isnan(values)):
        raise ValueError("Aggregation needs a nonempty one-dimensional sequence without NaNs.")
    strategy = {"mean": "uniform_mean", "max": "worst", "worst_case": "worst"}.get(strategy, strategy)
    if strategy == "uniform_mean":
        return float(values.mean())
    if strategy == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean needs explicit nonnegative weights.")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != values.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Weights must match values, be finite/nonnegative, and have positive sum.")
        positive = weights > 0
        return float(np.average(values[positive], weights=weights[positive]))
    if strategy == "median":
        return float(np.median(values))
    if strategy == "p95":
        # Explicit endpoint handling prevents NumPy's inf-inf interpolation NaN.
        position = .95*(len(values)-1)
        ordered = np.sort(values)
        lower, upper = ordered[int(math.floor(position))], ordered[int(math.ceil(position))]
        if math.isinf(float(upper)):
            return float(upper)
        return float(lower+(upper-lower)*(position-math.floor(position)))
    if strategy == "worst":
        return float(values.max())
    raise ValueError(f"Unknown aggregation strategy: {strategy}")


class DistortionAggregator:
    """Configurable distortion strategy; no implicit MSE + PSNR + LPIPS sum."""
    def __init__(self, metric="mse", strategy="uniform_mean", *, composite_weights=None, normalizers=None):
        self.metric, self.strategy = metric, strategy
        self.composite_weights, self.normalizers = composite_weights, normalizers
        if metric == "composite" and (not composite_weights or not normalizers):
            raise ValueError("Composite distortion requires explicit weights and normalizers.")
        if metric not in {"mse", "lpips", "composite"}:
            raise ValueError("Planner distortion supports mse, lpips, or an explicitly configured composite.")
        if metric == "composite":
            for name, weight in composite_weights.items():
                if name not in {"mse", "lpips"} or not math.isfinite(float(weight)) or float(weight) < 0:
                    raise ValueError("Composite weights require nonnegative finite MSE/LPIPS entries.")
                if name not in normalizers or not math.isfinite(float(normalizers[name])) or float(normalizers[name]) <= 0:
                    raise ValueError("Each composite signal requires a finite positive normalizer.")

    def distortion(self, metrics):
        metrics = metrics.to_dict() if isinstance(metrics, MetricResult) else metrics
        if self.metric == "composite":
            return sum(float(metrics[name])*float(weight)/float(self.normalizers[name])
                       for name, weight in self.composite_weights.items())
        if metrics.get(self.metric) is None:
            raise ValueError(f"Metric {self.metric} was not evaluated.")
        return float(metrics[self.metric])

    def __call__(self, results, weights=None):
        return aggregate([self.distortion(result) for result in results], self.strategy, weights)
