"""Chip preprocessing shared by embedding and fine-tuning: percentile stretch and a 5-tap blur."""
import torch


def stretch(x):
    """per-chip 2-98 % stretch to [0, 1]: removes scene-wide exposure and haze differences"""
    lo = torch.quantile(x.flatten(1), 0.02, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(x.flatten(1), 0.98, dim=1).view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo + 1)).clamp(0, 1)


def blur(x):
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device)
    k = (k[:, None] * k[None, :]) / 256
    return torch.nn.functional.conv2d(x, k.expand(3, 1, 5, 5), padding=2, groups=3)
