# utils.py  ────────────────────────────────────────────────────────────
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from monai.metrics import SSIMMetric

# SSIM for 3-D volumes in [0, 1]
_ssim3d = SSIMMetric(data_range=1.0, spatial_dims=3, reduction="mean")


def mae_psnr_ssim(
    prediction:    np.ndarray,
    ground_truth:  np.ndarray,
    mask:          np.ndarray | None = None,
):
    """
    MAE, PSNR, SSIM between `prediction` and `ground_truth`,
    optionally inside a binary `mask`.

    Inputs must be 3-D numpy arrays (D, H, W) already scaled to [0, 1].

    Works equally for CT ↔ MRI or any other modality.
    """
    p = prediction.copy()
    g = ground_truth.copy()

    if mask is not None:
        p[~mask] = 0
        g[~mask] = 0

    mae  = np.mean(np.abs(p - g))
    psnr = compare_psnr(g, p, data_range=1.0)

    # MONAI expects 5-D [B, C, D, H, W]
    p_t = torch.from_numpy(p[None, None]).float()
    g_t = torch.from_numpy(g[None, None]).float()
    with torch.no_grad():
        ssim = float(_ssim3d(p_t, g_t))

    return mae, psnr, ssim
