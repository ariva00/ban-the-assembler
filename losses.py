import torch


def _soft_histogram(values: torch.Tensor, weights: torch.Tensor, bin_centers: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """
    values:      (P,) flattened per-pixel channel values in [0, 1]
    weights:     (P,) per-pixel soft foreground weight in [0, 1]
    bin_centers: (n_bins,)
    Returns (n_bins,) weighted histogram, normalized to sum to 1.
    """
    diff   = values.unsqueeze(-1) - bin_centers.unsqueeze(0)         # (P, n_bins)
    assign = torch.exp(-0.5 * (diff / bandwidth) ** 2)               # soft (Gaussian) bin membership
    hist   = (assign * weights.unsqueeze(-1)).sum(dim=0)             # (n_bins,)
    return hist / (hist.sum() + 1e-8)


def color_histogram_loss(
    image:             torch.Tensor,   # (H, W, 3) rendered image, in [0, 1]
    fg_weight:         torch.Tensor,   # (H, W) render-side foreground weight (e.g. 1 - bg_prob)
    target:            torch.Tensor,   # (H, W, 3) target image, in [0, 1]
    target_fg_weight:  torch.Tensor,   # (H, W) target-side foreground weight (e.g. 1 - target_masks)
    n_bins:            int = 16,
    bandwidth:         float | None = None,
) -> torch.Tensor:
    """
    Per-channel L1 distance between soft, foreground-weighted color histograms of
    the render and the target. Unlike a pixel-aligned loss (recon_loss), this
    only cares whether the *distribution* of colors present matches — so it
    stays informative even while object positions/silhouettes (and therefore
    per-pixel alignment) are still wrong, which is exactly the regime where
    recon_loss's alpha/bg_color blending lets objects hide by going white
    (see optimize_scene_multiview.py's color-collapse investigation).

    Background pixels are excluded via fg_weight/target_fg_weight (both sides
    are already white/white, so including them would dilute the one thing this
    loss is meant to check: does the object color distribution match).
    """
    device = image.device
    bin_centers = torch.linspace(0.0, 1.0, n_bins, device=device)
    if bandwidth is None:
        bandwidth = 1.0 / n_bins

    fg_flat        = fg_weight.reshape(-1)
    target_fg_flat = target_fg_weight.reshape(-1)

    loss = image.new_zeros(())
    for c in range(3):
        render_hist = _soft_histogram(image[..., c].reshape(-1), fg_flat, bin_centers, bandwidth)
        target_hist = _soft_histogram(target[..., c].reshape(-1), target_fg_flat, bin_centers, bandwidth)
        loss = loss + (render_hist - target_hist).abs().sum()
    return loss / 3
