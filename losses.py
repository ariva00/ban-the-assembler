import torch
import torch.nn.functional as F

from misc import _rotvec_to_matrix


def scene_attribute_loss(
    pred_flat:   torch.Tensor,   # (..., latent_dim + 9): [embedding | translation | rot_vec | color]
    target_flat: torch.Tensor,   # same shape/layout as pred_flat
    latent_dim:  int,
    weights:     dict[str, float] | None = None,
) -> torch.Tensor:
    """
    MSE between two flattened SceneObject vectors (any leading batch shape,
    e.g. (n_obj, D) or (V, H, W, D) for a dense per-pixel prediction), split
    into embedding/translation/rotation/color and each averaged separately
    before combining -- a single flat MSE over the concatenated vector lets
    whichever component has the largest natural scale dominate the gradient
    and starve the others.

    Rotation specifically is compared as rotation matrices, not raw rot_vec
    components: axis-angle MSE is discontinuous at the angle-pi wraparound
    (two rotations that are nearly identical can have rot_vecs that are far
    apart), and its raw range (~+-pi) is much wider than embedding/color/
    translation's, so it would otherwise dominate on both fronts at once.
    Rotation matrix entries are bounded in [-1, 1], which incidentally also
    brings it back to a comparable scale to the other components -- equal
    weights (the default) are a reasonable starting point *because* of that,
    not despite it.
    """
    d = latent_dim
    emb_p,   emb_t   = pred_flat[..., :d],      target_flat[..., :d]
    trans_p, trans_t = pred_flat[..., d:d+3],   target_flat[..., d:d+3]
    rot_p,   rot_t   = pred_flat[..., d+3:d+6], target_flat[..., d+3:d+6]
    color_p, color_t = pred_flat[..., d+6:d+9], target_flat[..., d+6:d+9]

    w = weights or {}
    emb_loss   = F.mse_loss(emb_p, emb_t)
    trans_loss = F.mse_loss(trans_p, trans_t)
    rot_loss   = F.mse_loss(_rotvec_to_matrix(rot_p), _rotvec_to_matrix(rot_t))
    color_loss = F.mse_loss(color_p, color_t)

    return (w.get("embedding", 1.0)   * emb_loss +
            w.get("translation", 1.0) * trans_loss +
            w.get("rotation", 1.0)    * rot_loss +
            w.get("color", 1.0)       * color_loss)


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
