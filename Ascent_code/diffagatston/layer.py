"""
diffagatston/layer.py
---------------------
The differentiable Agatston surrogate, validated to agree with the standard
Agatston score (Spearman ~0.997, Pearson(log) ~0.993 on COCA GT masks).

Two uses:
  1. As a loss term: feed the SOFT prediction (prob in [0,1]) + RAW HU ->
     get a differentiable score estimate -> compare to the case-level GT score.
  2. calibrate_alpha(): least-squares fit of the global scale on a set of
     (pred_or_gt_mask, hu, true_score) so the surrogate is on the right scale.

IMPORTANT: always feed RAW Hounsfield Units here, never the normalized input.
"""

import torch
import torch.nn as nn


class DiffAgatston(nn.Module):
    def __init__(self, tau_g=0.10, tau_w=0.08, alpha=1.60,
                 pixel_area=None, slice_thickness=3.0):
        """
        tau_g, tau_w   : sigmoid sharpness for HU gate / density staircase
                         (chosen on COCA; method is robust to these).
        alpha          : global calibration constant (re-fit per dataset).
        pixel_area     : mm^2 per pixel = spacing_x * spacing_y. If None it must
                         be passed at forward time (volumes can differ).
        slice_thickness: mm; standard Agatston normalizes to 3 mm slices.
        """
        super().__init__()
        self.tau_g = tau_g
        self.tau_w = tau_w
        # alpha kept as a buffer so it moves with the module / is saved
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.pixel_area = pixel_area
        self.slice_thickness = slice_thickness

    def _per_voxel(self, hu):
        """soft gate * soft staircase, the density-weighted indicator."""
        g = torch.sigmoid(self.tau_g * (hu - 130.0))
        w = (1.0
             + torch.sigmoid(self.tau_w * (hu - 200.0))
             + torch.sigmoid(self.tau_w * (hu - 300.0))
             + torch.sigmoid(self.tau_w * (hu - 400.0)))
        return g * w

    def forward(self, prob, hu, pixel_area=None, reduce_dims=None):
        """
        prob : soft calcium probability, any shape (B,1,H,W) or (H,W)...
        hu   : RAW HU, same shape as prob
        pixel_area : mm^2 per pixel; falls back to self.pixel_area
        reduce_dims: dims to sum over. Default = all but batch dim 0 if prob is
                     4D (B,1,H,W); otherwise sum over everything.
        returns: differentiable score. scalar, or (B,) if batched.
        """
        pa = pixel_area if pixel_area is not None else self.pixel_area
        if pa is None:
            raise ValueError("pixel_area must be provided (spacing_x*spacing_y).")
        slice_w = self.slice_thickness / 3.0
        contrib = prob * self._per_voxel(hu) * pa * slice_w

        if reduce_dims is None:
            if contrib.dim() == 4:           # (B,1,H,W) -> per-sample score
                reduce_dims = (1, 2, 3)
            else:
                reduce_dims = tuple(range(contrib.dim()))
        return self.alpha * contrib.sum(dim=reduce_dims)

    @torch.no_grad()
    def set_alpha(self, value):
        self.alpha.fill_(float(value))


def calibrate_alpha(diff_layer, masks, hus, true_scores, pixel_areas):
    """Least-squares: true ~ alpha * raw_diff.  Lists/iterables of per-case
    tensors. Returns the fitted alpha (float). Run once with alpha=1 effect by
    using raw contributions."""
    raw, tgt = [], []
    old_alpha = float(diff_layer.alpha)
    diff_layer.set_alpha(1.0)
    with torch.no_grad():
        for m, hu, s, pa in zip(masks, hus, true_scores, pixel_areas):
            raw.append(float(diff_layer(m, hu, pixel_area=pa)))
            tgt.append(float(s))
    diff_layer.set_alpha(old_alpha)
    raw = torch.tensor(raw); tgt = torch.tensor(tgt)
    alpha = float((raw * tgt).sum() / (raw * raw).sum())
    return alpha


if __name__ == "__main__":
    # tiny sanity check
    layer = DiffAgatston()
    prob = torch.zeros(1, 1, 16, 16)
    hu = torch.full((1, 1, 16, 16), -50.0)
    prob[0, 0, 4:8, 4:8] = 1.0
    hu[0, 0, 4:8, 4:8] = 350.0          # 16 bright calcium voxels
    score = layer(prob, hu, pixel_area=0.1466)
    print("toy differentiable score:", score.item())