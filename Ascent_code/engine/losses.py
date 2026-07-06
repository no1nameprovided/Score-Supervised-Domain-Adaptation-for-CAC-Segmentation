"""
engine/losses.py
----------------
Segmentation loss for the supervised (COCA) branch.

Calcium voxels are extremely sparse, so plain BCE collapses to predicting
all-background. DiceFocalLoss combines Dice (handles imbalance via overlap)
with Focal (down-weights easy background). This is L_seg in the paper.

Later stages add: pseudo-consistency, DiffAgatston score loss, priors,
ranking. Those go in separate files; this one stays simple.
"""

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # sigmoid=True -> single-channel binary; include_background for 1-ch.
        self.loss = DiceFocalLoss(
            sigmoid=True,
            include_background=True,
            lambda_dice=1.0,
            lambda_focal=1.0,
            gamma=2.0,
        )

    def forward(self, logits, target):
        """logits: (B,1,H,W) raw output; target: (B,1,H,W) binary {0,1}."""
        return self.loss(logits, target)


if __name__ == "__main__":
    crit = SegLoss()
    logits = torch.randn(2, 1, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:12, 8:12] = 1.0
    print("seg loss:", crit(logits, target).item())