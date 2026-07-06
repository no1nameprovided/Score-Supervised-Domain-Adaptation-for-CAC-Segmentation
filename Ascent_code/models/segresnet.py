"""
models/segresnet.py
-------------------
The student network: a 2D SegResNet that consumes 2.5D input (neighbouring
slices stacked into channels). dropout_prob is kept > 0 so that later stages
can use MC-dropout for voxel uncertainty in the teacher-student refinement.

The same architecture is used for the EMA teacher (Stage 2), which is required
because Mean-Teacher averages weights -> teacher and student must be identical
in structure.
"""

from monai.networks.nets import SegResNet


def build_student(in_channels=3, out_channels=1, init_filters=16,
                  dropout_prob=0.2):
    """returns a SegResNet producing single-channel logits (apply sigmoid)."""
    model = SegResNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=dropout_prob,
    )
    return model


if __name__ == "__main__":
    import torch
    net = build_student()
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    print("output shape:", y.shape)   # expect (2,1,128,128)
    n_params = sum(p.numel() for p in net.parameters())
    print("params:", f"{n_params/1e6:.2f}M")