# Copyright (c) Paper3 Authors. All rights reserved.
import torch


def build_bev_reference_grid(
    bev_h, bev_w, voxel_size, pc_range, z_value=1.0, device="cuda", dtype=torch.float32
):
    ys = torch.arange(bev_h, device=device, dtype=dtype)
    xs = torch.arange(bev_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x_coord = (grid_x + 0.5) * voxel_size[0] + pc_range[0]
    y_coord = (grid_y + 0.5) * voxel_size[1] + pc_range[1]
    if isinstance(z_value, torch.Tensor):
        z_coord = z_value
    else:
        z_coord = torch.full_like(x_coord, fill_value=float(z_value))
    ref = torch.stack([x_coord, y_coord, z_coord], dim=-1)
    return ref
