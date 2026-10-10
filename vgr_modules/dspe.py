# Copyright (c) Paper3 Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm_eps=1e-05, norm_momentum=0.1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, mask):
        M, T, _ = inputs.shape
        x = self.linear(inputs)
        x = self.norm(x.view(M * T, -1)).view(M, T, -1)
        x = self.relu(x)
        x = x * mask
        x_max, _ = torch.max(x, dim=1)
        return x_max


class DualStreamPillarEncoder(nn.Module):

    def __init__(
        self,
        in_channels=7,
        feat_channels=(64,),
        voxel_size=(0.16, 0.16, 5.0),
        point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 2),
        num_heatmap_channels=3,
        class_mlp_hidden=16,
        v_rc_dim_idx=5,
        rcs_dim_idx=3,
        v_rc_max=3.0,
        z_ref=3.0,
        heatmap_kernel_size=7,
        heatmap_kernel_sigma=2.0,
        min_projection_depth=0.1,
        projection_epsilon=1e-05,
        sampling_mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        norm_eps=1e-05,
        norm_momentum=0.1,
        feature_epsilon=0.001,
    ):
        super().__init__()
        self.min_projection_depth = min_projection_depth
        self.projection_epsilon = projection_epsilon
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.feature_epsilon = feature_epsilon
        self.in_channels = in_channels
        self.num_heatmap_channels = num_heatmap_channels
        self.v_rc_dim_idx = v_rc_dim_idx
        self.rcs_dim_idx = rcs_dim_idx
        self.register_buffer("v_rc_max", torch.tensor(float(v_rc_max)))
        self.register_buffer("z_ref", torch.tensor(float(z_ref)))
        aug_channels = in_channels + 6
        self.stream_a_pfn = PFNLayer(
            aug_channels, feat_channels[0], self.norm_eps, self.norm_momentum
        )
        self.stream_b_pfn = PFNLayer(
            aug_channels, feat_channels[0], self.norm_eps, self.norm_momentum
        )
        if num_heatmap_channels > 1:
            self.class_mlp = nn.Sequential(
                nn.Linear(5, class_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(class_mlp_hidden, class_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(class_mlp_hidden, num_heatmap_channels),
            )
        else:
            self.class_mlp = None
        self.register_buffer(
            "y_abs_max",
            torch.tensor(
                max(abs(point_cloud_range[1]), abs(point_cloud_range[4])),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "depth_max", torch.tensor(float(point_cloud_range[3]), dtype=torch.float32)
        )
        assert heatmap_kernel_size % 2 == 1, "kernel_size must be odd"
        self.heatmap_kernel_size = heatmap_kernel_size
        r = heatmap_kernel_size // 2
        ys, xs = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing="ij",
        )
        offsets = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        dist_sq = (xs**2 + ys**2).flatten()
        weights = torch.exp(-dist_sq / (2 * heatmap_kernel_sigma**2))
        weights = weights / weights.sum()
        self.register_buffer("hm_offsets", offsets)
        self.register_buffer("hm_weights", weights)
        self.K_hm = heatmap_kernel_size**2
        self.vx, self.vy, self.vz = voxel_size
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.pc_range = tuple(point_cloud_range)

    def get_visual_weights(self, features, num_points, coors, heatmap_prior, img_metas):
        M, T, _ = features.shape
        device = features.device
        dtype = features.dtype
        B = heatmap_prior.shape[0]
        _, _, H, W = heatmap_prior.shape
        K = self.K_hm
        weights = torch.zeros((M, T, 1), device=device, dtype=dtype)
        batch_indices = coors[:, 0].long()
        valid_points = torch.arange(T, device=device).unsqueeze(
            0
        ) < num_points.unsqueeze(1)
        valid_rcs = features[..., self.rcs_dim_idx][valid_points]
        if valid_rcs.numel() > 1:
            batch_rcs_mean = valid_rcs.mean()
            batch_rcs_std = valid_rcs.std(unbiased=False).clamp(
                min=self.feature_epsilon
            )
        else:
            batch_rcs_mean = features.new_zeros(())
            batch_rcs_std = features.new_ones(())
        for b in range(B):
            batch_mask = batch_indices == b
            if not batch_mask.any():
                continue
            feats_b = features[batch_mask]
            m = feats_b.shape[0]
            pts_b = feats_b[..., :3].reshape(-1, 3)
            v_rc_b = feats_b[..., self.v_rc_dim_idx].reshape(-1, 1)
            rcs_b = feats_b[..., self.rcs_dim_idx].reshape(-1, 1)
            if img_metas[b].get("pcd_horizontal_flip", False):
                pts_b = pts_b.clone()
                pts_b[:, 1] = -pts_b[:, 1]
            radar2img = pts_b.new_tensor(img_metas[b]["radar2img"])
            ones = torch.ones((pts_b.shape[0], 1), device=device, dtype=dtype)
            pts_h = torch.cat([pts_b, ones], dim=-1)
            projected = pts_h @ radar2img.T
            depth = projected[:, 2:3]
            depth_mask = (depth > self.min_projection_depth).float()
            u = projected[:, 0:1] / torch.clamp(depth, min=self.projection_epsilon)
            v = projected[:, 1:2] / torch.clamp(depth, min=self.projection_epsilon)
            if img_metas[b].get("flip", False):
                u = W - 1 - u
            N = u.shape[0]
            du = self.hm_offsets[:, 0].to(device)
            dv = self.hm_offsets[:, 1].to(device)
            u_patch = u + du.view(1, K)
            v_patch = v + dv.view(1, K)
            u_norm = (2.0 * u_patch + 1.0) / W - 1.0
            v_norm = (2.0 * v_patch + 1.0) / H - 1.0
            grid = torch.stack([u_norm, v_norm], dim=-1)
            grid = grid.view(1, N, K, 2)
            hm_b = heatmap_prior[b : b + 1]
            sampled = F.grid_sample(
                hm_b,
                grid,
                mode=self.sampling_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            w_kernel = self.hm_weights.to(device).view(1, 1, 1, K)
            sampled_weighted = (sampled * w_kernel).sum(dim=-1)
            sampled_per_class = sampled_weighted.squeeze(0).transpose(0, 1)
            if self.class_mlp is not None:
                v_rc_norm = torch.abs(v_rc_b) / torch.clamp(
                    self.v_rc_max, min=self.feature_epsilon
                )
                rcs_norm = (rcs_b - batch_rcs_mean) / batch_rcs_std
                z_b = feats_b[..., 2].reshape(-1, 1) / torch.clamp(
                    self.z_ref, min=self.feature_epsilon
                )
                y_abs = torch.abs(feats_b[..., 1].reshape(-1, 1)) / torch.clamp(
                    self.y_abs_max, min=self.feature_epsilon
                )
                x_b = feats_b[..., 0].reshape(-1, 1) / torch.clamp(
                    self.depth_max, min=self.feature_epsilon
                )
                mlp_in = torch.cat([v_rc_norm, rcs_norm, z_b, y_abs, x_b], dim=-1)
                alpha = torch.softmax(self.class_mlp(mlp_in), dim=-1)
                w_scalar = (sampled_per_class * alpha).sum(dim=-1, keepdim=True)
            else:
                w_scalar = sampled_per_class
            w_scalar = w_scalar * depth_mask
            weights[batch_mask] = w_scalar.view(m, T, 1)
        return weights

    def forward(self, features, num_points, coors, heatmap_prior=None, img_metas=None):
        device = features.device
        M, T, _ = features.shape
        mask = (
            (torch.arange(T, device=device).unsqueeze(0) < num_points.unsqueeze(1))
            .float()
            .unsqueeze(-1)
        )
        pts_sum = (features[:, :, :3] * mask).sum(dim=1, keepdim=True)
        pts_mean = pts_sum / torch.clamp_min(
            num_points.view(-1, 1, 1).to(features.dtype), 1.0
        )
        f_cluster = features[:, :, :3] - pts_mean
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(features.dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(features.dtype).unsqueeze(1) * self.vy + self.y_offset
        )
        f_center[:, :, 2] = features[:, :, 2] - (
            coors[:, 1].to(features.dtype).unsqueeze(1) * self.vz + self.z_offset
        )
        aug_features = torch.cat([features, f_cluster, f_center], dim=-1)
        feat_a = self.stream_a_pfn(aug_features, mask)
        if heatmap_prior is not None and img_metas is not None:
            weights = self.get_visual_weights(
                features, num_points, coors, heatmap_prior, img_metas
            )
            weights = weights * mask
            gating = 1.0 + weights
            feat_b = self.stream_b_pfn(aug_features * gating, mask)
        else:
            feat_b = self.stream_b_pfn(aug_features, mask)
        return torch.cat([feat_a, feat_b], dim=-1)
