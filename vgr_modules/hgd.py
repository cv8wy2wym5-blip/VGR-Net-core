# Copyright (c) Paper3 Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import build_bev_reference_grid


class HGD(nn.Module):
    CHANNEL_ORDER = ("car", "pedestrian", "cyclist")

    def __init__(
        self,
        img_channels=384,
        num_heatmap_channels=3,
        bev_h=160,
        bev_w=160,
        voxel_size=(0.32, 0.32),
        pc_range=(0, -25.6, 51.2, 25.6),
        class_heights=(1.0, 0.85, 0.90),
        class_sigmas=(1.0, 0.5, 0.6),
        num_z_levels=8,
        z_range=(-3.0, 2.0),
        min_projection_depth=0.1,
        projection_epsilon=1e-05,
        sampling_mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        class_response_epsilon=1e-06,
        prior_log_epsilon=1e-08,
        height_bias_initial_value=0.0,
    ):
        super().__init__()
        self.min_projection_depth = min_projection_depth
        self.projection_epsilon = projection_epsilon
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.class_response_epsilon = class_response_epsilon
        self.prior_log_epsilon = prior_log_epsilon
        self.height_bias_initial_value = height_bias_initial_value
        self.num_heatmap_channels = num_heatmap_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.vx, self.vy = voxel_size
        self.x_min, self.y_min, self.x_max, self.y_max = pc_range
        self.class_heights = class_heights
        self.class_sigmas = class_sigmas
        self.num_z_levels = num_z_levels
        self.z_min, self.z_max = z_range
        self.out_channels = img_channels
        z_anchors = torch.linspace(self.z_min, self.z_max, num_z_levels)
        self.register_buffer("z_anchors", z_anchors)
        class_z_prior = torch.zeros(num_heatmap_channels, num_z_levels)
        for c in range(num_heatmap_channels):
            h_c = class_heights[c] if c < len(class_heights) else 1.0
            sigma = class_sigmas[c] if c < len(class_sigmas) else 0.8
            w = torch.exp(-((z_anchors - h_c) ** 2) / (2 * sigma**2))
            class_z_prior[c] = w / w.sum()
        self.register_buffer("class_z_prior", class_z_prior)
        self.z_logits_offset = nn.Parameter(
            torch.full(
                (num_heatmap_channels, num_z_levels), self.height_bias_initial_value
            )
        )

    def _heatmap_to_bev(
        self, heatmap_full, radar2img, img_h, img_w, flip_img_u, device, dtype
    ):
        C_h = heatmap_full.shape[0]
        per_class_bev = []
        for c in range(C_h):
            z_val = self.class_heights[c] if c < len(self.class_heights) else 1.0
            ref = build_bev_reference_grid(
                self.bev_h,
                self.bev_w,
                voxel_size=(self.vx, self.vy),
                pc_range=(self.x_min, self.y_min, self.x_max, self.y_max),
                z_value=z_val,
                device=device,
                dtype=dtype,
            )
            pts = ref.view(-1, 3)
            ones = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
            pts_h = torch.cat([pts, ones], dim=-1)
            projected = pts_h @ radar2img.T
            depth = projected[:, 2:3]
            valid = (depth > self.min_projection_depth).float()
            u = projected[:, 0:1] / torch.clamp(depth, min=self.projection_epsilon)
            v = projected[:, 1:2] / torch.clamp(depth, min=self.projection_epsilon)
            if flip_img_u:
                u = img_w - 1 - u
            u_norm = (2.0 * u + 1.0) / img_w - 1.0
            v_norm = (2.0 * v + 1.0) / img_h - 1.0
            grid = torch.cat([u_norm, v_norm], dim=-1).view(
                1, self.bev_h, self.bev_w, 2
            )
            hm_c = heatmap_full[c : c + 1].unsqueeze(0)
            sampled = F.grid_sample(
                hm_c,
                grid,
                mode=self.sampling_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            sampled = sampled.squeeze(0).squeeze(0)
            valid_bev = valid.view(self.bev_h, self.bev_w)
            sampled = sampled * valid_bev
            per_class_bev.append(sampled)
        return torch.stack(per_class_bev, dim=0)

    def _sample_image_to_bev(
        self,
        img_feats_mlvl,
        heatmap_bev,
        radar2img,
        img_h,
        img_w,
        flip_img_u,
        device,
        dtype,
    ):
        eff_class_prior = torch.softmax(
            torch.log(self.class_z_prior + self.prior_log_epsilon)
            + self.z_logits_offset,
            dim=-1,
        )
        hm_sum = heatmap_bev.sum(dim=0, keepdim=True)
        eps = self.class_response_epsilon
        hm_normed = (heatmap_bev + eps / heatmap_bev.shape[0]) / (hm_sum + eps)
        z_weights_per_cell = torch.einsum("chw,cz->zhw", hm_normed, eff_class_prior)
        accum = None
        for z_idx in range(self.num_z_levels):
            z_val = self.z_anchors[z_idx].item()
            ref = build_bev_reference_grid(
                self.bev_h,
                self.bev_w,
                voxel_size=(self.vx, self.vy),
                pc_range=(self.x_min, self.y_min, self.x_max, self.y_max),
                z_value=z_val,
                device=device,
                dtype=dtype,
            )
            pts = ref.view(-1, 3)
            ones = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
            pts_h = torch.cat([pts, ones], dim=-1)
            projected = pts_h @ radar2img.T
            depth = projected[:, 2:3]
            valid = (depth > self.min_projection_depth).float()
            u = projected[:, 0:1] / torch.clamp(depth, min=self.projection_epsilon)
            v = projected[:, 1:2] / torch.clamp(depth, min=self.projection_epsilon)
            if flip_img_u:
                u = img_w - 1 - u
            u_norm = (2.0 * u + 1.0) / img_w - 1.0
            v_norm = (2.0 * v + 1.0) / img_h - 1.0
            grid = torch.cat([u_norm, v_norm], dim=-1).view(
                1, self.bev_h, self.bev_w, 2
            )
            total = 0.0
            count_l = 0
            for feat_l in img_feats_mlvl:
                sampled = F.grid_sample(
                    feat_l,
                    grid,
                    mode=self.sampling_mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )
                total = total + sampled
                count_l += 1
            avg = total / max(count_l, 1)
            avg = avg.squeeze(0)
            valid_bev = valid.view(1, self.bev_h, self.bev_w)
            z_w = z_weights_per_cell[z_idx : z_idx + 1]
            contrib = avg * valid_bev * z_w
            accum = contrib if accum is None else accum + contrib
        return accum

    def forward(self, img_feats_mlvl, radar_bev_feats, heatmap_full_batch, img_metas):
        B = radar_bev_feats.shape[0]
        device = radar_bev_feats.device
        dtype = radar_bev_feats.dtype
        feat_list = []
        for b in range(B):
            img_meta = img_metas[b]
            radar2img = radar_bev_feats.new_tensor(img_meta["radar2img"])
            input_shape = img_meta["input_shape"]
            img_h, img_w = (input_shape[0], input_shape[1])
            flip_img_u = False
            hm_bev = self._heatmap_to_bev(
                heatmap_full_batch[b],
                radar2img,
                img_h,
                img_w,
                flip_img_u,
                device,
                dtype,
            )
            feats_mlvl_b = [f[b : b + 1] for f in img_feats_mlvl]
            img_bev_dense = self._sample_image_to_bev(
                feats_mlvl_b, hm_bev, radar2img, img_h, img_w, flip_img_u, device, dtype
            )
            feat_list.append(img_bev_dense)
        feats = torch.stack(feat_list, dim=0)
        return feats
