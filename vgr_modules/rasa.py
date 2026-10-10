# Copyright (c) Paper3 Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class RASA(nn.Module):

    def __init__(
        self,
        radar_channels=384,
        img_channels=384,
        num_samples=8,
        num_scales=3,
        depth_confidence_strength=0.2,
        depth_reference=20.0,
        max_pixel_offset=20.0,
        bev_h=160,
        bev_w=160,
        voxel_size=(0.32, 0.32),
        pc_range=(0, -25.6, 51.2, 25.6),
        patch_size=3,
        patch_context_dim=128,
        min_projection_depth=0.1,
        projection_epsilon=1e-05,
        sampling_mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        norm_eps=1e-05,
        norm_momentum=0.1,
        offset_mlp_hidden=128,
        diffusion_reduction=2,
        diffusion_kernel_size=5,
        diffusion_dilation=2,
        layer_norm_epsilon=1e-05,
        scatter_min_weight=1.0,
    ):
        super().__init__()
        self.min_projection_depth = min_projection_depth
        self.projection_epsilon = projection_epsilon
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.offset_mlp_hidden = offset_mlp_hidden
        self.diffusion_reduction = diffusion_reduction
        self.diffusion_kernel_size = diffusion_kernel_size
        self.diffusion_dilation = diffusion_dilation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scatter_min_weight = scatter_min_weight
        if depth_reference <= 0:
            raise ValueError("Depth reference must be positive.")
        self.depth_confidence_strength = float(depth_confidence_strength)
        self.depth_reference = float(depth_reference)
        self.num_samples = num_samples
        self.num_scales = num_scales
        self.max_pixel_offset = max_pixel_offset
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.vx, self.vy = voxel_size
        self.x_min, self.y_min, self.x_max, self.y_max = pc_range
        self.query_propagate = nn.Sequential(
            nn.Conv2d(
                radar_channels,
                radar_channels // self.diffusion_reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                radar_channels // self.diffusion_reduction,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                radar_channels // self.diffusion_reduction,
                radar_channels // self.diffusion_reduction,
                kernel_size=self.diffusion_kernel_size,
                padding=self.diffusion_dilation * (self.diffusion_kernel_size - 1) // 2,
                dilation=self.diffusion_dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                radar_channels // self.diffusion_reduction,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                radar_channels // self.diffusion_reduction,
                radar_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                radar_channels, eps=self.norm_eps, momentum=self.norm_momentum
            ),
        )
        nn.init.zeros_(self.query_propagate[-1].weight)
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.patch_size = patch_size
        r = patch_size // 2
        ys, xs = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing="ij",
        )
        patch_offsets = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        self.register_buffer("patch_offsets", patch_offsets)
        self.K = patch_size * patch_size
        self.patch_aggregate = nn.Sequential(
            nn.Conv2d(img_channels, patch_context_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(
                patch_context_dim, eps=self.norm_eps, momentum=self.norm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                patch_context_dim,
                patch_context_dim,
                kernel_size=patch_size,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                patch_context_dim, eps=self.norm_eps, momentum=self.norm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_context_dim, patch_context_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(
                patch_context_dim, eps=self.norm_eps, momentum=self.norm_momentum
            ),
        )
        nn.init.zeros_(self.patch_aggregate[-1].weight)
        self.offset_mlp = nn.Sequential(
            nn.Linear(radar_channels + patch_context_dim, self.offset_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.offset_mlp_hidden, num_samples * 4),
        )
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)
        self.q_proj = nn.Linear(radar_channels, img_channels, bias=False)
        self.k_proj = nn.Linear(img_channels, img_channels, bias=False)
        self.res_proj = nn.Linear(radar_channels, img_channels)
        self.layer_norm = nn.LayerNorm(img_channels, eps=self.layer_norm_epsilon)
        self.out_channels = img_channels
        self.scale_attn_temperature = img_channels ** (-0.5)

    def _project_points(self, pts_xyz, radar2img, flip_img_u, img_w):
        ones = torch.ones(
            (pts_xyz.shape[0], 1), device=pts_xyz.device, dtype=pts_xyz.dtype
        )
        pts_h = torch.cat([pts_xyz, ones], dim=-1)
        projected = pts_h @ radar2img.T
        depth = projected[:, 2]
        valid_depth = depth > self.min_projection_depth
        u = projected[:, 0] / torch.clamp(depth, min=self.projection_epsilon)
        v = projected[:, 1] / torch.clamp(depth, min=self.projection_epsilon)
        if flip_img_u:
            u = img_w - 1 - u
        return (u, v, depth, valid_depth)

    def _get_radar_feat_at_points(self, radar_bev, pts_xyz):
        C, H, W = radar_bev.shape
        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        w_idx = ((x - self.x_min) / self.vx).long()
        h_idx = ((y - self.y_min) / self.vy).long()
        valid = (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
        w_idx = torch.clamp(w_idx, 0, W - 1)
        h_idx = torch.clamp(h_idx, 0, H - 1)
        feats = radar_bev[:, h_idx, w_idx].transpose(0, 1).contiguous()
        return (feats, valid)

    def _sample_patch_context(self, feat_p3, u_v, v_v, img_h, img_w):
        N_v = u_v.shape[0]
        K = self.K
        device = u_v.device
        du = self.patch_offsets[:, 0].to(device)
        dv = self.patch_offsets[:, 1].to(device)
        u_patch = u_v.unsqueeze(-1) + du.unsqueeze(0)
        v_patch = v_v.unsqueeze(-1) + dv.unsqueeze(0)
        u_norm = (2.0 * u_patch + 1.0) / img_w - 1.0
        v_norm = (2.0 * v_patch + 1.0) / img_h - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)
        grid = grid.view(1, N_v, K, 2)
        sampled = F.grid_sample(
            feat_p3,
            grid,
            mode=self.sampling_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        sampled = sampled.squeeze(0).permute(1, 0, 2).contiguous()
        ps = self.patch_size
        sampled = sampled.view(N_v, -1, ps, ps)
        ctx = self.patch_aggregate(sampled)
        ctx = ctx.view(N_v, -1)
        return ctx

    def forward(
        self, img_feats_mlvl, radar_bev_feats, radar_points_per_batch, img_metas
    ):
        B = radar_bev_feats.shape[0]
        device = radar_bev_feats.device
        dtype = radar_bev_feats.dtype
        C_img = self.out_channels
        radar_bev_for_query = radar_bev_feats + self.query_propagate(radar_bev_feats)
        out = torch.zeros(
            (B, C_img, self.bev_h, self.bev_w), device=device, dtype=dtype
        )
        count = torch.zeros((B, 1, self.bev_h, self.bev_w), device=device, dtype=dtype)
        for b in range(B):
            pts = radar_points_per_batch[b]
            if pts is None or pts.shape[0] == 0:
                continue
            pts_xyz = pts[:, :3].contiguous()
            radar_feat_at_pts, valid_bev = self._get_radar_feat_at_points(
                radar_bev_for_query[b], pts_xyz
            )
            img_meta = img_metas[b]
            radar2img = pts_xyz.new_tensor(img_meta["radar2img"])
            input_shape = img_meta["input_shape"]
            img_h, img_w = (input_shape[0], input_shape[1])
            flip_img_u = False
            u, v, depth, valid_depth = self._project_points(
                pts_xyz, radar2img, flip_img_u, img_w
            )
            valid = valid_bev & valid_depth
            if not valid.any():
                continue
            pts_xyz_v = pts_xyz[valid]
            radar_feat_v = radar_feat_at_pts[valid]
            u_v = u[valid]
            v_v = v[valid]
            depth_v = depth[valid]
            N_v = pts_xyz_v.shape[0]
            feat_p3 = img_feats_mlvl[0][b : b + 1]
            patch_ctx = self._sample_patch_context(feat_p3, u_v, v_v, img_h, img_w)
            offset_input = torch.cat([radar_feat_v, patch_ctx], dim=-1)
            raw = self.offset_mlp(offset_input)
            raw = raw.view(N_v, self.num_samples, 4)
            du_raw, dv_raw = (raw[..., 0], raw[..., 1])
            attn_logit = raw[..., 2]
            conf_logit = raw[..., 3]
            du = self.max_pixel_offset * torch.tanh(du_raw)
            dv = self.max_pixel_offset * torch.tanh(dv_raw)
            u_samp = u_v.unsqueeze(-1) + du
            v_samp = v_v.unsqueeze(-1) + dv
            per_scale_feats = []
            for l in range(min(self.num_scales, len(img_feats_mlvl))):
                feat_l = img_feats_mlvl[l][b : b + 1]
                _, C_img_l, H_l, W_l = feat_l.shape
                u_norm = (2.0 * u_samp + 1.0) / img_w - 1.0
                v_norm = (2.0 * v_samp + 1.0) / img_h - 1.0
                grid = torch.stack([u_norm, v_norm], dim=-1)
                grid = grid.view(1, N_v, self.num_samples, 2)
                sampled = F.grid_sample(
                    feat_l,
                    grid,
                    mode=self.sampling_mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )
                sampled = sampled.squeeze(0).permute(1, 2, 0)
                attn = torch.softmax(attn_logit, dim=-1)
                pooled = (sampled * attn.unsqueeze(-1)).sum(dim=1)
                per_scale_feats.append(pooled)
            while len(per_scale_feats) < self.num_scales:
                per_scale_feats.append(torch.zeros_like(per_scale_feats[-1]))
            stacked = torch.stack(per_scale_feats, dim=1)
            q = self.q_proj(radar_feat_v)
            k = self.k_proj(stacked)
            scores = (q.unsqueeze(1) * k).sum(dim=-1) * self.scale_attn_temperature
            scale_weight = torch.softmax(scores, dim=-1)
            scale_pooled = (stacked * scale_weight.unsqueeze(-1)).sum(dim=1)
            fused = scale_pooled + self.res_proj(radar_feat_v)
            fused = self.layer_norm(fused)
            conf = torch.sigmoid(conf_logit.mean(dim=-1))
            depth_boost = 1.0 + self.depth_confidence_strength * torch.tanh(
                (depth_v - self.depth_reference) / self.depth_reference
            )
            conf = conf * depth_boost
            x_v = pts_xyz_v[:, 0]
            y_v = pts_xyz_v[:, 1]
            w_idx = ((x_v - self.x_min) / self.vx).long().clamp(0, self.bev_w - 1)
            h_idx = ((y_v - self.y_min) / self.vy).long().clamp(0, self.bev_h - 1)
            flat_idx = h_idx * self.bev_w + w_idx
            weighted = fused * conf.unsqueeze(-1)
            out_flat = out[b].view(C_img, -1)
            count_flat = count[b].view(1, -1)
            out_flat_T = out_flat.transpose(0, 1)
            out_flat_T.index_add_(0, flat_idx, weighted)
            count_flat[0].index_add_(0, flat_idx, conf)
            out[b] = out_flat_T.transpose(0, 1).view(C_img, self.bev_h, self.bev_w)
            count[b] = count_flat.view(1, self.bev_h, self.bev_w)
        out = out / torch.clamp(count, min=self.scatter_min_weight)
        return out
