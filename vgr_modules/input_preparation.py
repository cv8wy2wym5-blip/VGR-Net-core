import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .configuration import load_parameters


def image_shapes(original_hw, scale=0.48, pad_divisor=32):
    height, width = (int(value) for value in original_hw)
    if min(height, width, scale, pad_divisor) <= 0:
        raise ValueError("Image dimensions, scale, and padding divisor must be positive.")
    resized = (max(1, int(height * scale + 0.5)), max(1, int(width * scale + 0.5)))
    padded = tuple(int(math.ceil(value / pad_divisor) * pad_divisor) for value in resized)
    return resized, padded


def resize_projection(radar2img, original_hw, resized_hw):
    matrix = torch.as_tensor(radar2img).clone()
    if not matrix.is_floating_point():
        matrix = matrix.float()
    if tuple(matrix.shape) not in ((3, 4), (4, 4)):
        raise ValueError("radar2img must be a calibrated 3x4 or 4x4 matrix.")
    if not torch.isfinite(matrix).all() or min(*original_hw, *resized_hw) <= 0:
        raise ValueError("Invalid projection or image dimensions.")
    matrix[0] *= resized_hw[1] / original_hw[1]
    matrix[1] *= resized_hw[0] / original_hw[0]
    return matrix


def prepare_heatmap(heatmap, resized_hw, padded_hw):
    value = torch.as_tensor(heatmap)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("Heatmap channels must be Car, Pedestrian, Cyclist.")
    value = value.float() / 255.0 if value.dtype == torch.uint8 else value.float()
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Heatmap values must be finite and within [0,1].")
    if min(*resized_hw, *padded_hw) <= 0 or any(a > b for a, b in zip(resized_hw, padded_hw)):
        raise ValueError("Padded dimensions must contain the valid resized image.")
    resized = F.interpolate(value, size=tuple(resized_hw), mode="bilinear", align_corners=False)
    return F.pad(resized, (0, padded_hw[1] - resized_hw[1], 0, padded_hw[0] - resized_hw[0]))


def load_heatmap(path, resized_hw, padded_hw, config_path=None):
    path = Path(path)
    marker = path.parent / "provenance.json"
    if not marker.is_file():
        raise ValueError("Semantic cache provenance is missing.")
    metadata = json.loads(marker.read_text())
    settings = load_parameters(config_path)["semantic_prior"]
    expected = dict(taxonomy="traffic6", output_channels=7, class_mapping=settings["class_mapping"], checkpoint_sha256=settings["checkpoint_sha256"], input_scale=settings["cache_input_scale"], stride=settings["cache_stride"])
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Semantic cache provenance differs from the configured checkpoint.")
    with np.load(path, allow_pickle=False) as archive:
        heatmap = archive["hm"].copy()
    return prepare_heatmap(heatmap, resized_hw, padded_hw)
