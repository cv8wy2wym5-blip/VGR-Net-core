from pathlib import Path
import copy
import inspect
import math

import torch.nn as nn
import yaml

from .dspe import DualStreamPillarEncoder
from .rasa import RASA
from .hgd import HGD


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "parameters.yaml"


def load_parameters(path=None):
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream)
    if not isinstance(parameters, dict) or parameters.get("format_version") != 2:
        raise ValueError("Unsupported parameter format.")
    expected = {
        "format_version",
        "classes",
        "sampling",
        "normalization",
        "modules",
        "datasets",
        "training_reference",
        "semantic_prior",
        "radar_input",
    }
    if set(parameters) != expected:
        raise ValueError("Unexpected or missing top-level parameter sections.")
    if parameters["classes"] != ["Car", "Pedestrian", "Cyclist"]:
        raise ValueError("Semantic channel order must be Car, Pedestrian, Cyclist.")
    sampling = parameters["sampling"]
    if (
        sampling["sampling_mode"] != "bilinear"
        or sampling["padding_mode"] != "zeros"
        or sampling["align_corners"] is not False
    ):
        raise ValueError(
            "The released sampling convention is bilinear, zero padding, align_corners=False."
        )
    if sampling["min_projection_depth"] <= 0 or sampling["projection_epsilon"] <= 0:
        raise ValueError("Projection guards must be positive.")
    semantic = parameters["semantic_prior"]
    if semantic["architecture"] != "deeplabv3_mobilenet_v3_large" or semantic["taxonomy"] != "traffic6":
        raise ValueError("The released semantic architecture and taxonomy must match traffic6.")
    expected_names = ["background", "car", "bus", "truck", "person", "bicycle", "motorcycle"]
    expected_mapping = {"car": [1, 2, 3], "pedestrian": [4], "cyclist": [5, 6]}
    if semantic["class_names"] != expected_names or semantic["class_mapping"] != expected_mapping:
        raise ValueError("Semantic categories or the six-to-three mapping are incompatible.")
    if semantic["frozen"] is not True or semantic["renormalize_foreground"] is not False:
        raise ValueError("The prior must remain frozen without foreground renormalization.")
    if semantic["cache_stride"] < 1 or semantic["cache_input_scale"] != 1.0 or semantic["cache_dtype"] != "uint8":
        raise ValueError("Unsupported semantic cache convention.")
    if semantic["rgb_mean"] != [0.485, 0.456, 0.406] or semantic["rgb_std"] != [0.229, 0.224, 0.225]:
        raise ValueError("RGB normalization must match the trained semantic model.")
    if len(semantic["checkpoint_sha256"]) != 64:
        raise ValueError("A SHA256 identity is required for the supplied checkpoint.")
    if parameters["radar_input"]["dspe_affinity_velocity_index"] != parameters["modules"]["dspe"]["v_rc_dim_idx"]:
        raise ValueError("The DSPE affinity velocity index does not match the input contract.")
    return parameters


def _grid_shape(point_range, resolution):
    dimensions = [
        (point_range[4] - point_range[1]) / resolution[1],
        (point_range[3] - point_range[0]) / resolution[0],
    ]
    if any(
        not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-6)
        for value in dimensions
    ):
        raise ValueError("The point range must be divisible by the BEV resolution.")
    if any(value < 1 for value in dimensions):
        raise ValueError("BEV dimensions must be positive.")
    return [int(round(value)) for value in dimensions]


def resolve_module_parameters(parameters, dataset="vod"):
    parameters = copy.deepcopy(parameters)
    if dataset not in parameters["datasets"]:
        raise ValueError("Unknown dataset profile: " + str(dataset))
    profile = parameters["datasets"][dataset]
    point_range = profile["point_cloud_range"]
    pillar_size = profile["pillar_voxel_size"]
    fusion_size = profile["fusion_voxel_size"]
    if len(point_range) != 6 or len(pillar_size) != 3 or len(fusion_size) != 2:
        raise ValueError("Invalid geometry dimensions.")
    if any(value <= 0 for value in pillar_size + fusion_size):
        raise ValueError("Grid resolutions must be positive.")
    if any(point_range[index + 3] <= point_range[index] for index in range(3)):
        raise ValueError("Point range maxima must exceed minima.")
    bev_h, bev_w = _grid_shape(point_range, fusion_size)
    classes = len(parameters["classes"])
    dspe = dict(
        parameters["modules"]["dspe"],
        **parameters["sampling"],
        **parameters["normalization"],
    )
    dspe.update(
        num_heatmap_channels=classes,
        voxel_size=pillar_size,
        point_cloud_range=point_range,
    )
    rasa = dict(
        parameters["modules"]["rasa"],
        **parameters["sampling"],
        **parameters["normalization"],
    )
    hgd = dict(parameters["modules"]["hgd"], **parameters["sampling"])
    for configuration in (rasa, hgd):
        configuration.update(
            bev_h=bev_h,
            bev_w=bev_w,
            voxel_size=fusion_size,
            pc_range=[point_range[0], point_range[1], point_range[3], point_range[4]],
        )
    heights = profile["hgd_class_heights"]
    sigmas = profile["hgd_class_sigmas"]
    if (
        len(heights) != classes
        or len(sigmas) != classes
        or any(sigma <= 0 for sigma in sigmas)
    ):
        raise ValueError(
            "HGD needs one reference height and positive bandwidth per class."
        )
    hgd.update(
        num_heatmap_channels=classes,
        class_heights=heights,
        class_sigmas=sigmas,
        z_range=[point_range[2], point_range[5]],
    )
    if (
        dspe["heatmap_kernel_size"] < 1
        or dspe["heatmap_kernel_size"] % 2 != 1
        or dspe["heatmap_kernel_sigma"] <= 0
    ):
        raise ValueError(
            "DSPE requires an odd positive Gaussian kernel and positive sigma."
        )
    if len(dspe["feat_channels"]) != 1 or dspe["in_channels"] < 3:
        raise ValueError("This DSPE implementation uses one PFN layer per stream.")
    if any(
        not 0 <= dspe[key] < dspe["in_channels"]
        for key in ("rcs_dim_idx", "v_rc_dim_idx")
    ):
        raise ValueError("Radar attribute indices must fall inside the feature layout.")
    if rasa["patch_size"] < 1 or rasa["patch_size"] % 2 != 1:
        raise ValueError("RASA requires an odd positive patch size.")
    if (
        rasa["diffusion_kernel_size"] < 1
        or rasa["diffusion_kernel_size"] % 2 != 1
        or rasa["diffusion_dilation"] < 1
    ):
        raise ValueError("RASA diffusion requires an odd kernel and positive dilation.")
    if (
        rasa["diffusion_reduction"] < 1
        or rasa["radar_channels"] % rasa["diffusion_reduction"]
    ):
        raise ValueError("Radar channels must be divisible by the diffusion reduction.")
    if any(
        rasa[key] <= 0
        for key in (
            "num_samples",
            "num_scales",
            "patch_context_dim",
            "offset_mlp_hidden",
            "scatter_min_weight",
            "depth_reference",
        )
    ):
        raise ValueError(
            "RASA sizes, reference depth and scatter lower bound must be positive."
        )
    if rasa["max_pixel_offset"] < 0 or not 0 <= rasa["depth_confidence_strength"] < 1:
        raise ValueError("Invalid RASA offset bound or confidence strength.")
    if hgd["img_channels"] != rasa["img_channels"] or hgd["num_z_levels"] < 1:
        raise ValueError(
            "Image channels must agree and HGD needs at least one height anchor."
        )
    if any(dspe[key] <= 0 for key in ("feature_epsilon", "v_rc_max", "z_ref")):
        raise ValueError("DSPE normalization guards must be positive.")
    if any(hgd[key] <= 0 for key in ("class_response_epsilon", "prior_log_epsilon")):
        raise ValueError("HGD numerical guards must be positive.")
    return {"dspe": dspe, "rasa": rasa, "hgd": hgd}


def build_modules(dataset="vod", config_path=None, parameters=None):
    if config_path is not None and parameters is not None:
        raise ValueError("Provide config_path or parameters, not both.")
    if parameters is None:
        parameters = load_parameters(config_path)
    resolved = resolve_module_parameters(parameters, dataset)
    classes = {"dspe": DualStreamPillarEncoder, "rasa": RASA, "hgd": HGD}
    modules = {}
    for name, constructor in classes.items():
        accepted = set(inspect.signature(constructor).parameters)
        unknown = set(resolved[name]) - accepted
        if unknown:
            raise ValueError(
                "Unknown " + name + " parameters: " + ", ".join(sorted(unknown))
            )
        modules[name] = constructor(**resolved[name])
    return nn.ModuleDict(modules)
