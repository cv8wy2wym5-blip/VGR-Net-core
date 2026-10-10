import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from .configuration import DEFAULT_CONFIG, load_parameters


class SemanticPriorGenerator(nn.Module):
    def __init__(self, checkpoint_path, settings, verify_hash=True):
        super().__init__()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if verify_hash and self.checkpoint_sha256 != settings["checkpoint_sha256"]:
            raise ValueError("Semantic checkpoint SHA256 differs from the configuration.")
        checkpoint = torch.load(str(path), map_location="cpu")
        names = settings["class_names"]
        expected = dict(enumerate(names[1:], 1))
        if checkpoint.get("taxonomy") != "traffic6" or checkpoint.get("categories") != expected:
            raise ValueError("Checkpoint taxonomy or foreground order is incompatible.")
        state = checkpoint["state_dict"]
        if state["classifier.4.weight"].shape[0] != len(names):
            raise ValueError("The semantic model must have seven output channels.")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Semantic checkpoint contains non-finite parameters.")
        self.model = deeplabv3_mobilenet_v3_large(
            pretrained=False,
            pretrained_backbone=False,
            num_classes=len(names),
            aux_loss=any(key.startswith("aux_classifier.") for key in state),
        )
        self.model.load_state_dict(state, strict=True)
        self.class_mapping = settings["class_mapping"]
        self.channel_order = ("car", "pedestrian", "cyclist")
        self.register_buffer("mean", torch.tensor(settings["rgb_mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(settings["rgb_std"]).view(1, 3, 1, 1))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def probabilities(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected floating RGB images with shape [B,3,H,W].")
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be finite and within [0,1].")
        return self.model((rgb - self.mean) / self.std)["out"].softmax(dim=1)

    def aggregate_probs(self, probabilities):
        if probabilities.ndim != 4 or probabilities.shape[1] != 7:
            raise ValueError("Expected seven semantic probability channels.")
        return torch.cat(
            [probabilities[:, self.class_mapping[name]].sum(dim=1, keepdim=True)
             for name in self.channel_order], dim=1
        ).clamp(0, 1)

    def forward(self, rgb):
        return self.aggregate_probs(self.probabilities(rgb))


def build_prior_generator(config_path=None, checkpoint_path=None, device="cpu"):
    path = Path(config_path or DEFAULT_CONFIG).resolve()
    settings = load_parameters(path)["semantic_prior"]
    checkpoint = Path(checkpoint_path or settings["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    return SemanticPriorGenerator(checkpoint, settings).to(device)
