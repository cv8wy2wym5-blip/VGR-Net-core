import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vgr_modules import build_prior_generator, load_parameters


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--image-root")
    parser.add_argument("--image-list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    parameters = load_parameters(args.config)
    settings = parameters["semantic_prior"]
    model = build_prior_generator(args.config, device=args.device)
    if args.image:
        images = [Path(args.image)]
    elif args.image_list:
        images = [Path(args.image_root) / name.strip() for name in Path(args.image_list).read_text().splitlines() if name.strip()]
    else:
        images = sorted(path for path in Path(args.image_root).iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images or any(not path.is_file() for path in images):
        raise FileNotFoundError("No usable input images.")
    if len({path.stem for path in images}) != len(images):
        raise ValueError("Input image stems must be unique within a cache directory.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(taxonomy="traffic6", output_channels=7, class_mapping=settings["class_mapping"], checkpoint_sha256=model.checkpoint_sha256, input_scale=1.0, stride=settings["cache_stride"])
    marker = output / "provenance.json"
    if list(output.glob("*.npz")) and (not marker.is_file() or json.loads(marker.read_text()) != provenance):
        raise ValueError("Existing caches use different or missing provenance.")
    marker.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for path in images:
        image = np.array(Image.open(path).convert("RGB"), copy=True)
        height, width = image.shape[:2]
        rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.0
        with torch.no_grad():
            prior = model(rgb)
            shape = (max(1, height // settings["cache_stride"]), max(1, width // settings["cache_stride"]))
            small = F.interpolate(prior, size=shape, mode="bilinear", align_corners=False)[0]
        quantized = (small.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        np.savez_compressed(output / (path.stem + ".npz"), hm=quantized, orig_h=np.int32(height), orig_w=np.int32(width), stride=np.int32(settings["cache_stride"]))
        print(path.name, list(quantized.shape), flush=True)


if __name__ == "__main__":
    main()
