import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vgr_modules import build_modules, load_parameters, resolve_module_parameters
from vgr_modules.input_preparation import image_shapes, prepare_heatmap, resize_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    device = torch.device(args.device)
    parameters = load_parameters(args.config)
    report = dict(status="passed", input_type="synthetic tensors for module execution checks", trained_detector=False, profiles={})
    for dataset in parameters["datasets"]:
        settings = resolve_module_parameters(parameters, dataset)
        modules = build_modules(dataset, config_path=args.config).to(device).eval()
        height, width = modules["rasa"].bev_h, modules["rasa"].bev_w
        channels = settings["rasa"]["img_channels"]
        image = [torch.randn(1, channels, h, w, device=device, requires_grad=True) for h, w in ((8, 12), (4, 6), (2, 3))]
        bev = torch.randn(1, channels, height, width, device=device, requires_grad=True)
        prior = torch.rand(1, 3, 64, 96, device=device)
        projection = [[48, 40, 0, 0], [32, 0, -30, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        metadata = [dict(radar2img=projection, input_shape=(64, 96), flip=False, pcd_horizontal_flip=False)]
        features = torch.randn(4, 4, 7, device=device)
        features[:, :, 0] = torch.tensor([2., 6., 12., 20.], device=device)[:, None]
        features[:, :, 1:3] *= 0.1
        counts = torch.tensor([4, 2, 3, 1], device=device)
        profile = parameters["datasets"][dataset]
        voxel = profile["pillar_voxel_size"]
        extent = profile["point_cloud_range"]
        coors = torch.zeros(4, 4, dtype=torch.long, device=device)
        coors[:, 1] = int((0 - extent[2]) / voxel[2])
        coors[:, 2] = int((0 - extent[1]) / voxel[1])
        coors[:, 3] = ((features[:, 0, 0] - extent[0]) / voxel[0]).long()
        outputs = {
            "dspe": modules["dspe"](features, counts, coors, prior, metadata),
            "rasa": modules["rasa"](image, bev, [features[:, 0]], metadata),
            "hgd": modules["hgd"](image, bev, prior, metadata),
        }
        assert outputs["dspe"].shape == (4, 128)
        assert outputs["rasa"].shape == outputs["hgd"].shape == (1, 384, height, width)
        assert all(torch.isfinite(value).all() for value in outputs.values())
        sum(value.square().mean() for value in outputs.values()).backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in modules.parameters())
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in image)
        with torch.no_grad():
            assert torch.count_nonzero(modules["rasa"]([v.detach() for v in image], bev.detach(), [torch.empty(0, 7, device=device)], metadata)) == 0
            assert torch.isfinite(modules["hgd"]([v.detach() for v in image], bev.detach(), torch.zeros_like(prior), metadata)).all()
            invalid = [dict(radar2img=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], input_shape=(64, 96))]
            assert torch.count_nonzero(modules["hgd"]([v.detach() for v in image], bev.detach(), prior, invalid)) == 0
        original_hw = (1216, 1936) if dataset == "vod" else (1200, 1920)
        resized, padded = image_shapes(original_hw, profile["caller_preprocessing_reference"]["image_resize_scale"], profile["caller_preprocessing_reference"]["image_pad_divisor"])
        transformed = resize_projection(torch.tensor(projection), original_hw, resized)
        expected = torch.tensor(projection, dtype=torch.float32)
        expected[0] *= resized[1] / original_hw[1]
        expected[1] *= resized[0] / original_hw[0]
        torch.testing.assert_close(transformed, expected)
        prepared = prepare_heatmap(torch.full((3, 8, 12), 255, dtype=torch.uint8), resized, padded)
        assert prepared.shape == (1, 3, *padded)
        torch.testing.assert_close(prepared[:, :, :resized[0], :resized[1]], torch.ones(1, 3, *resized))
        assert torch.count_nonzero(prepared[:, :, resized[0]:]) == 0
        assert torch.count_nonzero(prepared[:, :, :, resized[1]:]) == 0
        report["profiles"][dataset] = dict(output_shapes={name:list(value.shape) for name,value in outputs.items()}, forward_backward="passed", empty_radar="passed", zero_semantics="passed", invalid_projection="passed", resize_projection_and_padding="passed", resized_hw=resized, padded_hw=padded)
        del outputs, modules, image, bev
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
