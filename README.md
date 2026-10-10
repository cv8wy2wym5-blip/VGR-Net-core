# VGR-Net Core

Core implementations for **VGR-Net: Vision-Guided 4D Radar Encoding and Sparse-Dense BEV Fusion for 3D Object Detection**.

## Contents

| Component | File |
|---|---|
| DSPE | `vgr_modules/dspe.py` |
| RASA | `vgr_modules/rasa.py` |
| HGD | `vgr_modules/hgd.py` |
| BEV geometry and image/projection preparation | `vgr_modules/geometry.py`, `vgr_modules/input_preparation.py` |
| Semantic-prior loading and inference | `vgr_modules/semantic_prior.py`, `tools/infer_prior.py` |
| Module check | `tools/verify_core.py` |
| Module parameters, dataset profiles, and reference training settings | `parameters.yaml` |

The full research codebase remains in use for ongoing, unpublished work and is not released in its entirety at this stage. This supporting release provides modules for integration into an independently implemented detector. It does not contain the complete detector architecture and training framework, integrated sparse-dense BEV fusion, the detection head, final detector checkpoints, or detection evaluation scripts. Datasets and semantic caches are stored externally.

## Download

The [v1.0.0 release](https://github.com/cv8wy2wym5-blip/VGR-Net-core/releases/tag/v1.0.0) provides `VGR-Net_core_code.zip`, which contains the core source files, documentation and semantic-prior checkpoint. Extract the package and run the commands below from its root.

## Environment and quick check

Environment: Python 3.8.20, PyTorch 1.12.0+cu113, Torchvision 0.13.0+cu113, and CUDA 11.3.

From the package root:

```bash
python tools/verify_core.py --device cpu --output /path/to/output/core_validation.json
```

The check executes both dataset profiles at the configured 384-channel BEV dimensions, performs backward propagation, and checks empty radar anchors, zero semantic responses, invalid depth, projection resizing, and heatmap padding. Its inputs are synthetic module-test tensors and are not detection experiments.

## Using the modules

```python
from vgr_modules import build_modules, build_prior_generator, image_shapes, prepare_heatmap, resize_projection

modules = build_modules(dataset="vod")
prior_generator = build_prior_generator(device="cuda:0")
```

Use `dataset="dual_radar"` for the second profile. Configured BEV outputs are 160x160 for VoD and 256x256 for Dual-Radar, at 0.32 m per cell. Both profiles use 384 image/radar feature channels, three image feature levels, eight RASA samples, and eight HGD height anchors.

| Module | Required inputs | Output |
|---|---|---|
| DSPE | Pillar tensor `[M,T,7]`, valid counts `[M]`, coordinates `[M,4]` in batch/z/y/x order, heatmap `[B,3,Hpad,Wpad]`, image metadata | Pillar features `[M,128]` |
| RASA | Three image feature maps `[B,384,Hl,Wl]`, radar BEV `[B,384,Hbev,Wbev]`, per-frame radar points, image metadata | Sparse image BEV `[B,384,Hbev,Wbev]` |
| HGD | Three image feature maps, radar BEV shape/device reference, padded semantic heatmap, image metadata | Dense image BEV `[B,384,Hbev,Wbev]` |

Metadata contains the calibrated `radar2img` matrix after image resizing and `input_shape=(Hpad,Wpad)`. Radar coordinates use x forward, y left, and z up, in the same frame as calibration. Filter points to the configured half-open point-cloud range before pillarization and RASA. Caller-side radar and detector-image normalization settings are recorded under each dataset profile.

The seven radar features follow `radar_input.feature_order`. DSPE uses the compensated-velocity slot at index 5 for class affinity. Dataset-specific loading and motion compensation are caller responsibilities. A raw five-field radar record with one Doppler value does not establish a separate compensated-velocity measurement.

Image features are expected from the frozen YOLOX-S image backbone/neck with trainable channel projections described in the manuscript. They are distinct from the semantic model's input. The semantic model accepts unnormalized RGB tensors in `[0,1]` and internally applies ImageNet normalization.

## Semantic priors

The pretrained semantic-prior checkpoint is `weights/deeplabv3_mobilenet_v3_large_traffic6.pth`. The model remains frozen during inference.

Output channels are background, car, bus, truck, person, bicycle, and motorcycle. Softmax is taken over all seven. Car sums car/bus/truck, Pedestrian uses person, and Cyclist sums bicycle/motorcycle. The background is omitted, and the three foreground channels are not renormalized. Their order is Car, Pedestrian, Cyclist.

Generate real-image caches externally:

```bash
python tools/infer_prior.py --image /path/to/frame.png --output-dir /path/to/priors --device cuda:0
python tools/infer_prior.py --image-root /path/to/images --image-list /path/to/frame_names.txt --output-dir /path/to/priors --device cuda:0
```

## Detection evaluation scope

The manuscript uses the official VoD evaluation for the entire annotated area and driving corridor, and Dual-Radar Easy/Moderate/Hard evaluation, with Moderate used for detailed robustness analysis. Apply the official evaluators to predictions from a complete detector using the same partition and category conventions.
