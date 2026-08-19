"""Engineering-only export smoke: pretrained encoder, UNTRAINED six-class head."""

from pathlib import Path

import torch
import yaml

from underwater_vision.semantic.model import SegmentationModel
from underwater_vision.semantic.protocol import new_run
from underwater_vision.serving.export import export_onnx


def main():
    root = new_run("outputs/semantic_deployment_smoke_v2")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    specification = yaml.safe_load(Path("configs/semantic/model/segformer_b0.yaml").read_text())
    model = SegmentationModel(specification, pretrained=True).eval()
    specification["hf_config"] = model.network.config.to_dict()
    specification["hf_config"].pop("num_labels", None)
    checkpoint = root / "untrained_smoke_v2.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_spec": "segmentation_six_v2",
            "model_config": specification,
            "size": [384, 512],
            "temperature": 1.0,
            "status": "untrained_engineering_smoke",
            "training_images": [],
            "validation_images": [],
        },
        checkpoint,
    )
    result = export_onnx(checkpoint, "outputs/semantic_onnx_smoke_v2")
    print({k: v for k, v in result.items() if k != "timing"}, flush=True)
    print(result["timing"], flush=True)


if __name__ == "__main__":
    main()
