"""Cache pretrained official frozen DINOv2 B/14 (optionally S/14) weights."""

import argparse

from underwater_vision.features.dinov2 import DinoExtractor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backbone", default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vits14"]
    )
    p.add_argument("--revision", default="7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    args = p.parse_args()
    extractor = DinoExtractor(args.backbone, "cpu", args.revision)
    print(extractor.provenance)


if __name__ == "__main__":
    main()
