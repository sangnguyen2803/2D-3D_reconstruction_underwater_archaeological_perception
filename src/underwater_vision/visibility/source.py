"""One explicit reliability source, detachable from matchability feature inputs."""

import numpy as np


def validate_source_images(configuration, names, evaluation_mode=True):
    """Hash weights and reject fitted/out-of-slab reliability evaluation images."""
    result = dict(configuration)
    if result.get("source") == "dino_head":
        from underwater_vision.matchability.protocol import check_evaluation, digest
        from underwater_vision.visibility.learned import load_head

        _, metadata = load_head(result.get("checkpoint"))
        result["checkpoint_sha256"] = digest(result["checkpoint"])
        if evaluation_mode:
            used = set(metadata.get("training_images", []) + metadata.get("validation_images", []))
            if used & set(names):
                raise ValueError("DINO reliability source training/validation leakage")
            if "spatial_split" in metadata:
                check_evaluation(
                    {"validation_images": [], **metadata}, names, metadata["spatial_split"]
                )
    return result


def apply_source(record, source="handcrafted", checkpoint=None):
    result = dict(record)
    if source == "handcrafted":
        result["reliability"] = np.asarray(record["quality"], np.float32).copy()
    elif source == "matchability":
        if "matchability" not in record:
            raise ValueError("Matchability source requires a map provider")
        result["reliability"] = np.asarray(record["matchability"], np.float32).copy()
    elif source == "dino_head":
        from underwater_vision.visibility.learned import predict

        result["reliability"] = predict(record["semantic"], checkpoint, calibrated=True)
    else:
        raise ValueError("visibility.source must be handcrafted, dino_head or matchability")
    return result
