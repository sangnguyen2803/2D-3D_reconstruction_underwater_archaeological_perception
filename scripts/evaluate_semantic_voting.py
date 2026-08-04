"""Q5: fuse all other views and compare to a held-out hand-labeled image."""

import argparse
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

from underwater_vision.semantic.infer import SemanticPredictor
from underwater_vision.semantic.masks import sample_probabilities
from underwater_vision.semantic.protocol import load_manifest, new_run
from underwater_vision.semantic.voting import evaluate_label_holdouts
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_voting_evaluation_v2"))
    a = p.parse_args()
    m = load_manifest(a.manifest, require_frozen=True)
    entries = {e["image_id"]: e for e in m["entries"] if e["role"] == "test"}
    model = pycolmap.Reconstruction(str(a.run / "sparse/0"))
    semantic = SemanticPredictor(a.checkpoint)
    observed = {i.name for i in model.images.values()}
    if observed & set(
        semantic.metadata["training_images"] + semantic.metadata["validation_images"]
    ):
        raise ValueError("Semantic voting test leaks segmentation training/validation views")
    masks = {n: np.asarray(Image.open(e["mask"])) for n, e in entries.items() if n in observed}
    probabilities = {n: semantic.cached(a.run / "images" / n) for n in observed}
    tracks = []
    for point in model.points3D.values():
        image_ids, xy, p = [], [], []
        for element in point.track.elements:
            image = model.images[element.image_id]
            coordinate = np.asarray(image.points2D[element.point2D_idx].xy) - 0.5
            values, _ = sample_probabilities(probabilities[image.name], [coordinate])
            if image.name in masks:
                height, width = probabilities[image.name].shape[1:]
                mh, mw = masks[image.name].shape
                coordinate = (coordinate + 0.5) * [mw / width, mh / height] - 0.5
            image_ids.append(image.name)
            xy.append(coordinate)
            p.append(values[0])
        if len(set(image_ids)) >= 2:
            tracks.append({"image_ids": image_ids, "xy": xy, "probabilities": p})
    root = new_run(a.output)
    write_json(root / "evaluation_v2.json", evaluate_label_holdouts(tracks, masks))


if __name__ == "__main__":
    main()
