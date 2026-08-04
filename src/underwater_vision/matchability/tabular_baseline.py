"""Strong statistical controls and a freshly fitted frozen-DINO MLP per fold."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from underwater_vision.evaluation.matchability import metrics
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.calibration import fit_temperature, probabilities
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json
from underwater_vision.visibility.learned import ReliabilityHead


class DinoFeatureCache:
    """Cache sampled descriptors rather than dense grids to fit local disk limits."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = Path("cache/matchability_dino_features_v3")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.extractor = None

    def descriptors(self, name, xy):
        cache = self.cache / (name + ".npz")
        image_path = self.root / "images" / (name + ".png")
        # The image digest, fixed pretrained revision and padding convention are checked.
        image_hash = digest(image_path)
        all_xy = load_features(self.root, name)["xy"]
        descriptors = None
        if cache.exists():
            with np.load(cache) as z:
                if (
                    str(z["image_hash"]) == image_hash
                    and str(z["revision"]) == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
                    and np.array_equal(z["xy"], all_xy)
                ):
                    descriptors = z["descriptors"].astype(np.float32)
        if descriptors is None:
            if self.extractor is None:
                self.extractor = DinoExtractor(revision="7764ea0f912e53c92e82eb78a2a1631e92725fc8")
            descriptors = self.extractor.extract(read_rgb(image_path), all_xy)
            atomic_npz(
                cache,
                descriptors=descriptors.astype(np.float16),
                xy=all_xy,
                image_hash=np.asarray(image_hash),
                revision=np.asarray("7764ea0f912e53c92e82eb78a2a1631e92725fc8"),
            )
            # Use the same cached precision on both initial and repeat inference.
            descriptors = descriptors.astype(np.float16).astype(np.float32)
        if np.array_equal(xy, all_xy):
            selected = descriptors
        else:
            lookup = {tuple(point): i for i, point in enumerate(all_xy)}
            selected = descriptors[[lookup[tuple(point)] for point in xy]]
        return F.normalize(torch.from_numpy(selected), dim=1).numpy()


def train_baselines(dataset, protocol, output, seed=42, graph="hybrid", sample_per_image=512):
    import joblib

    root = new_run(output)
    source = Path(dataset)
    manifest = read_json(source / "manifest_v3.json")
    split = next(p for p in manifest["protocols"] if p["name"] == protocol)
    rng = np.random.default_rng(seed)
    cache = DinoFeatureCache(source)
    records = {}
    for role, names in split["groups"].items():
        if role == "excluded":
            continue
        rows = []
        for i, name in enumerate(names):
            feature = load_features(source, name)
            with np.load(source / "labels" / protocol / graph / (name + ".npz")) as z:
                y = z["l1"]
            indices = np.flatnonzero(y >= 0)
            # Sample only fitting/selection rows; score every test keypoint.
            if role != "test" and len(indices) > sample_per_image:
                indices = rng.choice(indices, sample_per_image, replace=False)
            else:
                indices = np.arange(len(y))
            rows.append(
                {
                    "name": name,
                    "labels": y[indices],
                    "statistics": feature["statistics"][indices],
                    "quality": feature["quality"][indices],
                    "dino": cache.descriptors(name, feature["xy"][indices]),
                }
            )
            if i % 50 == 0:
                print(f"Baselines {protocol} {role} {i + 1}/{len(names)}", flush=True)
        records[role] = rows
    train = records["train"]
    x, y = np.concatenate([r["statistics"] for r in train]), np.concatenate(
        [r["labels"] for r in train]
    )
    known = y >= 0
    models = {
        "scale_response": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=200, random_state=seed)
        ),
        "tabular_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=200, random_state=seed)
        ),
        "tabular_boosting": HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=15, l2_regularization=1, random_state=seed
        ),
    }
    for name, model in models.items():
        model.fit(x[known, :2] if name == "scale_response" else x[known], y[known])
        joblib.dump(model, root / (name + "_v3.joblib"))
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = ReliabilityHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003)
    dino_x = torch.from_numpy(np.concatenate([r["dino"] for r in train])[known]).to(device)
    dino_y = torch.from_numpy(y[known].astype(np.float32)).to(device)
    best, state = -1.0, None
    validation_y = np.concatenate([r["labels"] for r in records["validation"]])

    @torch.inference_mode()
    def dino_logits(rows):
        return [head(torch.from_numpy(r["dino"]).to(device)).cpu().numpy() for r in rows]

    for _epoch in range(10):
        head.train()
        for idx in torch.randperm(len(dino_y), device=device).split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(dino_x[idx]), dino_y[idx])
            loss.backward()
            optimizer.step()
        head.eval()
        z = np.concatenate(dino_logits(records["validation"]))
        ap = metrics(validation_y, probabilities(z))["average_precision"]
        if ap > best:
            best, state = ap, {k: v.cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    validation_z = np.concatenate(dino_logits(records["validation"]))
    temperature = fit_temperature(validation_z, validation_y)
    torch.save(
        {
            "state_dict": state,
            "channels": 768,
            "backbone": "dinov2_vitb14",
            "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "temperature": temperature,
            "training_images": split["groups"]["train"],
            "validation_images": split["groups"]["validation"],
            "spatial_split": split,
            "label_spec": manifest["label_spec"],
        },
        root / "dino_reliability_v3.pt",
    )
    (root / "predictions").mkdir()
    predictions = []
    for row, z in zip(records["test"], dino_logits(records["test"]), strict=True):
        stats = row["statistics"]
        result = {
            "name": np.asarray(row["name"]),
            "labels": row["labels"],
            "random": rng.random(len(z)).astype(np.float32),
            "response": np.clip(stats[:, 0] / 0.1, 0, 1),
            "handcrafted": row["quality"],
            "dino_mlp": probabilities(z, temperature),
        }
        for name, model in models.items():
            result[name] = model.predict_proba(stats[:, :2] if name == "scale_response" else stats)[
                :, 1
            ].astype(np.float32)
        atomic_npz(root / "predictions" / (row["name"] + ".npz"), **result)
        predictions.append(result)
    names = [
        "random",
        "response",
        "scale_response",
        "handcrafted",
        "dino_mlp",
        "tabular_logistic",
        "tabular_boosting",
    ]
    labels = np.concatenate([r["labels"] for r in predictions])
    report = {
        "status": "complete",
        "seed": seed,
        "protocol": protocol,
        "manifest_sha256": digest(source / "manifest_v3.json"),
        "label_spec": manifest["label_spec"],
        "fitting_sample_per_image": sample_per_image,
        "test_scoring": "all detected keypoints; ignored labels excluded by metrics",
        "models": {
            name: metrics(labels, np.concatenate([r[name] for r in predictions])) for name in names
        },
        "training_images": split["groups"]["train"],
        "validation_images": split["groups"]["validation"],
        "test_images": split["groups"]["test"],
    }
    write_json(root / "evaluation_v3.json", report)
    return root
