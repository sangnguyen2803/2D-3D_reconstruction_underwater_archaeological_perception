"""All-pose label construction with unchanged v1 RGB/SIFT preprocessing."""

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.geometry.reference import camera_parameters, reference_labels
from underwater_vision.matchability.labels import LABEL_SPEC, label_counts
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.retrieval.pairs import retrieve_pairs
from underwater_vision.semantic.protocol import spatial_folds
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json, write_rgb
from underwater_vision.visibility.deterministic import estimate_visibility, sample_map

TABULAR_NAMES = ["response", "scale", "contrast", "gradient", "entropy", "exposure"]


def extract_local(frame, root):
    image, _ = resize_rgb(read_rgb(frame.path), 960)
    points, raw = cv2.SIFT_create(nfeatures=4096).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
    )
    xy = np.asarray([p.pt for p in points], np.float32).reshape(-1, 2)
    raw = np.empty((0, 128), np.float32) if raw is None else raw
    local = np.sqrt(raw / (raw.sum(1, keepdims=True) + 1e-12)).astype(np.float32)
    quality, cues = estimate_visibility(image)
    response = np.asarray([p.response for p in points], np.float32)
    scale = np.asarray([p.size for p in points], np.float32)
    stats = np.c_[
        response,
        scale,
        *[sample_map(cues[n], xy) for n in ["contrast", "texture", "entropy", "exposure"]],
    ].astype(np.float32)
    record = dict(
        xy=xy,
        local=local,
        response=response,
        scales=scale,
        statistics=stats,
        quality=sample_map(quality, xy),
        sharpness=sample_map(cues["sharpness"], xy),
        brightness=float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean() / 255),
        shape=np.asarray(image.shape[:2]),
    )
    atomic_npz(root / "features" / (frame.name + ".npz"), **record)
    write_rgb(root / "images" / (frame.name + ".png"), image)
    return frame.name, len(xy)


def load_features(root, name):
    with np.load(Path(root) / "features" / (name + ".npz")) as z:
        return {k: z[k] for k in z.files}


@torch.inference_mode()
def bidirectional_nearest(a, b, device="cuda"):
    if not len(a) or not len(b):
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32)
    ta, tb = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
    distance = (ta.square().sum(1)[:, None] + tb.square().sum(1)[None] - 2 * ta @ tb.T).clamp_min_(
        0
    )
    forward, reverse = distance.argmin(1).cpu().numpy(), distance.argmin(0).cpu().numpy()
    return np.c_[np.arange(len(a)), forward].astype(np.int32), np.c_[
        reverse, np.arange(len(b))
    ].astype(np.int32)


def aggregate_protocol(
    root, pairs, protocol, counts, graph="hybrid", positive=1.5, negative=4, sensitivity=True
):
    """Recompute pair eligibility for EACH positive threshold, within each slab."""
    groups = protocol["groups"]
    role = {n: s for s, names in groups.items() for n in names}
    selected = [n for n in counts if role[n] != "excluded"]
    offsets, total = {}, 0
    for n in selected:
        offsets[n] = total
        total += counts[n]
    pos_grid, neg_grid = (
        ([1.0, 1.5, 2.0], [3.0, 4.0, 6.0]) if sensitivity else ([positive], [negative])
    )
    connected = np.zeros((len(pos_grid), total), np.uint16)
    good = np.zeros_like(connected)
    bad = np.zeros((len(pos_grid), len(neg_grid), total), np.uint16)
    pair_counts = {
        s: {"candidate_pairs": 0, "eligible_by_positive_threshold": [0] * len(pos_grid)}
        for s in ["train", "validation", "test"]
    }
    hist = np.zeros(9, np.int64)
    bins = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10.0, 100.0, np.inf]
    for k, info in enumerate(pairs):
        a, b = info["a"], info["b"]
        if (
            role[a] != role[b]
            or role[a] == "excluded"
            or (graph == "sequential" and not info["sequential"])
        ):
            continue
        s = role[a]
        pair_counts[s]["candidate_pairs"] += 1
        with np.load(root / "pairs" / f"{k:06d}.npz") as z:
            indices = [z["forward"][:, 0], z["reverse"][:, 1]]
            errors = [z["forward_error"], z["reverse_error"]]
        for pi, p in enumerate(pos_grid):
            if sum(int(np.sum(e <= p)) for e in errors) < 15:
                continue
            pair_counts[s]["eligible_by_positive_threshold"][pi] += 1
            if p == positive:
                hist += np.histogram(np.r_[*errors], bins)[0]
            for n, ids, error in zip([a, b], indices, errors, strict=True):
                slots = offsets[n] + ids
                connected[pi, slots] += 1
                good[pi, slots] += error <= p
                for ni, neg in enumerate(neg_grid):
                    bad[pi, ni, slots] += error >= neg
    labels = {}
    summaries = {}
    for pi, pos in enumerate(pos_grid):
        for ni, neg in enumerate(neg_grid):
            y = np.full(total, -1, np.int8)
            y[(connected[pi] >= 2) & (bad[pi, ni] == connected[pi])] = 0
            y[good[pi] > 0] = 1
            summaries[f"pos_{pos:g}_neg_{neg:g}"] = {
                s: {
                    key: sum(
                        label_counts(y[offsets[n] : offsets[n] + counts[n]])[key] for n in groups[s]
                    )
                    for key in ["positive", "negative", "ignore"]
                }
                for s in ["train", "validation", "test"]
            }
            if pos == positive and neg == negative:
                labels["default"] = y
                denominator = good[pi] + bad[pi, ni]
                soft = np.divide(
                    good[pi], denominator, out=np.zeros(total, np.float32), where=denominator > 0
                )
                soft[y < 0] = -1
                labels["soft"] = soft
    default = labels["default"]
    # Stability counts relative to the frozen default; includes ignored transitions.
    stability = {}
    for pi, pos in enumerate(pos_grid):
        for ni, neg in enumerate(neg_grid):
            y = np.full(total, -1, np.int8)
            y[(connected[pi] >= 2) & (bad[pi, ni] == connected[pi])] = 0
            y[good[pi] > 0] = 1
            stability[f"pos_{pos:g}_neg_{neg:g}"] = float(np.mean(y == default))
    target = root / "labels" / protocol["name"] / graph
    target.mkdir(parents=True)
    for n in selected:
        interval = slice(offsets[n], offsets[n] + counts[n])
        atomic_npz(
            target / (n + ".npz"),
            l1=default[interval],
            soft=labels["soft"][interval],
            eligible_candidates=connected[pos_grid.index(positive), interval],
        )
    report = {
        "counts": summaries,
        "stability": stability,
        "pairs": pair_counts,
        "error_histogram": {
            "finite_bin_edges_px": bins[:-1],
            "last_bin_unbounded": True,
            "counts": hist.tolist(),
        },
        "graph": graph,
    }
    write_json(root / f"audit_{protocol['name']}_{graph}_v3.json", report)
    return report


def build_dataset(output, workers=4, window=6, retrieval=True):
    root = new_run(output)
    for name in ["features", "images", "pairs", "labels"]:
        (root / name).mkdir()
    data = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())
    frames = [f for f in MermaidDataset(data["root"]).frames if f.reference_c2w is not None]
    lookup = {f.name: f for f in frames}
    names = list(lookup)
    counts = {}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, count) in enumerate(pool.map(lambda f: extract_local(f, root), frames)):
            counts[name] = count
            if i % 50 == 0:
                print(f"RootSIFT RGB extraction {i + 1}/{len(frames)}", flush=True)
    sequential = {
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, min(len(names), i + window + 1))
    }
    graph = set(sequential)
    if retrieval:
        with np.load("outputs/retrieval_v1/retrieval_v1.npz") as z:
            index = {n: i for i, n in enumerate(z["image_names"].tolist())}
            descriptors = z["descriptors"][[index[n] for n in names]]
        retrieved, _ = retrieve_pairs(descriptors, 12)
        graph.update((names[i], names[j]) for i, j in retrieved)
    v1 = read_json("outputs/reference_audit_v1/spatial_split_v1.json")
    v1["name"] = "v1_partition"
    protocols = [v1]
    for fold in spatial_folds(names, np.stack([lookup[n].reference_c2w[:3, 3] for n in names])):
        fold["name"] = f"fold{fold['fold']}"
        protocols.append(fold)
    # Never even compute candidates if no protocol can assign this pair to one slab.
    maps = [{n: s for s, ns in protocol["groups"].items() for n in ns} for protocol in protocols]
    graph = [
        (a, b) for a, b in sorted(graph) if any(m[a] == m[b] and m[a] != "excluded" for m in maps)
    ]
    pairs = []
    cache = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    for i, (a, b) in enumerate(graph):
        for name in [a, b]:
            if name not in cache:
                cache[name] = load_features(root, name)
        fa, fb = cache[a], cache[b]
        forward, reverse = bidirectional_nearest(fa["local"], fb["local"], device)
        h, w = map(int, fa["shape"])
        K, distortion = camera_parameters(w, h, data["calibration"])
        errors = []
        for ids in [forward, reverse]:
            _, error, _ = reference_labels(
                fa["xy"][ids[:, 0]],
                fb["xy"][ids[:, 1]],
                lookup[a].reference_c2w,
                lookup[b].reference_c2w,
                K,
                distortion,
            )
            errors.append(error.astype(np.float32))
        atomic_npz(
            root / "pairs" / f"{i:06d}.npz",
            forward=forward,
            reverse=reverse,
            forward_error=errors[0],
            reverse_error=errors[1],
        )
        pairs.append({"a": a, "b": b, "sequential": (a, b) in sequential})
        if len(cache) > 48:
            cache.pop(next(iter(cache)))
        if i % 100 == 0:
            print(f"Bidirectional reference pairs {i + 1}/{len(graph)}", flush=True)
    write_json(root / "pairs_v3.json", pairs)
    manifest = {
        "label_spec": LABEL_SPEC,
        "image_size": [720, 960],
        "requested_features": 4096,
        "enhance": False,
        "thresholds": {"positive": 1.5, "negative": 4.0, "n_min": 2, "eligible_minimum": 15},
        "pair_rule": "15 compatible directed top1 candidates across both directions; pairs within same slab",
        "window": window,
        "protocols": protocols,
        "images": [
            {
                "name": f.name,
                "path": str(f.path.resolve()),
                "bytes": f.path.stat().st_size,
                "mtime_ns": f.path.stat().st_mtime_ns,
                "features": counts[f.name],
            }
            for f in frames
        ],
        "tabular_names": TABULAR_NAMES,
        "retrieval_sha256": digest("outputs/retrieval_v1/retrieval_v1.npz") if retrieval else None,
    }
    write_json(root / "manifest_v3.json", manifest)
    audits = {}
    for protocol, kind in itertools.product(
        protocols, ["sequential", "hybrid"] if retrieval else ["sequential"]
    ):
        audits[f"{protocol['name']}_{kind}"] = aggregate_protocol(
            root, pairs, protocol, counts, kind
        )
        print(f"Audited {protocol['name']} {kind}", flush=True)
    write_json(
        root / "audit_v3.json",
        {
            "status": "complete",
            "images": len(frames),
            "audits": audits,
            "label_limitations": "SIFT-selected, top1 matcher dependent; published BA geometry is not independently surveyed truth; no RANSAC supervision",
        },
    )
    return root
