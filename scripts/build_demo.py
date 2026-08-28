"""Portable offline comparison with camera-aligned sparse/dense interactive clouds."""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plyfile import PlyData

from underwater_vision.evaluation.metrics import similarity_alignment


def load_poses(run):
    return {
        n: np.asarray(p) for n, p in json.loads((run / "camera_poses.json").read_text()).items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/demo.html"))
    args = p.parse_args()
    titles = [r.name + " — " + kind for kind in ["Sparse", "Dense"] for r in args.runs]
    fig = make_subplots(
        rows=2,
        cols=len(args.runs),
        specs=[[{"type": "scene"}] * len(args.runs)] * 2,
        subplot_titles=titles,
    )
    content = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Underwater Reconstruction</title>',
        "<style>body{font:16px system-ui;background:#071923;color:#e1f0f3;margin:32px}img{max-width:100%}section{margin:36px 0}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #466}</style>",
        "<h1>Visibility-Aware Underwater Reconstruction</h1>",
        "<p>Measured Mermaid development runs. Clouds are aligned to the first run using common camera centers. Confidence is a heuristic evidence score.</p>",
        "<table><tr><th>Run</th><th>Cameras</th><th>Sparse points</th><th>Dense points</th><th>Reprojection px</th><th>Inlier ratio</th></tr>",
    ]
    target_poses = load_poses(args.runs[0])
    rgb_colors, confidence_colors = [], []
    for index, run in enumerate(args.runs):
        m = json.loads((run / "metrics.json").read_text())
        content.append(
            f'<tr><td>{html.escape(run.name)}</td><td>{m["registered_images"]}</td><td>{m["sparse_points"]}</td><td>{m.get("dense_points","Unavailable")}</td><td>{m["mean_reprojection_error_px"]:.4f}</td><td>{m["inlier_ratio"]:.4f}</td></tr>'
        )
        poses = load_poses(run)
        common = sorted(set(poses) & set(target_poses))
        if index:
            scale, rotation, translation = similarity_alignment(
                np.array([poses[n][:3, 3] for n in common]),
                np.array([target_poses[n][:3, 3] for n in common]),
            )
        else:
            scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
        for row in [1, 2]:
            if row == 1:
                with np.load(run / "sparse_confidence.npz") as cloud:
                    xyz, rgb, confidence = cloud["xyz"], cloud["rgb"], cloud["confidence"]
            else:
                path = run / "dense" / "fused_confidence.ply"
                if not path.exists():
                    continue
                vertex = PlyData.read(str(path))["vertex"]
                xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]]
                rgb = np.c_[vertex["red"], vertex["green"], vertex["blue"]]
                confidence = vertex["confidence"]
            step = max(1, (len(xyz) + 24999) // 25000)
            xyz = scale * (xyz[::step] @ rotation.T) + translation
            confidence = confidence[::step]
            colors = [f"rgb({r},{g},{b})" for r, g, b in rgb[::step]]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker=dict(size=1.5, color=confidence, colorscale="Viridis", cmin=0, cmax=1),
                    name=run.name,
                    text=[f"Confidence: {c:.3f}" for c in confidence],
                ),
                row=row,
                col=index + 1,
            )
            rgb_colors.append(colors)
            confidence_colors.append(confidence.tolist())
            centers = np.array([poses[n][:3, 3] for n in sorted(poses)])
            centers = scale * (centers @ rotation.T) + translation
            fig.add_trace(
                go.Scatter3d(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    z=centers[:, 2],
                    mode="lines+markers",
                    marker=dict(size=3, color="red"),
                    line=dict(color="red"),
                    name="Camera trajectory",
                ),
                row=row,
                col=index + 1,
            )
            rgb_colors.append("red")
            confidence_colors.append("red")
    content.append("</table>")
    fig.update_layout(
        height=1050,
        template="plotly_dark",
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        label="Confidence",
                        method="restyle",
                        args=[{"marker.color": confidence_colors}],
                    ),
                    dict(
                        label="Original RGB", method="restyle", args=[{"marker.color": rgb_colors}]
                    ),
                ],
            )
        ],
    )
    fig.update_scenes(aspectmode="data")
    content.append(fig.to_html(full_html=False, include_plotlyjs=True))
    for run in args.runs:
        content.append("<section><h2>" + html.escape(run.name) + "</h2>")
        for image in sorted((run / "figures").glob("*.png")):
            content.append(
                f'<h3>{html.escape(image.stem)}</h3><img src="data:image/png;base64,{base64.b64encode(image.read_bytes()).decode()}">'
            )
        content.append("</section>")
    content.append(
        "<p>Camera references were estimated by bundle adjustment. Independent dense surface ground truth is unavailable. Display clouds are sampled to at most 25,000 points; exported PLY files retain every point.</p></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(content), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
