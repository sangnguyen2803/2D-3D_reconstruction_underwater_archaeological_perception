"""Validate post-SfM dense artifacts and append measured results to a run."""

import argparse
import json
from pathlib import Path

from plyfile import PlyData

from underwater_vision.pointcloud.dense_confidence import dense_confidence
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    dense = args.run / "dense"
    mesh = PlyData.read(str(dense / "mesh.ply"))
    vertices, faces = len(mesh["vertex"]), len(mesh["face"])
    if vertices == 0 or faces == 0:
        raise ValueError("Dense mesh is empty")
    statistics = dense_confidence(dense)
    statistics.update(
        mesh_vertices=vertices,
        mesh_faces=faces,
        dense_status="completed",
        dense_backend="COLMAP 4.2.1 CUDA",
        meshing={"method": "Poisson", "depth": 9, "trim": 0, "threads": 4},
    )
    write_json(dense / "mvs_metrics.json", statistics)
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    metrics.update(statistics)
    write_json(args.run / "metrics.json", metrics)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
