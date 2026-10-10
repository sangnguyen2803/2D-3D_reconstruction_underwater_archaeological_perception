"""Write a supported mesh beside immutable original COLMAP reconstruction artifacts."""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from underwater_vision.matchability.protocol import digest
from underwater_vision.pointcloud.mesh_cleanup import trim_mesh
from underwater_vision.utils.io import write_json


def xyz(data):
    return np.column_stack([data[k] for k in ["x", "y", "z"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-factor", type=float, default=3.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mesh_hash, cloud_hash = digest(args.mesh), digest(args.cloud)
    mesh, cloud = PlyData.read(str(args.mesh)), PlyData.read(str(args.cloud))
    used, faces, measured = trim_mesh(
        xyz(mesh["vertex"]),
        np.stack(mesh["face"]["vertex_indices"]),
        xyz(cloud["vertex"]),
        radius_factor=args.radius_factor,
    )
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    result = args.output / "mesh_supported_v1.ply"
    PlyData(
        [
            PlyElement.describe(mesh["vertex"].data[used], "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    ).write(str(result))
    assert digest(args.mesh) == mesh_hash and digest(args.cloud) == cloud_hash
    measured.update(
        source_mesh=str(args.mesh),
        source_mesh_sha256=mesh_hash,
        source_cloud=str(args.cloud),
        source_cloud_sha256=cloud_hash,
        output_mesh_sha256=digest(result),
    )
    write_json(args.output / "comparison_v1.json", measured)
    print(measured)


if __name__ == "__main__":
    main()
