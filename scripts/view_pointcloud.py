"""Open a colored/confidence cloud locally in Open3D."""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--confidence", action="store_true")
    args = p.parse_args()
    import open3d as o3d
    from plyfile import PlyData

    vertex = PlyData.read(args.path)["vertex"]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.c_[vertex["x"], vertex["y"], vertex["z"]])
    if args.confidence:
        import matplotlib.pyplot as plt

        cloud.colors = o3d.utility.Vector3dVector(plt.cm.viridis(vertex["confidence"])[:, :3])
    else:
        cloud.colors = o3d.utility.Vector3dVector(
            np.c_[vertex["red"], vertex["green"], vertex["blue"]] / 255
        )
    o3d.visualization.draw_geometries([cloud], window_name="Underwater 3D evidence confidence")


if __name__ == "__main__":
    main()
