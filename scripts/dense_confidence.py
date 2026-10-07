import argparse

from underwater_vision.pointcloud.dense_confidence import dense_confidence


def main():
    p = argparse.ArgumentParser(description="Attach depth-verified confidence to fused MVS cloud")
    p.add_argument("workspace")
    p.add_argument("--depth-tolerance", type=float, default=0.02)
    args = p.parse_args()
    print(dense_confidence(args.workspace, args.depth_tolerance))


if __name__ == "__main__":
    main()
