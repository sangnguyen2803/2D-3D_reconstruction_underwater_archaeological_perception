import argparse
from pathlib import Path

from underwater_vision.reconstruction.colmap.mvs import run_mvs


def main():
    p = argparse.ArgumentParser(description="Run COLMAP CUDA MVS on an existing sparse model")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", default="0")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--max-size", type=int, default=960)
    p.add_argument("--no-mesh", action="store_true")
    args = p.parse_args()
    run_mvs(
        args.run / "sparse" / args.model,
        args.run / "images",
        args.run / "dense",
        args.colmap,
        args.max_size,
        not args.no_mesh,
    )


if __name__ == "__main__":
    main()
