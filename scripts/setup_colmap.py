"""Install the pinned official Windows CUDA COLMAP binary into this project."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from download_data import download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="tools", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "colmap-x64-windows-cuda.zip"
    download(
        "https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip",
        target,
        414666626,
    )
    root = (args.output / "colmap").resolve()
    with zipfile.ZipFile(target) as archive:
        for entry in archive.infolist():
            if not (root / entry.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe release archive")
        archive.extractall(root)
    with target.open("rb") as handle:
        print("Local archive SHA256:", hashlib.file_digest(handle, "sha256").hexdigest())
    print(root / "bin" / "colmap.exe")


if __name__ == "__main__":
    main()
