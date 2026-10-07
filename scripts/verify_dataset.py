"""Compare extracted original images with the official archive member CRCs."""

import argparse
import json
import zipfile
import zlib
from pathlib import Path

from underwater_vision.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/mermaid", type=Path)
    args = parser.parse_args()
    checked = 0
    with zipfile.ZipFile(args.root / "107179.zip") as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith(".jpg") or "__MACOSX" in member.filename:
                continue
            path = args.root / member.filename
            if path.stat().st_size != member.file_size:
                raise ValueError(f"Extracted size mismatch: {path}")
            checksum = 0
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024**2), b""):
                    checksum = zlib.crc32(block, checksum)
            if checksum != member.CRC:
                raise ValueError(f"Extracted CRC mismatch: {path}")
            checked += 1
    result = {"images_verified": checked, "size_and_crc_match_archive": True}
    write_json("outputs/dataset_integrity.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
