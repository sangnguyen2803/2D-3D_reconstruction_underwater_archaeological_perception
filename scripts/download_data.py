"""Download official Mermaid assets, with validated Range resume and ZIP CRC checks."""

import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SOURCE = "https://www.seanoe.org/data/00868/97987/"


def download_parallel(url, target, expected, workers=6):
    """Each range has its own resumable file; assemble only verified ranges."""
    target = Path(target)
    if target.exists() and target.stat().st_size == expected:
        return
    chunk_size = 64 * 1024**2
    chunk_root = target.parent / (target.name + ".chunks")
    chunk_root.mkdir(exist_ok=True)

    def fetch(index):
        start = index * chunk_size
        end = min(expected, start + chunk_size) - 1
        path = chunk_root / f"{index:04d}.part"
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            if have == end - start + 1:
                return path
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={start+have}-{end}"},
                    stream=True,
                    timeout=(30, 120),
                ) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(
                        f"bytes {start+have}-{end}/"
                    ):
                        raise ValueError("Range requests not supported correctly")
                    with path.open("ab") as f:
                        for block in r.iter_content(1024**2):
                            f.write(block)
                if path.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                print(
                    f"Range {index + 1}/{(expected+chunk_size-1)//chunk_size} complete", flush=True
                )
                return path
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 15))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, range((expected + chunk_size - 1) // chunk_size)))
    combined = target.with_suffix(target.suffix + ".assembling")
    with combined.open("wb") as out:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, out, length=1024**2)
    if combined.stat().st_size != expected:
        raise ValueError("Assembled archive size mismatch")
    combined.replace(target)
    # Delete only the enumerated range files in this download directory.
    for part in parts:
        part.unlink()
    chunk_root.rmdir()


def download(url, target, expected):
    target = Path(target)
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected:
        return
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected:
            part.replace(target)
            return
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={offset}-"} if offset else {},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an invalid resume offset")
                    mode = "ab"
                else:
                    mode = "wb"
                    offset = 0
                with part.open(mode) as handle:
                    last = time.monotonic()
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
                        offset += len(block)
                        if time.monotonic() - last > 15:
                            print(
                                f"{target.name}: {offset / 1e6:.1f}/{expected / 1e6:.1f} MB",
                                flush=True,
                            )
                            last = time.monotonic()
            if part.stat().st_size != expected:
                raise ValueError(f"Incomplete file: {part.stat().st_size} != {expected}")
            part.replace(target)
            return
        except (requests.RequestException, ValueError) as exc:
            print(f"Attempt {attempt + 1}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 15))


def extract_archive(target, root):
    """Extract CRC-checked members, including explicit ZIP directory entries."""
    root = Path(root).resolve()
    with zipfile.ZipFile(target) as archive:
        for entry in archive.infolist():
            destination = (root / entry.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Unsafe ZIP member")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not destination.exists() or destination.stat().st_size != entry.file_size:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".extracting")
                try:
                    with archive.open(entry) as source, temporary.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/mermaid")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE, timeout=60)
    response.raise_for_status()
    match = re.search(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', response.text, re.S
    )
    if not match:
        raise RuntimeError("Official page format changed; inspect SEANOE download links")
    source = json.loads(match.group(1))
    (root / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    manifest = []
    for item in source["distribution"]:
        url = item["contentUrl"]
        name = url.rsplit("/", 1)[-1]
        if args.metadata_only and name.endswith(".zip"):
            continue
        size = int(item["fileSize"])
        if shutil.disk_usage(root).free < size * 2 + 1024**3:
            raise RuntimeError("Insufficient free space for download and extraction")
        print(f"Downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        target = root / name
        if name.endswith(".zip") and args.workers > 1:
            download_parallel(url, target, size, args.workers)
        else:
            download(url, target, size)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest.append({"url": url, "file": name, "bytes": size, "sha256": digest})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if name.endswith(".zip") and not args.no_extract:
            extract_archive(target, root)
    print("Download completed.", flush=True)


if __name__ == "__main__":
    main()
