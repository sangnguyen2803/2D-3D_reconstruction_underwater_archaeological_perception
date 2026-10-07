"""Extract selected real Mermaid images by HTTP ranges while full ZIP downloads."""

import argparse
import io
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        retry = Retry(
            total=5, connect=5, read=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        end = self.position + size - 1
        r = self.session.get(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}, timeout=120
        )
        r.raise_for_status()
        if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(
            f"bytes {self.position}-{end}/"
        ):
            raise ValueError("Invalid HTTP range")
        self.position += len(r.content)
        return r.content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", default="data/raw/mermaid")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        RemoteZip("https://www.seanoe.org/data/00868/97987/data/107179.zip", 3690513210)
    ) as archive:
        entries = sorted(
            [
                x
                for x in archive.infolist()
                if x.filename.lower().endswith(".jpg")
                and "__MACOSX" not in x.filename
                and not Path(x.filename).name.startswith(".")
            ],
            key=lambda x: x.filename,
        )
        print(f"Official ZIP contains {len(entries)} JPEG images", flush=True)
        for entry in entries[args.start :: args.stride][: args.count]:
            destination = (root / entry.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Unsafe archive entry")
            if not destination.exists() or destination.stat().st_size != entry.file_size:
                # Write atomically: a disconnected response must not expose a partial JPEG.
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".downloading")
                with archive.open(entry) as source, temporary.open("wb") as output:
                    import shutil

                    shutil.copyfileobj(source, output)
                temporary.replace(destination)
            print(entry.filename, flush=True)


if __name__ == "__main__":
    main()
