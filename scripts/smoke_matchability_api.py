"""Real HTTP smoke test against the launched Docker or local matchability API."""

import argparse
import base64
import io
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from underwater_vision.matchability.protocol import new_run
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8003")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = new_run(a.output)
    health = requests.get(a.url + "/health", timeout=30)
    health.raise_for_status()
    start = time.perf_counter()
    with a.image.open("rb") as file:
        response = requests.post(
            a.url + "/predict?top_n=32",
            files={"image": (a.image.name, file, "image/png")},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    decoded = base64.b64decode(data["matchability_png_base64"])
    image = np.asarray(Image.open(io.BytesIO(decoded)))
    if image.shape != (data["height"], data["width"]) or not data.get("keypoints"):
        raise ValueError("Unexpected map shape or absent top-N keypoints")
    (root / "api_matchability_v3.png").write_bytes(decoded)
    bad = requests.post(
        a.url + "/predict", files={"image": ("bad.png", b"invalid", "image/png")}, timeout=30
    )
    report = {
        "status": "passed",
        "url": a.url,
        "health": health.json(),
        "checkpoint_hash": data["checkpoint_sha256"],
        "map_shape": list(image.shape),
        "requested_top_n": 32,
        "returned_keypoints": len(data["keypoints"]),
        "statistics": data["statistics"],
        "invalid_image_http_status": bad.status_code,
        "one_request_seconds": time.perf_counter() - start,
        "timing_scope": "One smoke request including HTTP/decoding/resizing/model/top-N; not a latency benchmark",
    }
    if bad.status_code != 400:
        raise ValueError("Invalid image must return HTTP 400")
    write_json(root / "api_smoke_v3.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
