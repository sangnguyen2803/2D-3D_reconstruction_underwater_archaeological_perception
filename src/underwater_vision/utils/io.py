import hashlib
import json
import uuid
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    data.tofile(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path, settings):
    path = Path(path)
    state = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime_ns,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:24]


def atomic_npz(path, **values):
    """Atomic shared-cache writes prevent a second process reading partial NPZ data."""
    path = Path(path)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **values)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
