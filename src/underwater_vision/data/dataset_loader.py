"""Mermaid images and published bundle-adjustment reference poses.

No semantic labels or dense ground truth are fabricated. The reference STL is
only a micro-geodesic control network, not a surveyed surface of the scene.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class Frame:
    path: Path
    name: str
    reference_c2w: np.ndarray | None


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(path))]


def load_reference_poses(path):
    result = {}
    for camera in ElementTree.parse(path).getroot().findall("camera"):
        values = np.fromstring(camera.findtext("transform", ""), sep=" ")
        if values.size != 16:
            raise ValueError(f"Invalid transform for {camera.attrib}")
        transform = values.reshape(4, 4)
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
        if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Invalid reference rotation")
        result[camera.attrib["label"]] = transform
    return result


class MermaidDataset:
    def __init__(self, root, image_dir=None):
        self.root = Path(root)
        xml = self.root / "107177.xml"
        self.poses = load_reference_poses(xml) if xml.exists() else {}
        folder = Path(image_dir) if image_dir else self.root
        paths = sorted(
            [
                p
                for p in folder.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
                and not any(
                    part.startswith(".") or part == "__MACOSX"
                    for part in p.relative_to(folder).parts
                )
            ],
            key=natural_key,
        )
        if not paths:
            raise FileNotFoundError(f"No images in {folder}. Run scripts/download_data.py first.")
        if len({p.name for p in paths}) != len(paths):
            raise ValueError("Duplicate image names; select a single acquisition image directory")
        self.frames = [Frame(p, p.name, self.poses.get(p.stem)) for p in paths]

    def select(self, start=0, count=None, stride=1):
        if start < 0 or stride < 1 or (count is not None and count < 1):
            raise ValueError("start >= 0, stride >= 1 and count >= 1 required")
        frames = self.frames[start::stride]
        return frames[:count] if count else frames


def trajectory_split(frames, train_fraction=0.6, gap=20):
    """Contiguous blocks with discarded boundary frames, never random adjacent views."""
    if not 0 < train_fraction < 1 or gap < 0:
        raise ValueError("Invalid split settings")
    boundary = int(len(frames) * train_fraction)
    return frames[: max(0, boundary - gap)], frames[min(len(frames), boundary + gap) :]
