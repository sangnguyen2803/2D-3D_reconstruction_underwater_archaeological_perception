from pathlib import Path

import numpy as np

from underwater_vision.data.dataset_loader import (
    Frame,
    MermaidDataset,
    load_reference_poses,
    trajectory_split,
)
from underwater_vision.preprocessing.images import resize_rgb


def test_xml_pose_loader(tmp_path):
    xml = tmp_path / "poses.xml"
    xml.write_text(
        '<camera_poses><camera label="frame"><transform>1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1</transform></camera></camera_poses>'
    )
    np.testing.assert_equal(load_reference_poses(xml)["frame"][:3, 3], [2, 3, 4])


def test_split_has_gap():
    frames = [Frame(Path(str(i)), str(i), None) for i in range(100)]
    train, test = trajectory_split(frames, gap=10)
    assert train[-1].name == "49"
    assert test[0].name == "70"
    assert not set(f.name for f in train) & set(f.name for f in test)


def test_resize_exact_scale():
    image, (sx, sy) = resize_rgb(np.zeros((2880, 3840, 3), np.uint8), 960)
    assert image.shape == (720, 960, 3)
    assert sx == sy == 0.25


def test_loader_excludes_macos_resource_forks(tmp_path):
    from PIL import Image

    real = tmp_path / "images" / "frame_0001.JPG"
    real.parent.mkdir()
    Image.new("RGB", (20, 20)).save(real)
    metadata = tmp_path / "__MACOSX" / "images" / "._frame_0001.JPG"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"macOS metadata, not a JPEG")
    dataset = MermaidDataset(tmp_path)
    assert [frame.name for frame in dataset.frames] == ["frame_0001.JPG"]
