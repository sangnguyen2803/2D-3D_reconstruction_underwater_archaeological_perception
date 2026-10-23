from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

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


def test_loader_ignores_directories_with_image_extensions(tmp_path):
    directory = tmp_path / "archive.jpg"
    directory.mkdir()
    image = directory / "frame_0001.JPG"
    image.write_bytes(b"image placeholder")
    dataset = MermaidDataset(tmp_path)
    assert [frame.path for frame in dataset.frames] == [image]


def _write_reference_poses(path, cameras):
    root = ElementTree.Element("camera_poses")
    for label, transform in cameras:
        camera = ElementTree.SubElement(root, "camera", {} if label is None else {"label": label})
        ElementTree.SubElement(camera, "transform").text = " ".join(map(str, transform.ravel()))
    ElementTree.ElementTree(root).write(path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_reference_poses_reject_nonfinite_translation(tmp_path, value):
    transform = np.eye(4)
    transform[0, 3] = value
    path = tmp_path / "poses.xml"
    _write_reference_poses(path, [("frame", transform)])
    with pytest.raises(ValueError, match="must be finite"):
        load_reference_poses(path)


def test_reference_poses_reject_reflections(tmp_path):
    path = tmp_path / "poses.xml"
    _write_reference_poses(path, [("frame", np.diag([-1, 1, 1, 1]))])
    with pytest.raises(ValueError, match="Invalid reference rotation"):
        load_reference_poses(path)


@pytest.mark.parametrize("label", [None, "", "frame"])
def test_reference_poses_reject_missing_or_duplicate_labels(tmp_path, label):
    path = tmp_path / "poses.xml"
    _write_reference_poses(path, [("frame", np.eye(4)), (label, np.eye(4))])
    with pytest.raises(ValueError, match="labels must be nonempty and unique"):
        load_reference_poses(path)
