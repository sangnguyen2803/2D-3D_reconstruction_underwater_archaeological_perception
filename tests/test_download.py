import zipfile
from unittest.mock import patch

import pytest

from scripts.download_data import download, extract_archive


def test_extract_explicit_directory_entries(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/", b"")
        handle.writestr("images/frame.jpg", b"image bytes")
    root = tmp_path / "extracted"
    extract_archive(archive, root)
    assert (root / "images").is_dir()
    assert (root / "images/frame.jpg").read_bytes() == b"image bytes"
    extract_archive(archive, root)


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", b"unsafe")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_archive(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()


def test_completed_partial_download_is_finalized_without_network(tmp_path):
    target = tmp_path / "asset.bin"
    target.with_suffix(".bin.part").write_bytes(b"complete")
    with patch("scripts.download_data.requests.get") as request:
        download("https://example.invalid/asset.bin", target, 8)
        request.assert_not_called()
    assert target.read_bytes() == b"complete"
    assert not target.with_suffix(".bin.part").exists()
