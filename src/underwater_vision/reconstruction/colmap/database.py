"""Use COLMAP's own schema/API; never fake uint8 neural descriptors."""

from pathlib import Path

import pycolmap


def create_database(path, frames, features, camera):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing database: {path}")
    with pycolmap.Database.open(str(path)) as database:
        camera_id = database.write_camera(camera)
        ids = {}
        for frame in frames:
            image = pycolmap.Image(name=frame.name, camera_id=camera_id)
            image_id = database.write_image(image)
            ids[frame.name] = image_id
            # COLMAP uses pixel centers at (0.5, 0.5); OpenCV uses (0, 0).
            database.write_keypoints(image_id, features[frame.name]["xy"] + 0.5)
    return ids


def insert_matches(path, ids, results):
    with pycolmap.Database.open(str(path)) as database:
        for result in results:
            pair = result["pairs"][result["inliers"]]
            database.write_matches(ids[result["a"]], ids[result["b"]], pair)


def make_camera(width, height, calibration):
    """PDF calibration is interpreted as Metashape Brown center offsets.

    FULL_OPENCV numerator k1,k2,k3 and zero denominator model is equivalent to
    this Brown model. The PDF does not explicitly specify its convention, so
    this interpretation is documented and should be independently verified.
    """
    sx, sy = width / calibration["width"], height / calibration["height"]
    f = calibration["f"]
    cx = calibration["width"] / 2 + calibration["cx_offset"]
    cy = calibration["height"] / 2 + calibration["cy_offset"]
    return pycolmap.Camera(
        model="FULL_OPENCV",
        width=width,
        height=height,
        params=[
            f * sx,
            f * sy,
            cx * sx,
            cy * sy,
            calibration["k1"],
            calibration["k2"],
            calibration["p1"],
            calibration["p2"],
            calibration["k3"],
            0,
            0,
            0,
        ],
    )
