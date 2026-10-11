"""One-channel dense matchability service; entirely independent of COLMAP."""

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.serving.api import png_base64


def create_matchability_app(predictor):
    from fastapi import FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="Underwater dense matchability", version="3.0")
    upload_field = File(...)

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "task": "dense_matchability",
            "checkpoint_sha256": predictor.checkpoint_hash,
            "feature_spec": "dense_matchability_v3",
            "calibration_sha256": getattr(predictor, "calibration_hash", None),
        }

    @app.post("/predict")
    async def predict(image: UploadFile = upload_field, top_n: int = 0):
        if not 0 <= top_n <= 4096:
            raise HTTPException(400, "top_n must be between 0 and 4096")
        data = await image.read()
        if len(data) > 20_000_000:
            raise HTTPException(400, "Image exceeds 20 MB")
        try:
            opened = Image.open(io.BytesIO(data))
            if opened.width * opened.height > 25_000_000:
                raise ValueError("Image exceeds 25 megapixels")
            rgb = np.asarray(opened.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                400, "Input must be a readable image of at most 25 megapixels"
            ) from exc
        rgb, _ = resize_rgb(rgb, 960)
        values = predictor.predict(rgb)
        response = {
            "matchability_png_base64": png_base64(
                np.clip(np.rint(values * 65535), 0, 65535).astype(np.uint16)
            ),
            "map_encoding": "16-bit grayscale PNG; score = pixel / 65535",
            "width": values.shape[1],
            "height": values.shape[0],
            "statistics": {
                "mean": float(values.mean()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "fraction_above_half": float((values >= 0.5).mean()),
            },
            "checkpoint_sha256": predictor.checkpoint_hash,
            "temperature": predictor.temperature,
            "calibration_method": (
                getattr(predictor, "calibration", None) or {"method": "temperature"}
            )["method"],
            "calibration_sha256": getattr(predictor, "calibration_hash", None),
        }
        if top_n:
            from underwater_vision.matchability.infer import sample_probability

            points = cv2.SIFT_create(nfeatures=4096).detect(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None
            )
            xy = np.asarray([p.pt for p in points], np.float32).reshape(-1, 2)
            scores = sample_probability(values, xy)
            ids = np.argsort(-scores, kind="stable")[:top_n]
            response["keypoints"] = [
                {"source_index": int(i), "xy": xy[i].tolist(), "score": float(scores[i])}
                for i in ids
            ]
            response["requested_top_n"] = top_n
            response["detected_keypoints"] = len(xy)
        return response

    return app
