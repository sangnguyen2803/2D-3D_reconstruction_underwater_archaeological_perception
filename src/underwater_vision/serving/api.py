"""Small image-to-mask API. Construct with any predictor exposing predict(rgb)."""

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from underwater_vision.semantic.schema import NAMES


def png_base64(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def create_app(predictor):
    from fastapi import FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="Underwater semantic perception", version="2.0")
    upload_field = File(...)

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "checkpoint_sha256": predictor.checkpoint_hash,
            "classes": list(NAMES),
        }

    @app.post("/predict")
    async def predict(image: UploadFile = upload_field, include_dynamic: bool = True):
        try:
            rgb = np.asarray(Image.open(io.BytesIO(await image.read())).convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise HTTPException(400, "Input must be a readable image") from e
        p = predictor.predict(rgb)
        mask = p.argmax(0).astype(np.uint8)
        fractions = {name: float((mask == i).mean()) for i, name in enumerate(NAMES)}
        response = {
            "mask_png_base64": png_base64(mask),
            "width": mask.shape[1],
            "height": mask.shape[0],
            "class_pixel_fraction": fractions,
            "checkpoint_sha256": predictor.checkpoint_hash,
        }
        if include_dynamic:
            threshold = predictor.metadata.get("dynamic_threshold", 0.5)
            response["dynamic_mask_png_base64"] = png_base64(
                (p[5] >= threshold).astype(np.uint8) * 255
            )
        return response

    return app
