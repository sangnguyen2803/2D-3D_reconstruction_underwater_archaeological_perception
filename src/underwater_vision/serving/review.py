"""Local human annotation editor. Suggestions and reviewed masks stay distinct."""

import base64
import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image

from underwater_vision.semantic.protocol import digest_file
from underwater_vision.semantic.schema import validate_mask
from underwater_vision.utils.io import write_json


def create_review_app(label_root, audit_root):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response

    root, audit = Path(label_root), Path(audit_root)
    manifest_path = root / "manifest.json"
    app = FastAPI(title="Local underwater annotation review")
    lock = RLock()

    def read():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def entry(name):
        matches = [e for e in read()["entries"] if e["image_id"] == name and e["role"] != "train"]
        if not matches:
            raise HTTPException(404, "Unknown hand-label task")
        return matches[0]

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("review.html"))

    @app.get("/tasks")
    def tasks():
        m = read()
        sample = json.loads((audit / "sample_manifest.json").read_text())
        return {
            "hand": [e for e in m["entries"] if e["role"] != "train"],
            "repeat": m["repeat_tasks"],
            "audit": [n for names in sample["groups"].values() for n in names],
            "frozen": bool(m.get("frozen")),
        }

    @app.get("/asset/{kind}/{name}")
    def asset(kind: str, name: str):
        if Path(name).name != name:
            raise HTTPException(400, "Invalid image id")
        if kind in {"audit_image", "audit_mask"}:
            names = {
                n
                for group in json.loads((audit / "sample_manifest.json").read_text())[
                    "groups"
                ].values()
                for n in group
            }
            if name not in names:
                raise HTTPException(404, "Unknown audit image")
            path = (
                audit / "images" / name
                if kind == "audit_image"
                else audit / "teacher" / (Path(name).stem + ".png")
            )
        else:
            e = entry(name)
            if kind == "image":
                path = Path(e["image"])
            elif kind == "mask":
                path = Path(e["mask"])
            elif kind == "suggestion":
                path = root / "suggestions" / (Path(name).stem + ".png")
                if not path.is_file():
                    path = audit / "teacher" / (Path(name).stem + ".png")
            elif kind == "repeat":
                path = root / "repeat_masks" / (Path(name).stem + ".png")
                if not path.exists():
                    shape = np.asarray(Image.open(e["mask"])).shape
                    stream = io.BytesIO()
                    Image.fromarray(np.full(shape, 255, np.uint8)).save(stream, format="PNG")
                    return Response(stream.getvalue(), media_type="image/png")
            else:
                raise HTTPException(404, "Unknown asset")
        if not path.is_file():
            raise HTTPException(404, "No suggestion for this image; annotate from ignore")
        return FileResponse(path)

    @app.post("/annotation/{name}")
    def save_annotation(name: str, payload: dict):
        with lock:
            m = read()
            if m.get("frozen"):
                raise HTTPException(409, "Frozen label manifest cannot be edited")
            e = next(
                (e for e in m["entries"] if e["image_id"] == name and e["role"] != "train"), None
            )
            if e is None or not str(payload.get("annotator", "")).strip():
                raise HTTPException(400, "Known task and named annotator required")
            try:
                decoded = np.asarray(
                    Image.open(io.BytesIO(base64.b64decode(payload["mask_png_base64"])))
                )
                if decoded.ndim == 3 and np.all(decoded[:, :, :3] == decoded[:, :, :1]):
                    decoded = decoded[:, :, 0]
                mask = validate_mask(decoded)
            except (ValueError, KeyError, OSError) as exc:
                raise HTTPException(400, "Expected PNG with class ids 0..5 and 255") from exc
            original = np.asarray(Image.open(e["mask"]))
            if mask.shape != original.shape or not (mask != 255).any():
                raise HTTPException(
                    400, "Mask must match task dimensions and contain labeled pixels"
                )
            now = datetime.now(UTC)
            repeat = bool(payload.get("repeat"))
            if repeat:
                if name not in m["repeat_tasks"] or not e.get("reviewed"):
                    raise HTTPException(400, "Repeat task requires completed pass 1")
                if (now - datetime.fromisoformat(e["date"])).total_seconds() < 48 * 3600:
                    raise HTTPException(
                        409, "Wait at least 48 hours after pass 1 for the repeat annotation"
                    )
                path = root / "repeat_masks" / (Path(name).stem + ".png")
                Image.fromarray(mask).save(path)
                m.setdefault("repeat_annotations", {})[name] = {
                    "mask": str(path.resolve()),
                    "annotator": payload["annotator"],
                    "date": now.isoformat(),
                    "pass": 2,
                    "mask_sha256": digest_file(path),
                }
            else:
                path = Path(e["mask"])
                Image.fromarray(mask).save(path)
                e.update(
                    reviewed=True,
                    annotator=payload["annotator"].strip(),
                    date=now.isoformat(),
                    pass_number=1,
                    mask_sha256=digest_file(path),
                )
                e["pass"] = 1
            write_json(manifest_path, m)
        return {"status": "saved_human_annotation", "image_id": name, "repeat": repeat}

    @app.post("/audit/{name}")
    def save_audit(name: str, payload: dict):
        with lock:
            path = audit / "human_review.csv"
            if (audit / "gate_decision_v2.json").exists():
                raise HTTPException(409, "Gate review already finalized")
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            row = next((r for r in rows if r["image_id"] == name), None)
            if (
                row is None
                or not payload.get("reviewer")
                or payload.get("classes_separable") not in {"yes", "no"}
            ):
                raise HTTPException(
                    400, "Named human reviewer, known image and separability decision required"
                )
            count = payload.get("dynamic_instances")
            if not isinstance(count, int) or count < 0:
                raise HTTPException(400, "Dynamic instances must be a nonnegative count")
            row.update(
                reviewed="yes",
                reviewer=payload["reviewer"],
                classes_separable=payload["classes_separable"],
                dynamic_instances=str(count),
                notes=str(payload.get("notes", "")),
            )
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return {"status": "saved_human_audit", "image_id": name}

    return app
