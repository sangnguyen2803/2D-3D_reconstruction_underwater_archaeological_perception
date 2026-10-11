"""Pinned text-grounded SAM teacher; low-confidence/conflicting pixels ignored."""

import inspect

import numpy as np
import torch
from PIL import Image

from underwater_vision.semantic.schema import IGNORE, PROMPT_CLASSES, PROMPTS

GROUNDING_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"


class GroundedSAMTeacher:
    def __init__(
        self,
        device="auto",
        box_threshold=0.25,
        text_threshold=0.2,
        mask_threshold=0.7,
        max_boxes=24,
    ):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.detector_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", revision=GROUNDING_REVISION
        )
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny", revision=GROUNDING_REVISION
            )
            .eval()
            .to(self.device)
        )
        self.sam_processor = SamProcessor.from_pretrained(
            "facebook/sam-vit-base", revision=SAM_REVISION
        )
        self.sam = (
            SamModel.from_pretrained("facebook/sam-vit-base", revision=SAM_REVISION)
            .eval()
            .to(self.device)
        )
        self.box_threshold, self.text_threshold, self.mask_threshold, self.max_boxes = (
            box_threshold,
            text_threshold,
            mask_threshold,
            max_boxes,
        )
        self.provenance = {
            "grounding_model": "IDEA-Research/grounding-dino-tiny",
            "grounding_revision": GROUNDING_REVISION,
            "sam_model": "facebook/sam-vit-base",
            "sam_revision": SAM_REVISION,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "mask_iou_threshold": mask_threshold,
            "prompts": list(PROMPTS),
            "max_boxes": max_boxes,
            "label_source": "pseudo_not_hand",
            "probabilities_calibrated": False,
        }

    @torch.inference_mode()
    def predict(self, rgb):
        image = Image.fromarray(rgb)
        inputs = self.detector_processor(
            images=image, text=". ".join(PROMPTS) + ".", return_tensors="pt"
        ).to(self.device)
        output = self.detector(**inputs)
        post = self.detector_processor.post_process_grounded_object_detection
        threshold_name = (
            "box_threshold"
            if "box_threshold" in inspect.signature(post).parameters
            else "threshold"
        )
        result = post(
            output,
            inputs.input_ids,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.shape[:2]],
            **{threshold_name: self.box_threshold},
        )[0]
        labels = result.get("text_labels", result["labels"])
        proposals = []
        for box, score, label in zip(result["boxes"], result["scores"], labels, strict=True):
            classes = {PROMPT_CLASSES[i] for i, p in enumerate(PROMPTS) if p in str(label).lower()}
            if len(classes) == 1:
                proposals.append(
                    {
                        "box": box.cpu().tolist(),
                        "score": float(score),
                        "class": classes.pop(),
                        "text": str(label),
                    }
                )
        proposals = sorted(proposals, key=lambda p: -p["score"])[: self.max_boxes]
        mask = np.full(rgb.shape[:2], IGNORE, np.uint8)
        scores = np.zeros((6, *rgb.shape[:2]), np.float32)
        if proposals:
            sam_input = self.sam_processor(
                image, input_boxes=[[p["box"] for p in proposals]], return_tensors="pt"
            ).to(self.device)
            sam_output = self.sam(**sam_input, multimask_output=False)
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_output.pred_masks.cpu(),
                sam_input["original_sizes"].cpu(),
                sam_input["reshaped_input_sizes"].cpu(),
            )[0][:, 0].numpy()
            ious = sam_output.iou_scores[0, :, 0].cpu().numpy()
            for p, region, iou in zip(proposals, masks, ious, strict=True):
                p["sam_iou_score"] = float(iou)
                p["accepted"] = bool(iou >= self.mask_threshold)
                if p["accepted"]:
                    scores[p["class"]] = np.maximum(
                        scores[p["class"]], region * p["score"] * np.clip(iou, 0, 1)
                    )
            ordered = np.sort(scores, axis=0)
            confident = (ordered[-1] > 0.15) & (ordered[-1] - ordered[-2] > 0.03)
            mask[confident] = scores.argmax(0)[confident]
        return {"mask": mask, "confidence": scores.max(0), "proposals": proposals}
