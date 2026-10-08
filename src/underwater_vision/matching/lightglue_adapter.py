"""Optional official pretrained SIFT LightGlue on exactly the same local features."""

import numpy as np
import torch


class LightGlueAdapter:
    def __init__(self, device="cuda"):
        from lightglue import LightGlue

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LightGlue(features="sift", flash=False).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, a, b, width, height):
        def convert(feature):
            if "scales" not in feature or "oris" not in feature:
                raise ValueError("LightGlue SIFT requires scales and orientations")
            return {
                "keypoints": torch.from_numpy(feature["xy"])[None].to(self.device),
                "descriptors": torch.from_numpy(feature["local"])[None].to(self.device),
                "scales": torch.from_numpy(feature["scales"])[None].to(self.device),
                "oris": torch.from_numpy(feature["oris"])[None].to(self.device),
                "image_size": torch.tensor([[width, height]], device=self.device),
            }

        result = self.model({"image0": convert(a), "image1": convert(b)})
        return (
            result["matches"][0].cpu().numpy().astype(np.uint32),
            result["scores"][0].cpu().numpy(),
        )
