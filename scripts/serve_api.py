"""Serve semantic inference independently of COLMAP and the research pipeline."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--onnx")
    p.add_argument("--metadata", help="deployment_v2.json required for ONNX")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--task", choices=["semantic", "matchability"], default="semantic")
    p.add_argument("--calibration", help="Optional validation-fitted matchability Platt sidecar")
    a = p.parse_args()
    if a.calibration and a.task != "matchability":
        p.error("--calibration requires --task matchability")
    import uvicorn

    from underwater_vision.semantic.infer import OnnxSemanticPredictor, SemanticPredictor
    from underwater_vision.serving.api import create_app

    if a.onnx and not a.metadata:
        p.error("--onnx requires --metadata")
    if a.task == "matchability":
        from underwater_vision.matchability.infer import (
            MatchabilityPredictor,
            OnnxMatchabilityPredictor,
        )
        from underwater_vision.serving.matchability_api import create_matchability_app

        predictor = (
            OnnxMatchabilityPredictor(a.onnx, a.metadata, calibration=a.calibration)
            if a.onnx
            else MatchabilityPredictor(a.checkpoint, a.device, calibration=a.calibration)
        )
        uvicorn.run(create_matchability_app(predictor), host=a.host, port=a.port)
        return
    predictor = (
        OnnxSemanticPredictor(a.onnx, a.metadata)
        if a.onnx
        else SemanticPredictor(a.checkpoint, a.device)
    )
    uvicorn.run(create_app(predictor), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
