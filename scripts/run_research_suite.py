"""Reproduce v1 components in a new versioned root; previous artifacts are immutable."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/reproduction_v1"))
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def invoke(script, *arguments):
        subprocess.run([sys.executable, "scripts/" + script, *map(str, arguments)], check=True)

    audit, retrieval, matches, confidence = [
        root / n for n in ["audit_v1", "retrieval_v1", "matches_v1", "confidence_v1"]
    ]
    invoke("audit_reference.py", "--output", audit)
    invoke("retrieve_pairs.py", "--output", retrieval, "--audit", audit)
    invoke("train_match_filter.py", "--output", matches, "--audit", audit)
    invoke("evaluate_match_filter.py", "--run", matches, "--output", root / "match_evaluation_v1")
    shared = [
        "data.image_list=configs/evaluation_images.txt",
        f"output_root={json.dumps(root.as_posix())}",
    ]
    invoke("run_experiment.py", "experiment=classical", "run_name=baseline_v1", *shared)
    invoke(
        "train_point_confidence.py",
        "--matches",
        matches,
        "--output",
        confidence,
        "--baseline",
        root / "baseline_v1",
    )
    selected = json.loads((matches / "training_v1.json").read_text())["selected_architecture"]
    invoke(
        "run_experiment.py",
        "experiment=learned_matches",
        "run_name=combined_v1",
        *shared,
        f"matching.filter_checkpoint={json.dumps((matches/(selected+'_v1.pt')).as_posix())}",
        f"matching.retrieval_file={json.dumps((retrieval/'retrieval_v1.npz').as_posix())}",
        "matching.pairing=hybrid",
        f"reconstruction.learned_confidence_checkpoint={json.dumps((confidence/'error_model_v1.pt').as_posix())}",
    )


if __name__ == "__main__":
    main()
