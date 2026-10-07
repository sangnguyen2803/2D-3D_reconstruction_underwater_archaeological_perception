import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script,stage",
    [
        ("run_experiment.py", "all"),
        ("run_sfm.py", "all"),
        ("extract_features.py", "features"),
        ("match_features.py", "matches"),
    ],
)
def test_hydra_entry_point_finds_configuration_from_other_directory(tmp_path, script, stage):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert f"stage: {stage}" in result.stdout
