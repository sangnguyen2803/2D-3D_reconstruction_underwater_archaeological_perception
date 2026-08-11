"""Dense COLMAP reconstruction through a CUDA-capable external executable."""

import shutil
import subprocess
from pathlib import Path


def run_mvs(sparse, image_dir, output, executable="colmap", max_size=1280, mesh=True):
    binary = shutil.which(executable)
    if binary is None and Path(executable).is_file():
        binary = str(Path(executable).resolve())
    if binary is None:
        raise FileNotFoundError(
            "COLMAP CUDA executable missing. Set reconstruction.colmap_executable."
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            "image_undistorter",
            "--image_path",
            str(Path(image_dir).resolve()),
            "--input_path",
            str(Path(sparse).resolve()),
            "--output_path",
            str(output),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(max_size),
        ],
        [
            "patch_match_stereo",
            "--workspace_path",
            str(output),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.max_image_size",
            str(max_size),
        ],
        [
            "stereo_fusion",
            "--workspace_path",
            str(output),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(output / "fused.ply"),
        ],
    ]
    if mesh:
        commands.append(
            [
                "poisson_mesher",
                "--input_path",
                str(output / "fused.ply"),
                "--output_path",
                str(output / "mesh.ply"),
                "--PoissonMeshing.depth",
                "9",
                "--PoissonMeshing.trim",
                "0",
                "--PoissonMeshing.num_threads",
                "4",
            ]
        )
    for command in commands:
        with (output / (command[0] + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                [binary, *command],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(Path(binary).resolve().parent),
            )
    from plyfile import PlyData

    if len(PlyData.read(str(output / "fused.ply"))["vertex"]) == 0:
        raise RuntimeError("Dense fusion produced zero points")
    if mesh and len(PlyData.read(str(output / "mesh.ply"))["vertex"]) == 0:
        raise RuntimeError("Poisson meshing produced an empty mesh")
    return output / "fused.ply"
