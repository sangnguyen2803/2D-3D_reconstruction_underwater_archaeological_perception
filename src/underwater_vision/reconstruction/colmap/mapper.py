from pathlib import Path

import pycolmap


def reconstruct(database, image_dir, output, pair_file, settings, seed=42):
    pycolmap.set_random_seed(seed)
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.max_error = settings["verification_error"]
    verification.ransac.random_seed = seed
    verification.compute_relative_pose = True
    pycolmap.verify_matches(str(database), str(pair_file), verification)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = seed
    options.num_threads = settings.get("threads", 4)
    options.mapper.init_min_num_inliers = settings["init_min_inliers"]
    options.mapper.init_min_tri_angle = settings["init_min_angle"]
    options.ba_refine_focal_length = settings["refine_intrinsics"]
    options.ba_refine_extra_params = settings["refine_intrinsics"]
    options.min_model_size = min(settings["min_model_size"], settings["image_count"])
    models = pycolmap.incremental_mapping(str(database), str(image_dir), str(output), options)
    if not models:
        raise RuntimeError(
            "COLMAP registered no model. Inspect pairs/inliers and initial baseline."
        )
    best = max(models, key=lambda k: models[k].num_reg_images())
    return models[best], output / str(best)
