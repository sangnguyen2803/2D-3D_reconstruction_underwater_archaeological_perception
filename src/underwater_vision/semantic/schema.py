"""Canonical six-class schema; ambiguous pixels never become background."""

import numpy as np

NAMES = ("sand_sediment", "rock", "statue", "tile_artifacts", "sessile_life", "dynamic")
IGNORE = 255
DYNAMIC = 5
COLORS = np.array(
    [
        [225, 196, 125],
        [132, 137, 144],
        [210, 103, 180],
        [230, 120, 55],
        [77, 170, 92],
        [239, 61, 67],
    ],
    dtype=np.uint8,
)
PROMPTS = ("sand", "rock", "statue", "tile", "algae", "fish", "diver", "equipment")
PROMPT_CLASSES = (0, 1, 2, 3, 4, 5, 5, 5)


def validate_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [*range(len(NAMES)), IGNORE]).all():
        raise ValueError("Masks must be HxW class ids 0..5 or ignore=255")
    return mask.astype(np.uint8)


def merge_mask(mask, mapping=None):
    mask = validate_mask(mask)
    result = mask.copy()
    for source, target in (mapping or {}).items():
        if int(source) == IGNORE or int(target) not in [*range(len(NAMES)), IGNORE]:
            raise ValueError("Ignore cannot be remapped; target must be a valid class")
        result[mask == int(source)] = int(target)
    return result


def colorize(mask):
    mask = validate_mask(mask)
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    valid = mask != IGNORE
    rgb[valid] = COLORS[mask[valid]]
    return rgb
