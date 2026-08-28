"""Run a reproducible Hydra experiment, e.g. experiment=visibility_aware."""

import hydra
from omegaconf import DictConfig

from underwater_vision.pipeline import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig):
    run(config)


if __name__ == "__main__":
    main()
