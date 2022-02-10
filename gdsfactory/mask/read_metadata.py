from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from gdsfactory.types import PathType


def read_metadata(gdspath: PathType) -> DictConfig:
    """Return DictConfig from YAML mask metadata.

    Args:
        gdspath: GDSpath
    """
    yaml_path = Path(gdspath).with_suffix(".yml")
    return OmegaConf.load(yaml_path)
