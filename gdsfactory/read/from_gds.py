"""Read component GDS, YAML metadata and ports."""
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.import_gds import import_gds


def from_gds(gdspath: Union[str, Path], **kwargs) -> Component:
    """Returns Component with ports and metadata (YAML) info (if any).

    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        snap_to_grid_nm: snap to different nm grid (does not snap if False)

    """
    gdspath = Path(gdspath)
    metadata_filepath = gdspath.with_suffix(".yml")
    if not gdspath.exists():
        raise FileNotFoundError(f"No such file '{gdspath}'")
    component = import_gds(gdspath)

    if metadata_filepath.exists():
        metadata = OmegaConf.load(metadata_filepath)

        for port_name, port in metadata.ports.items():
            if port_name not in component.ports:
                component.add_port(
                    name=port_name,
                    midpoint=port.midpoint,
                    width=port.width,
                    orientation=port.orientation,
                    layer=port.layer,
                    port_type=port.port_type,
                )

        component.info = metadata.info
    return component


if __name__ == "__main__":
    gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    c = gf.read.from_gds(gdspath)
    c.show()
    print(c.to_yaml())

    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    # c = gf.c.straight()
    # c.write_gds_with_metadata(gdspath)

    # gdspath = gf.CONFIG["gdsdir"] / "straight.gds"
    # metadata_filepath = gdspath.with_suffix(".yml")
    # m = OmegaConf.load(metadata_filepath)
    # c2 = import_gds(gdspath)
    # c2.show()
