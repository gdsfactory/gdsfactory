"""Write Sparameters with for different components."""
from typing import Optional

from tqdm.auto import tqdm

import gdsfactory as gf
from gdsfactory.simulation.lumerical.write_sparameters_lumerical import (
    write_sparameters_lumerical,
)
from gdsfactory.types import ComponentSpec, List


def write_sparameters_lumerical_components(
    components: List[ComponentSpec],
    run: bool = True,
    session: Optional[object] = None,
    **kwargs,
) -> None:
    """Writes Sparameters for a list of components using Lumerical FDTD.

    Args:
        factory: list of component or component functions to simulate.
        run: if False, prompts you to review each simulation.
        session: Optional Lumerical FDTD session. Creates lumapi.FDTD() if None.

    Keyword Args:
        simulation settings

    """
    import lumapi

    session = session or lumapi.FDTD()
    need_review = []

    for component in tqdm(components):
        component = gf.get_component(component)
        write_sparameters_lumerical(component, run=run, session=session, **kwargs)
        if not run:
            response = input(
                f"does the simulation for {component.name} look good? (y/n)"
            )
            if response.upper()[0] == "N":
                need_review.append(component.name)


if __name__ == "__main__":
    from gdsfactory.components import _factory_passives

    write_sparameters_lumerical_components(factory=_factory_passives.values())
