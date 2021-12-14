"""Write Sparameters with for different components."""
from typing import Optional

from gdsfactory.simulation.write_sparameters_lumerical import (
    write_sparameters_lumerical,
)
from gdsfactory.types import ComponentFactoryDict


def write_sparameters_components_lumerical(
    factory: ComponentFactoryDict,
    run: bool = False,
    session: Optional[object] = None,
    **kwargs,
) -> None:
    """writes component Sparameters using Lumerical FDTD.

    Args:
        factory: dict of component functions
        run: if False, does not run and prompts you to review each simulation
        session: lumapi.FDTD() Lumerical FDTD session

    Keyword Args:
        simulation settings
    """
    import lumapi

    session = session or lumapi.FDTD()
    need_review = []

    for component_name in factory.keys():
        component = factory[component_name]()
        write_sparameters_lumerical(component, run=run, session=session, **kwargs)
        if not run:
            response = input(
                f"does the simulation for {component_name} look good? (y/n)"
            )
            if response.upper()[0] == "N":
                need_review.append(component_name)


if __name__ == "__main__":
    from gdsfactory.components import _factory_passives

    write_sparameters_components_lumerical(factory=_factory_passives)
