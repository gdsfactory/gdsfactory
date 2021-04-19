import pathlib
from typing import Union

from phidl import quickplot

from pp.component import Component
from pp.import_gds import import_gds


def plotgds(
    gds: Union[Component, str, pathlib.Path],
    show_ports=True,
    show_subports=False,
    label_ports=True,
    label_aliases=False,
    new_window=False,
) -> None:
    """Plot GDS in matplotlib.

    Args:
        gds: Component or gdspath
        show_ports: show port markers
        show_subports: for each reference
        label_ports: add port labels
        label_aliases: add alias labels
        new_window: add new window

    """
    if isinstance(gds, (str, pathlib.Path)):
        gds = import_gds(gds)

    quickplot(
        gds,
        show_ports=show_ports,
        show_subports=show_subports,
        label_ports=label_ports,
        label_aliases=label_aliases,
        new_window=new_window,
    )


if __name__ == "__main__":
    import pp

    c = pp.components.straight()
    plotgds(c)
