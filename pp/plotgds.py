import pathlib
from phidl import quickplot


from pp.import_gds import import_gds


def plotgds(
    gds,
    show_ports=True,
    show_subports=False,
    label_ports=True,
    label_aliases=False,
    new_window=False,
):
    if isinstance(gds, (str, pathlib.Path)):
        gds = import_gds(gds)

    quickplot(
        gds,
        show_ports=show_ports,
        label_ports=label_ports,
        label_aliases=label_aliases,
        new_window=new_window,
    )


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    plotgds(c)
