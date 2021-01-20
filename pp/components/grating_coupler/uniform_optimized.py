import pathlib

import numpy as np

import pp
from pp.component import Component

data_path = pathlib.Path(__file__).parent / "csv_data"


@pp.cell
def grating_coupler_uniform_optimized(
    widths=(0.5, 0.2, 0.3),
    width_grating=11,
    length_taper=150,
    width=0.5,
    partial_etch=False,
    layer=pp.LAYER.WG,
    layer_partial_etch=pp.LAYER.SLAB150,
    taper=None,
    polarization="te",
    wavelength=1500,
):
    """ Grating coupler uniform (not focusing)

    Args:
        widths: of each teeth
        width_grating: 11
        length_taper: 150
        width: 0.5
        partial_etch: False

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_uniform_optimized()
      pp.plotgds(c)

    """
    # returns a fiber grating
    c = Component()
    x = 0

    if partial_etch:
        partetch_overhang = 5

        # make the etched areas (opposite to teeth)
        for i, wt in enumerate(widths):
            if i % 2 == 1:
                _compass = pp.c.compass(
                    size=[wt, width_grating + partetch_overhang * 2],
                    layer=layer_partial_etch,
                )
                cgrating = c.add_ref(_compass)
                cgrating.x += x + wt / 2
            x += wt

        # draw the deep etched square around the grating
        xgrating = np.sum(widths)
        deepbox = c.add_ref(pp.c.compass(size=[xgrating, width_grating], layer=layer))
        deepbox.movex(xgrating / 2)
    else:
        for i, wt in enumerate(widths):
            if i % 2 == 0:
                cgrating = c.add_ref(
                    pp.c.compass(size=[wt, width_grating], layer=layer)
                )
                cgrating.x += x + wt / 2
            x += wt

    # make the taper
    if taper is None:
        taper = pp.c.taper(
            length=length_taper,
            width1=width,
            width2=width_grating,
            port=None,
            layer=layer,
        )
    taper_ref = c.add_ref(taper)
    taper_ref.xmax = 0
    port = taper_ref.ports.get("W0") or taper_ref.ports.get("1")
    c.polarization = polarization
    c.wavelength = wavelength
    c.add_port(port=taper_ref.ports[port.name], name="W0")
    pp.assert_grating_coupler_properties(c)
    return c


@pp.cell
def grating_coupler_uniform_1etch_h220_e70(**kwargs):
    csv_path = data_path / "grating_coupler_1etch_h220_e70.csv"
    import pandas as pd

    d = pd.read_csv(csv_path)
    return grating_coupler_uniform_optimized(
        widths=d["widths"], partial_etch=True, **kwargs
    )


@pp.cell
def grating_coupler_uniform_2etch_h220_e70(**kwargs):
    csv_path = data_path / "grating_coupler_2etch_h220_e70_e220.csv"
    import pandas as pd

    d = pd.read_csv(csv_path)
    return grating_coupler_uniform_optimized(
        widths=d["widths"], partial_etch=True, **kwargs
    )


@pp.cell
def grating_coupler_uniform_1etch_h220_e70_taper_w11_l200(**kwargs):
    from pp.components.taper_from_csv import taper_w11_l200

    taper = taper_w11_l200()
    return grating_coupler_uniform_1etch_h220_e70(taper=taper)


@pp.cell
def grating_coupler_uniform_1etch_h220_e70_taper_w10_l200(**kwargs):
    from pp.components.taper_from_csv import taper_w10_l200

    taper = taper_w10_l200()
    return grating_coupler_uniform_1etch_h220_e70(taper=taper, width_grating=10)


@pp.cell
def grating_coupler_uniform_1etch_h220_e70_taper_w10_l100(**kwargs):
    from pp.components.taper_from_csv import taper_w10_l100

    taper = taper_w10_l100()
    return grating_coupler_uniform_1etch_h220_e70(taper=taper, width_grating=10)


if __name__ == "__main__":
    # widths = [0.3, 0.5, 0.3]
    # c = grating_coupler_uniform_optimized(widths=widths, partial_etch=False)
    # c = grating_coupler_uniform_optimized(widths=widths, partial_etch=True)

    # c = grating_coupler_uniform_1etch_h220_e70()
    c = grating_coupler_uniform_2etch_h220_e70()
    # c = grating_coupler_uniform_1etch_h220_e70_taper_w11_l200()
    # c = grating_coupler_uniform_1etch_h220_e70_taper_w10_l200()
    # c = grating_coupler_uniform_1etch_h220_e70_taper_w10_l100()
    print(c.ports)
    pp.show(c)
