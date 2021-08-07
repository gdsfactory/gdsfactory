"""You can define a path with a list of points

To create a component you need to extrude the path with a cross-section.

Based on phidl.device_layout.CrossSection
"""
import dataclasses
from typing import Dict, Iterable, Optional, Tuple, Union

import pydantic
from phidl.device_layout import CrossSection

from gdsfactory.tech import TECH, Section

LAYER = TECH.layer
Layer = Tuple[int, int]
StrOrDict = Union[str, Dict]


def get_cross_section(waveguide: StrOrDict, **kwargs) -> CrossSection:
    """Returns CrossSection from a string from TECH.waveguide or from a dict."""
    if isinstance(waveguide, str):
        settings = getattr(TECH.waveguide, waveguide)
        settings = dataclasses.asdict(settings)
        if not settings:
            raise ValueError(f"no settings found for {waveguide}")
    elif isinstance(waveguide, dict) and "component" in waveguide:
        waveguide_name = waveguide.pop("component")
        settings = getattr(TECH.waveguide, waveguide_name)
        settings = dataclasses.asdict(settings)
        settings.update(**waveguide)
    elif isinstance(waveguide, dict):
        settings = waveguide.copy()
    else:
        raise ValueError(
            f"{waveguide} needs to be a string or dict, got {type(waveguide)}"
        )

    settings.update(**kwargs)
    return cross_section(**settings)


def get_waveguide_settings(waveguide: StrOrDict, **kwargs):
    x = get_cross_section(waveguide, **kwargs)
    return x.info


def cross_section(
    width: float = 0.5,
    layer: Optional[Tuple[int, int]] = (1, 0),
    width_wide: Optional[float] = None,
    auto_widen: bool = True,
    auto_widen_minimum_length: float = 200.0,
    taper_length: float = 10.0,
    radius: float = 10.0,
    cladding_offset: float = 3.0,
    layer_cladding: Optional[Layer] = None,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    sections: Optional[Tuple[Section, ...]] = None,
    port_names: Tuple[str, str] = ("W0", "E0"),
    min_length: float = 10e-3,
) -> CrossSection:
    """Returns CrossSection from TECH.waveguide settings.

    Args:
        width: main of the waveguide
        layer: main layer
        width_wide: taper to widen waveguides for lower loss
        auto_widen: taper to widen waveguides for lower loss
        auto_widen_minimum_length: minimum straight length for auto_widen
        taper_length: taper_length for auto_widen
        radius: bend radius
        cladding_offset: offset for layers_cladding
        layer_cladding:
        layers_cladding:
        sections: Sections(width, offset, layer, ports)
        port_names: for input and output ("W0", "E0"),
        min_length: 10e-3 for routing
    """

    x = CrossSection()

    if layer:
        x.add(width=width, offset=0, layer=layer, ports=port_names)

    sections = sections or []
    for section in sections:
        if isinstance(section, dict):
            x.add(
                width=section["width"],
                offset=section["offset"],
                layer=section["layer"],
                ports=section["ports"],
            )
        else:
            x.add(
                width=section.width,
                offset=section.offset,
                layer=section.layer,
                ports=section.ports,
            )

    x.info = dict(
        width=width,
        layer=layer,
        width_wide=width_wide,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        taper_length=taper_length,
        radius=radius,
        cladding_offset=cladding_offset,
        layer_cladding=layer_cladding,
        layers_cladding=layers_cladding,
        sections=sections,
        min_length=min_length,
    )
    return x


def pin(
    width: float,
    layer: Tuple[int, int] = LAYER.WG,
    layer_slab: Tuple[int, int] = LAYER.SLAB90,
    width_i: float = 0.0,
    width_p: float = 1.0,
    width_n: float = 1.0,
    width_pp: float = 1.0,
    width_np: float = 1.0,
    width_ppp: float = 1.0,
    width_npp: float = 1.0,
    layer_p: Tuple[int, int] = LAYER.P,
    layer_n: Tuple[int, int] = LAYER.N,
    layer_pp: Tuple[int, int] = LAYER.Pp,
    layer_np: Tuple[int, int] = LAYER.Np,
    layer_ppp: Tuple[int, int] = LAYER.Ppp,
    layer_npp: Tuple[int, int] = LAYER.Npp,
    cladding_offset: float = 0,
    layers_cladding: Optional[Iterable[Tuple[int, int]]] = None,
) -> CrossSection:
    """PIN doped straight.

    .. code::

                                   layer
                           |<------width------>|
                            ____________________
                           |     |       |     |
        ___________________|     |       |     |__________________________|
                                 |       |                                |
            P++     P+     P     |   I   |     N        N+         N++    |
        __________________________________________________________________|
                                                                          |
                                 |width_i| width_n | width_np | width_npp |
                                    0    oi        on        onp         onpp

    """
    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=["in", "out"])

    oi = width_i / 2
    on = oi + width_n
    onp = oi + width_n + width_np
    onpp = oi + width_n + width_np + width_npp

    op = -oi - width_p
    opp = op - width_pp
    oppp = opp - width_ppp

    offset_n = (oi + on) / 2
    offset_np = (on + onp) / 2
    offset_npp = (onp + onpp) / 2

    offset_p = (-oi + op) / 2
    offset_pp = (op + opp) / 2
    offset_ppp = (opp + oppp) / 2

    width_slab = abs(onpp) + abs(oppp)
    x.add(width=width_slab, offset=0, layer=layer_slab)

    x.add(width=width_n, offset=offset_n, layer=layer_n)
    x.add(width=width_np, offset=offset_np, layer=layer_np)
    x.add(width=width_npp, offset=offset_npp, layer=layer_npp)

    x.add(width=width_p, offset=offset_p, layer=layer_p)
    x.add(width=width_pp, offset=offset_pp, layer=layer_pp)
    x.add(width=width_ppp, offset=offset_ppp, layer=layer_ppp)

    for layer_cladding in layers_cladding or []:
        x.add(width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
    )
    x.info = s
    return x


@pydantic.validate_arguments
def heater_with_undercut(
    waveguide_width: float = 0.5,
    heater_width: float = 1.0,
    trench_width: float = 8.0,
    trench_offset: float = 8.0,
    layer_waveguide: Layer = LAYER.WG,
    layer_heater: Layer = LAYER.HEATER,
    layer_trench: Layer = LAYER.DEEPTRENCH,
    **kwargs,
):
    """Returns heater with undercut."""
    return cross_section(
        width=waveguide_width,
        layer=layer_waveguide,
        sections=(
            Section(layer=layer_heater, width=heater_width, ports=("HW", "HE")),
            Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
            Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
        ),
        **kwargs,
    )


if __name__ == "__main__":
    import gdsfactory as gf

    s = get_waveguide_settings("strip")
    print(s)

    P = gf.path.straight()
    # P = gf.path.euler(radius=10, use_eff=True)
    # P = euler()
    # P = gf.Path()
    # P.append(gf.path.straight(length=5))
    # P.append(gf.path.arc(radius=10, angle=90))
    # P.append(gf.path.spiral())

    # Create a blank CrossSection
    # X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    # X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # Combine the Path and the CrossSection into a Component
    # X = pin(width=0.5, width_i=0.5)
    # x = strip(width=0.5)

    # X = cross_section(width=3, layer=(2, 0))
    # X = cross_section(**s)
    X = heater_with_undercut()
    c = gf.path.extrude(P, X)

    # c = gf.path.component(P, strip(width=2, layer=LAYER.WG, cladding_offset=3))

    # c = gf.add_pins(c)
    # c << gf.components.bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    c.show()
