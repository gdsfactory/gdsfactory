from __future__ import annotations

__all__ = [
    "taper_cross_section",
    "taper_cross_section_linear",
    "taper_cross_section_parabolic",
    "taper_cross_section_sine",
]

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import (
    AsymmetricExtrusionSpec,
    ExtrusionSection,
    SectionReference,
    SymmetricExtrusionSpec,
    TransitionSection,
)
from gdsfactory.typings import CrossSectionSpec, LayerSpec, LayerSpecs

from .._schematic import transition_schematic


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_cross_section(
    cross_section1: CrossSectionSpec = "strip_rib_tip",
    cross_section2: CrossSectionSpec = "rib2",
    length: float = 10,
    npoints: int = 100,
    linear: bool = False,
    width_type: str = "sine",
    exclude_layers: LayerSpecs | None = None,
) -> Component:
    r"""Returns taper transition between cross_section1 and cross_section2.

    Args:
        cross_section1: start cross_section factory.
        cross_section2: end cross_section factory.
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        width_type: shape of the transition ONLY IF linear is False
        exclude_layers: layers to exclude from the transition.
            Sections on these layers will be omitted from the component.

    ```text
                           _____________________
                          /
                  _______/______________________
                        /
       cross_section1  |        cross_section2
                  ______\_______________________
                         \
                          \_____________________
    ```


    """
    x1 = gf.get_cross_section(cross_section1)
    x2 = gf.get_cross_section(cross_section2)

    excluded: set[int] = set()
    if exclude_layers:
        layers: list[LayerSpec] = (
            list(exclude_layers)
            if isinstance(exclude_layers, (list, tuple))
            and not (len(exclude_layers) == 2 and isinstance(exclude_layers[0], int))
            else [exclude_layers]  # type: ignore[list-item]
        )
        excluded = {int(gf.get_layer(layer)) for layer in layers}

    def _references(
        xs: gf.CrossSection,
    ) -> dict[tuple[tuple[int, int], int], SectionReference]:
        counts: dict[tuple[int, int], int] = {}
        result: dict[tuple[tuple[int, int], int], SectionReference] = {}
        for section in xs.get_sections():
            key = gf.get_layer_tuple(section.layer)
            index = counts.get(key, 0)
            counts[key] = index + 1
            result[(key, index)] = SectionReference(layer=key, index=index)
        return result

    references1 = _references(x1)
    references2 = _references(x2)
    mappings: list[TransitionSection] = []
    for key, start_ref in references1.items():
        end_ref = references2.get(key)
        layer = key[0][0]
        mappings.append(
            TransitionSection(
                start=start_ref,
                end=end_ref,
                extrusion=ExtrusionSection(
                    port_names=("o1", "o2")
                    if key == ((gf.get_layer_tuple(x1.layer)), 0)
                    else (None, None),
                    hidden=layer in excluded,
                ),
            )
        )
    for key, end_ref in references2.items():
        if key not in references1:
            mappings.append(TransitionSection(start=None, end=end_ref))

    if x1.is_symmetric() and x2.is_symmetric():
        extrusion_spec: SymmetricExtrusionSpec | AsymmetricExtrusionSpec = (
            SymmetricExtrusionSpec(sections=tuple(mappings))
        )
    else:
        extrusion_spec = AsymmetricExtrusionSpec(sections=tuple(mappings))

    if x1 == x2 and not exclude_layers:
        return gf.components.straight(length=length, cross_section=x1)

    if x1.is_symmetric() and x2.is_symmetric():
        transition = gf.path.transition(
            cross_section1=x1,
            cross_section2=x2,
            width_type="linear" if linear else width_type,  # type: ignore
            offset_type="linear" if linear else width_type,  # type: ignore
            extrusion_spec=extrusion_spec,  # type: ignore[arg-type]
        )
    else:
        transition = gf.path.transition_asymmetric(
            cross_section1=x1,
            cross_section2=x2,
            width_type1="linear" if linear else width_type,  # type: ignore
            width_type2="linear" if linear else width_type,  # type: ignore
            offset_type1="linear" if linear else width_type,  # type: ignore
            offset_type2="linear" if linear else width_type,  # type: ignore
            extrusion_spec=extrusion_spec,  # type: ignore[arg-type]
        )
    taper_path = gf.path.straight(length=length, npoints=npoints)

    c = gf.Component()
    ref = c << gf.path.extrude_transition(taper_path, transition=transition)
    c.add_ports(ref.ports)
    c.add_route_info(cross_section=x1, length=length, taper=True)
    c.flatten()
    return c


taper_cross_section_linear = partial(taper_cross_section, linear=True, npoints=2)
taper_cross_section_sine = partial(taper_cross_section, linear=False, npoints=101)
taper_cross_section_parabolic = partial(
    taper_cross_section, linear=False, width_type="parabolic", npoints=101
)
