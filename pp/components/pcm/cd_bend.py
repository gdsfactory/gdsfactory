import numpy as np

import pp
from pp.components.bend_circular import (
    bend_circular,
    bend_circular_ridge,
    bend_circular_slot,
)
from pp.components.pcm.cd import CENTER_SHAPES_MAP, square_middle
from pp.components.waveguide import waveguide, waveguide_slab, waveguide_slot


@pp.cell
def cd_bend(
    L=2.0,
    radius=2.0,
    width=0.4,
    center_shapes="SU",
    bend90_factory=pp.c.bend_circular,
    waveguide_factory=pp.c.waveguide,
):
    """ bends and straights connected together
    for CDSEM measurement
    """

    component = pp.Component()
    _straight = waveguide_factory(length=L, width=width)
    _bend = bend90_factory(radius=radius, width=width)

    straight1 = _straight.ref(rotation=90, port_id="W0")
    component.add(straight1)

    bend1 = component.add_ref(_bend)
    bend1.connect(port="N0", destination=straight1.ports["E0"])

    straight2 = component.add_ref(_straight)
    straight2.connect(port="W0", destination=bend1.ports["W0"])

    bend2 = component.add_ref(_bend)
    bend2.connect(port="N0", destination=straight2.ports["E0"])

    # Add center shapes.
    # Center the first shape in the list
    # Then stack the others underneath

    # If we do ridge waveguides, add a slab
    markers_with_slabs = bend90_factory == bend_circular_ridge

    center = np.array([radius + L / 2, L / 2])
    center_shape_side = 0.4 if markers_with_slabs else 0.5
    center_shape_spacing = 0.2
    sep = center_shape_side + center_shape_spacing
    for cs_name in center_shapes:
        _shape_func = CENTER_SHAPES_MAP[cs_name]
        _shape = _shape_func(side=center_shape_side)
        _shape_ref = _shape.ref(position=center)
        component.add(_shape_ref)
        component.absorb(_shape_ref)

        # If with slabs, add the square slab on top of each marker
        if markers_with_slabs:
            _shape = square_middle(side=0.5)
            _shape_ref = _shape.ref(position=center)
            component.add(_shape_ref)
            component.absorb(_shape_ref)

        center += (0, -sep)

    return component


@pp.cell
def cd_bend_strip(**kwargs):
    return cd_bend(**kwargs, bend90_factory=bend_circular, waveguide_factory=waveguide)


@pp.cell
def cd_bend_rib(cladding=0.3, **kwargs):
    def bend(*a, **k):
        return bend_circular_ridge(*a, cladding=cladding, **k)

    def wg(*a, **k):
        return waveguide_slab(*a, cladding=cladding, **k)

    return cd_bend(**kwargs, bend90_factory=bend, waveguide_factory=wg)


@pp.cell
def cd_bend_slot(gap=0.2, **kwargs):
    def bend(*a, **k):
        return bend_circular_slot(*a, gap=gap, **k)

    def wg(*a, **k):
        return waveguide_slot(*a, gap=gap, **k)

    return cd_bend(**kwargs, bend90_factory=bend, waveguide_factory=wg)


if __name__ == "__main__":
    c = cd_bend_strip(width=0.46)
    pp.show(c)
