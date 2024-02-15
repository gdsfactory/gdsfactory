from __future__ import annotations

import gdsfactory as gf


@gf.cell
def mzi_phase_shifter(length_x: float = 100.0, **kwargs) -> gf.Component:
    """Returns MZI with TE grating couplers.

    Args:
        length_x: length of the phase shifter.
        kwargs: extra arguments passed to the MZI component.
    """
    test_info_mzi_heaters = dict(
        doe="mzis_heaters",
        analysis="mzi_heater",
        measurement="optical_loopback4_heater_sweep",
    )
    c = gf.components.mzi2x2_2x2_phase_shifter(length_x=length_x, **kwargs)
    c = gf.components.add_fiber_array_optical_south_electrical_north(
        c,
        electrical_port_names=["top_l_e2", "top_r_e2"],
    )
    c.info.update(test_info_mzi_heaters)
    return c


@gf.cell
def ring_single_heater(length_x: float = 10.0, **kwargs) -> gf.Component:
    """Returns ring with TE grating couplers.

    Args:
        length_x: length of the ring.
        kwargs: extra arguments passed to the ring component.
    """
    test_info_ring_heaters = dict(
        doe="ring_heaters",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )
    c = gf.components.ring_single_heater(length_x=length_x, **kwargs)
    c = gf.components.add_fiber_array_optical_south_electrical_north(
        c,
        electrical_port_names=["l_e2", "r_e2"],
    )
    c.info.update(test_info_ring_heaters)
    return c


@gf.cell
def spiral_sc(length: float = 20e3, **kwargs) -> gf.Component:
    """Returns spiral with TE grating couplers.

    Args:
        length: length of the spiral.
        kwargs: extra arguments passed to the spiral component.
    """
    info = dict(
        doe="spirals_sc",
        measurement="optical_loopback4",
        analysis="optical_loopback4_spirals",
    )
    c = gf.components.spiral_inner_io_fiber_array(length=length, **kwargs)
    c.info.update(info)
    return c


def sample_reticle(grid: bool = True, **kwargs) -> gf.Component:
    """Returns MZI with TE grating couplers.

    Args:
        grid: if True returns components on a regular grid. False packs them as close as possible.
        kwargs: passed to pack or grid.
    """

    mzis_te = [mzi_phase_shifter(length_x=length) for length in [100, 200, 300]]
    rings_te = [ring_single_heater(length_x=length_x) for length_x in [10, 20, 30]]
    spirals_te = [spiral_sc(length=length) for length in [20e3, 40e3, 60e3]]

    components = mzis_te + rings_te + spirals_te

    if grid:
        return gf.grid(components, name_ports_with_component_name=True, **kwargs)
    c = gf.pack(components, **kwargs)
    if len(c) > 1:
        raise ValueError(f"failed to pack into single group. Made {len(c)} groups.")
    return c[0]


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    # c = mzi_phase_shifter()
    # c = spiral_sc()
    c.show(show_ports=True)
    # c.pprint_ports()
