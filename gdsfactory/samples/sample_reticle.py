from __future__ import annotations

import gdsfactory as gf


def sample_reticle(grid: bool = True, **kwargs) -> gf.Component:
    """Returns MZI with TE grating couplers.

    Args:
        grid: if True returns components on a regular grid. False packs them as close as possible.
        kwargs: passed to pack or grid.
    """
    test_info_mzi_heaters = dict(
        doe="mzis_heaters",
        analysis="mzi_heater",
        measurement="optical_loopback4_heater_sweep",
    )
    test_info_ring_heaters = dict(
        doe="ring_heaters",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )

    mzis = [
        gf.components.mzi2x2_2x2_phase_shifter(
            length_x=length, name=f"mzi_heater_{length}"
        )
        for length in [100, 200, 300]
    ]
    rings = [
        gf.components.ring_single_heater(
            length_x=length_x, name=f"ring_single_heater_{length_x}"
        )
        for length_x in [10, 20, 30]
    ]

    spirals_sc = [
        gf.components.spiral_inner_io_fiber_array(
            name=f"spiral_sc_{int(length/1e3)}mm",
            length=length,
            info=dict(
                doe="spirals_sc",
                measurement="optical_loopback4",
                analysis="optical_loopback4_spirals",
            ),
        )
        for length in [20e3, 40e3, 60e3]
    ]

    mzis_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            mzi,
            electrical_port_names=["top_l_e2", "top_r_e2"],
            info=test_info_mzi_heaters,
            name=f"{mzi.name}_te",
        )
        for mzi in mzis
    ]
    rings_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            ring,
            electrical_port_names=["l_e2", "r_e2"],
            info=test_info_ring_heaters,
            name=f"{ring.name}_te",
        )
        for ring in rings
    ]

    components = mzis_te + rings_te + spirals_sc

    if grid:
        return gf.grid(components, name_ports_with_component_name=True, **kwargs)
    c = gf.pack(components, **kwargs)
    if len(c) > 1:
        raise ValueError(f"failed to pack into single group. Made {len(c)} groups.")
    return c[0]


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    c.show(show_ports=True)
    c.pprint_ports()
