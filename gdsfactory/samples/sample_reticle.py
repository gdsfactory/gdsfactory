from __future__ import annotations

import pandas as pd

import gdsfactory as gf


def add_info_spirals(c):
    c.info["doe"] = "spirals_sc"
    c.info["measurement"] = "optical_loopback4"
    c.info["analysis"] = "optical_loopback4_spirals"


def add_info_mzi(c):
    c.info["doe"] = "mzi"
    c.info["measurement"] = "optical_loopback4"
    c.info["analysis"] = "optical_loopback4_mzi"


def sample_reticle(grid: bool = True, **kwargs) -> gf.Component:
    """Returns MZI with TE grating couplers.

    Args:
        grid: if True returns a grid of components.
        kwargs: passed to pack or grid.

    """
    dict(
        doe="mzis_heaters",
        analysis="mzi_heater",
        measurement="optical_loopback4_heater_sweep",
    )
    dict(
        doe="ring_heaters",
        analysis="ring_heater",
        measurement="optical_loopback2_heater_sweep",
    )

    mzis = [
        gf.components.mzi2x2_2x2_phase_shifter(length_x=lengths)
        for lengths in [100, 200, 300]
    ]
    rings = [
        gf.components.ring_single_heater(length_x=length_x) for length_x in [10, 20, 30]
    ]

    # spirals_sc = [
    #     add_info_spirals(
    #         gf.components.spiral_inner_io_fiber_array(
    #             length=length,
    #         )
    #     )
    #     for length in [20e3, 40e3, 60e3]
    # ]
    spirals_sc = []

    mzis_te = [
        add_info_mzi(
            gf.components.add_fiber_array_optical_south_electrical_north(
                mzi,
                electrical_port_names=("top_l_e2", "top_r_e2"),
            )
        )
        for mzi in mzis
    ]

    rings_te = [
        gf.components.add_fiber_array_optical_south_electrical_north(
            ring,
            electrical_port_names=["l_e2", "r_e2"],
        )
        for ring in rings
    ]

    components = mzis_te + rings_te + spirals_sc

    if grid:
        return gf.grid(components, **kwargs)
    c = gf.pack(components, **kwargs)
    if len(c) > 1:
        raise ValueError(f"failed to pack into single group. Made {len(c)} groups.")
    return c[0]


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    gdspath = c.write_gds("mask.gds", with_metadata=True)
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=["{"], layer_label="TEXT"
    )

    df = pd.read_csv(csvpath)
    print(df)
    c.show()
