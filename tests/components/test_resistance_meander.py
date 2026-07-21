from __future__ import annotations

import gdsfactory as gf


def test_resistance_meander_multiple_parameter_sets() -> None:
    component = gf.Component()

    first = component.add_ref(
        gf.components.resistance_meander(
            pad_size=(50, 50),
            num_squares=1_000,
            width=1,
            res_layer="MTOP",
            pad_layer="MTOP",
        )
    )
    second = component.add_ref(
        gf.components.resistance_meander(
            pad_size=(50, 50),
            num_squares=1_500,
            width=1,
            res_layer="MTOP",
            pad_layer="MTOP",
        )
    )

    assert first.cell.name != second.cell.name
    assert len(component.insts) == 2
