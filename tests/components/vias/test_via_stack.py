import warnings

import pytest

from gdsfactory.components.vias.via_stack import via_stack


def test_via_stack_one_via_per_gap_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = via_stack(
            layers=("M1", "M2", "MTOP"),
            layer_offsets=(0, 1, 2),
            vias=("via1", "via2"),
        )

    padded = via_stack(
        layers=("M1", "M2", "MTOP"),
        layer_offsets=(0, 1, 2),
        vias=("via1", "via2", None),
    )
    for layer in ("M1", "M2", "MTOP", "VIA1", "VIA2"):
        assert c.area(layer) == padded.area(layer)


@pytest.mark.parametrize("vias", [("via1",), ("via1", "via2", None, None)])
def test_via_stack_wrong_via_count_warns(vias: tuple[str | None, ...]) -> None:
    with pytest.warns(UserWarning, match="3 layers"):
        via_stack(layers=("M1", "M2", "MTOP"), vias=vias)
