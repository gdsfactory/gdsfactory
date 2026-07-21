import gdsfactory as gf
from gdsfactory.components.vias.via_stack import via_stack
from gdsfactory.components.vias.via_stack_with_offset import via_stack_with_offset


class TestViaStackPorts:
    def test_default_ports_on_top_layer(self) -> None:
        c = via_stack()
        port_names = [p.name for p in c.ports]
        assert port_names == ["e1", "e2", "e3", "e4"]

    def test_string_key_resolves(self) -> None:
        c = via_stack(
            layers=("M1", "M2", "MTOP"),
            vias=("via1", "via2", None),
            layer_to_port_orientations={"M1": [180, 90, 0, -90]},
        )
        port_names = [p.name for p in c.ports]
        assert port_names == ["e1", "e2", "e3", "e4"]
        assert all(p.layer == gf.get_layer("M1") for p in c.ports)

    def test_multiple_layers_suffixed(self) -> None:
        c = via_stack(
            layers=("M1", "M2", "MTOP"),
            vias=("via1", "via2", None),
            layer_to_port_orientations={"M1": [180, 0], "MTOP": [180, 0]},
        )
        port_names = sorted(p.name for p in c.ports)
        m1_layer = gf.get_layer("M1")
        mtop_layer = gf.get_layer("MTOP")
        m1_name = (
            m1_layer.name
            if hasattr(m1_layer, "name")
            else f"{m1_layer[0]}_{m1_layer[1]}"
        )
        mtop_name = (
            mtop_layer.name
            if hasattr(mtop_layer, "name")
            else f"{mtop_layer[0]}_{mtop_layer[1]}"
        )
        assert f"e1_{m1_name}" in port_names
        assert f"e1_{mtop_name}" in port_names
        assert len(port_names) == 4

    def test_no_ports_when_none_orientations(self) -> None:
        c = via_stack(
            layers=("M1", "M2", "MTOP"),
            vias=("via1", "via2", None),
            port_orientations=None,
            layer_to_port_orientations={},
        )
        electrical = [p for p in c.ports if p.port_type == "electrical"]
        assert len(electrical) == 0


class TestViaStackWithOffsetPorts:
    def test_default_ports_on_top_layer(self) -> None:
        c = via_stack_with_offset()
        port_names = [p.name for p in c.ports]
        assert port_names == ["e1", "e2", "e3", "e4"]

    def test_string_key_resolves(self) -> None:
        c = via_stack_with_offset(
            layers=("PPP", "M1"),
            vias=(None, "viac"),
            layer_to_port_orientations={"PPP": [180, 90, 0, -90]},
        )
        port_names = [p.name for p in c.ports]
        assert port_names == ["e1", "e2", "e3", "e4"]
        assert all(p.layer == gf.get_layer("PPP") for p in c.ports)

    def test_multiple_layers_suffixed(self) -> None:
        c = via_stack_with_offset(
            layers=("M1", "M2", "MTOP"),
            vias=(None, "via1", "via2"),
            layer_to_port_orientations={"M1": [180, 0], "MTOP": [180, 0]},
        )
        port_names = sorted(p.name for p in c.ports)
        m1_layer = gf.get_layer("M1")
        mtop_layer = gf.get_layer("MTOP")
        m1_name = (
            m1_layer.name
            if hasattr(m1_layer, "name")
            else f"{m1_layer[0]}_{m1_layer[1]}"
        )
        mtop_name = (
            mtop_layer.name
            if hasattr(mtop_layer, "name")
            else f"{mtop_layer[0]}_{mtop_layer[1]}"
        )
        assert f"e1_{m1_name}" in port_names
        assert f"e1_{mtop_name}" in port_names
        assert len(port_names) == 4

    def test_invalid_layer_in_port_orientations_raises(self) -> None:
        import pytest

        with pytest.raises(
            ValueError, match="layer_to_port_orientations not in layers"
        ):
            via_stack_with_offset(
                layers=("PPP", "M1"),
                vias=(None, "viac"),
                layer_to_port_orientations={"MTOP": [180, 0]},
            )
