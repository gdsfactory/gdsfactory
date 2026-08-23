import yaml

import gdsfactory as gf


@gf.vcell
def extend_all_angle(component: gf.Component) -> gf.ComponentAllAngle:
    extended = gf.ComponentAllAngle()
    extended.add_ref_off_grid(component)
    extension = extended.add_ref_off_grid(
        gf.components.straight_all_angle(
            cross_section=gf.get_cross_section(component.ports["o2"].cross_section)
        )
    )
    extension.connect("o2", component.ports["o1"])
    return extended


def test_all_angle_netlist_yaml_round_trip() -> None:
    component = extend_all_angle(gf.components.straight())
    serialized = yaml.safe_dump(component.get_netlist())

    restored = gf.read.from_yaml(serialized)

    assert len(restored.vinsts) == 2
