from __future__ import annotations

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.read import from_yaml


@cell
def demo_all_angle_routing() -> Component:
    """Demonstrate all-angle routing."""
    yaml = """
    instances:
        mmi_long:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 10
        mmi_short:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 5

    placements:
        mmi_long:
            rotation: 190
            x: 100
            y: 100

    routes:
        optical:
            routing_strategy: get_bundle_all_angle
            settings:
                steps:
                    - ds: 50
                      exit_angle: 90  # TODO: why do paths cross when set to i.e. 100?
            links:
                mmi_short,o2: mmi_long,o3
                mmi_short,o3: mmi_long,o2

    ports:
        o2: mmi_short,o1
        o1: mmi_long,o1
    """

    return from_yaml(yaml)


if __name__ == "__main__":
    from gdsfactory.pdk import get_active_pdk

    # IMPORTANT: always use this gds write flag when using non-manhattan features
    get_active_pdk().gds_write_settings.flatten_invalid_refs = True
    c = demo_all_angle_routing()
    c.show(show_ports=True)
