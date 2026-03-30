"""Sample: routing with a designated path length using meander components.

Demonstrates ``route_bundle_variable_path_length`` which inserts a
``delay_snake`` meander into the route so that the total routed path
length matches a user-specified target.

The meander component absorbs the difference between the natural
(baseline) Manhattan routing length and the requested target.
"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    xs = gf.get_cross_section("strip")
    layer = gf.get_layer(xs.sections[0].layer)

    # --- Example 1: single route with waypoints and target path length ---
    c1 = gf.Component("variable_path_single")

    port1 = gf.Port(
        name="in0",
        center=(0, 0),
        orientation=0,
        layer=layer,
        width=0.5,
    )
    port2 = gf.Port(
        name="out0",
        center=(400, 100),
        orientation=180,
        layer=layer,
        width=0.5,
    )

    results = gf.routing.route_bundle_variable_path_length(
        c1,
        ports1=port1,
        ports2=port2,
        target_length=1200,
        cross_section="strip",
        waypoints=[(200, 0), (200, 100)],
        meander_n_loops=4,
    )

    for r in results:
        print(
            f"[Single] target={r.target_length:.1f}  "
            f"actual={r.actual_length:.1f}  "
            f"baseline={r.baseline_length:.1f}  "
            f"has_meander={r.meander_instance is not None}"
        )
    c1.show()

    # --- Example 2: straight route, no waypoints ---
    c2 = gf.Component("variable_path_straight")

    port3 = gf.Port(
        name="in1",
        center=(0, 0),
        orientation=0,
        layer=layer,
        width=0.5,
    )
    port4 = gf.Port(
        name="out1",
        center=(500, 0),
        orientation=180,
        layer=layer,
        width=0.5,
    )

    results2 = gf.routing.route_bundle_variable_path_length(
        c2,
        ports1=port3,
        ports2=port4,
        target_length=2000,
        cross_section="strip",
        meander_n_loops=4,
    )

    for r in results2:
        print(
            f"[Straight] target={r.target_length:.1f}  "
            f"actual={r.actual_length:.1f}  "
            f"baseline={r.baseline_length:.1f}"
        )
    c2.show()

    # --- Example 3: bundle with per-route target lengths ---
    c3 = gf.Component("variable_path_bundle")

    start_ports = [
        gf.Port(
            name=f"in{i}",
            center=(0, i * 20),
            orientation=0,
            layer=layer,
            width=0.5,
        )
        for i in range(3)
    ]
    end_ports = [
        gf.Port(
            name=f"out{i}",
            center=(500, 50 + i * 20),
            orientation=180,
            layer=layer,
            width=0.5,
        )
        for i in range(3)
    ]

    results3 = gf.routing.route_bundle_variable_path_length(
        c3,
        ports1=start_ports,
        ports2=end_ports,
        target_length=[1500, 1500, 1500],
        cross_section="strip",
        waypoints=[(250, 0), (250, 50)],
        meander_n_loops=4,
    )

    for i, r in enumerate(results3):
        print(
            f"[Bundle {i}] target={r.target_length:.1f}  "
            f"actual={r.actual_length:.1f}  "
            f"baseline={r.baseline_length:.1f}"
        )
    c3.show()
