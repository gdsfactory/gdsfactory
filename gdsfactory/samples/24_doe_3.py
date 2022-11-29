"""Design of Experiment (DOE) with custom add_fiber_array function.

In this case add_fiber_array does not add labels.

You can use gf.add_labels.add_labels_to_ports.

"""
from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.pack_doe_grid(
        gf.components.straight,
        settings={"length": [5, 5]},
        function=gf.partial(gf.routing.add_fiber_array, get_input_labels_function=None),
    )
    c = gf.add_labels.add_labels_to_ports(
        component=c, prefix="opt_te1550_", port_type="vertical_te"
    )
    print(len(c.labels))
    c.show(show_ports=False)
    # c.write_gds_with_metadata(f"{__file__[:-3]}/test.gds")
