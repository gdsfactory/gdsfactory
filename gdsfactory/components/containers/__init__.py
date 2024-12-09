from gdsfactory.components.containers import (
    add_fiber_array_optical_south_electrical_north,
    add_termination,
    add_trenches,
    array_component,
    component_sequence,
    copy_layers,
    extend_ports_list,
    extension,
    pack_doe,
    splitter_chain,
    splitter_tree,
)
from gdsfactory.components.containers.add_fiber_array_optical_south_electrical_north import (
    add_fiber_array_optical_south_electrical_north,
)
from gdsfactory.components.containers.add_termination import (
    add_termination,
    terminator_function,
)
from gdsfactory.components.containers.add_trenches import (
    add_trenches,
    add_trenches90,
)
from gdsfactory.components.containers.array_component import (
    array,
)
from gdsfactory.components.containers.component_sequence import (
    SequenceGenerator,
    component_sequence,
    parse_component_name,
)
from gdsfactory.components.containers.copy_layers import (
    copy_layers,
)
from gdsfactory.components.containers.extend_ports_list import (
    extend_ports_list,
)
from gdsfactory.components.containers.extension import (
    DEG2RAD,
    extend_ports,
    line,
    move_polar_rad_copy,
)
from gdsfactory.components.containers.pack_doe import (
    generate_doe,
    pack_doe,
    pack_doe_grid,
)
from gdsfactory.components.containers.splitter_chain import (
    splitter_chain,
)
from gdsfactory.components.containers.splitter_tree import (
    splitter_tree,
    switch_tree,
    test_splitter_tree_ports,
    test_splitter_tree_ports_no_sbend,
)

__all__ = [
    "DEG2RAD",
    "SequenceGenerator",
    "add_fiber_array_optical_south_electrical_north",
    "add_termination",
    "add_trenches",
    "add_trenches90",
    "array",
    "array_component",
    "component_sequence",
    "copy_layers",
    "extend_ports",
    "extend_ports_list",
    "extension",
    "generate_doe",
    "line",
    "move_polar_rad_copy",
    "pack_doe",
    "pack_doe_grid",
    "parse_component_name",
    "splitter_chain",
    "splitter_tree",
    "switch_tree",
    "terminator_function",
    "test_splitter_tree_ports",
    "test_splitter_tree_ports_no_sbend",
]
