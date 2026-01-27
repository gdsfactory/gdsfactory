"""Generate Python code from YAML component definitions.

This module provides code generation functionality that mirrors from_yaml.py,
but instead of creating components, it generates Python code strings that would
create those components.
"""

from __future__ import annotations

import pathlib
from typing import IO, Any

import networkx as nx

from gdsfactory.read.from_yaml import (
    _get_dependency_graph,
    _get_directed_connections,
    _graph_roots,
    _load_yaml_str,
    _parse_maybe_arrayed_instance,
    _split_route_link,
    valid_anchor_point_keywords,
    valid_anchor_value_keywords,
)
from gdsfactory.schematic import GridArray, Netlist, OrthogonalGridArray, Placement
from gdsfactory.typings import RoutingStrategies


def _format_value(value: Any, indent: int = 0) -> str:
    """Format a value for Python code generation.

    Args:
        value: Value to format.
        indent: Current indentation level.

    Returns:
        Formatted string representation.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{_format_value(k)}: {_format_value(v)}" for k, v in value.items()]
        if len(items) == 1 and len(items[0]) < 60:
            return "{" + items[0] + "}"
        return "{" + ", ".join(items) + "}"
    if isinstance(value, list | tuple):
        if not value:
            return "[]" if isinstance(value, list) else "()"
        items = [_format_value(v) for v in value]
        result = ", ".join(items)
        if isinstance(value, list):
            return f"[{result}]"
        return f"({result})" if len(value) != 1 else f"({result},)"
    return repr(value)


def from_yaml_to_code(
    yaml_str: str | pathlib.Path | IO[Any] | dict[str, Any],
    function_name: str = "create_component",
    routing_strategies: RoutingStrategies | None = None,
) -> str:
    """Generate Python code that creates a Component from YAML.

    Args:
        yaml_str: YAML string, file path, or dict.
        function_name: Name of the generated function.
        routing_strategies: Optional routing strategies for validation.

    Returns:
        Python code as a string that creates the component.

    Example:
        >>> yaml_str = '''
        ... name: my_component
        ... instances:
        ...     mmi:
        ...         component: mmi1x2
        ... '''
        >>> code = from_yaml_to_code(yaml_str)
        >>> print(code)
    """
    # Load and validate YAML using existing infrastructure
    dct = _load_yaml_str(yaml_str)
    # Keep the raw dict to know which settings were actually specified
    raw_instances = dct.get("instances", {})
    net = Netlist.model_validate(dct)
    g = _get_dependency_graph(net)

    lines: list[str] = []

    # Track required imports
    needs_kf = False

    # Add imports
    lines.append("import gdsfactory as gf")
    lines.append("from gdsfactory.component import Component")
    if any(inst.virtual for inst in net.instances.values()):
        lines.append("from gdsfactory.component import ComponentAllAngle")
    lines.append("from gdsfactory.pdk import get_active_pdk")
    lines.append("from gdsfactory.add_pins import add_instance_label")

    # Check if we need kfactory for GridArray instances
    for inst in net.instances.values():
        if isinstance(inst.array, GridArray):
            needs_kf = True
            break

    if needs_kf:
        lines.append("import kfactory as kf")

    lines.append("")
    lines.append("")

    # Function definition
    lines.append("@gf.cell")
    lines.append(f"def {function_name}() -> Component:")
    if net.name:
        lines.append(f'    """Create {net.name} component."""')
    lines.append("    pdk = get_active_pdk()")
    lines.append("    c = Component()")
    lines.append("")

    # Add instances
    if net.instances:
        lines.append("    # Create instances")
        for name, inst in net.instances.items():
            # Get raw settings from YAML (before validation filled in defaults)
            raw_inst = raw_instances.get(name, {})
            raw_settings = (
                raw_inst.get("settings", {}) if isinstance(raw_inst, dict) else {}
            )
            lines.extend(_generate_instance_code(name, inst, raw_settings))
        lines.append("")

    # Process placements and connections in dependency order
    directed_connections = _get_directed_connections(net.connections)

    has_placements_or_connections = bool(net.placements) or bool(net.connections)
    if has_placements_or_connections:
        lines.append("    # Place instances and make connections")

    for root in _graph_roots(g):
        # Place root if it has placement
        if root in net.placements:
            placement_lines = _generate_placement_code(root, net.placements[root])
            lines.extend(placement_lines)

        # Traverse the graph in DFS order
        for i2, i1 in nx.dfs_edges(g, root):
            # Get connection info
            ports = directed_connections.get(i1, {}).get(i2, None)

            # Place i1 if it has placement
            if i1 in net.placements:
                placement_lines = _generate_placement_code(i1, net.placements[i1])
                lines.extend(placement_lines)

            # Make connection if it exists
            if ports is not None:
                connection_lines = _generate_connection_code(i1, i2, ports)
                lines.extend(connection_lines)

    if has_placements_or_connections:
        lines.append("")

    # Add routes
    if net.routes:
        lines.append("    # Add routes")
        for bundle_name, bundle in net.routes.items():
            lines.extend(_generate_route_code(bundle_name, bundle))
        lines.append("")

    # Add instance labels
    if net.instances:
        lines.append("    # Add instance labels")
        lines.extend(
            f"    add_instance_label(c, {name}, instance_name={_format_value(name)})"
            for name in net.instances
        )
        lines.append("")

    # Add ports
    if net.ports:
        lines.append("    # Expose ports")
        for port_name, port_spec in net.ports.items():
            lines.extend(_generate_port_code(port_name, port_spec))
        lines.append("")

    # Set component info
    if net.info:
        lines.append("    # Set component info")
        for key, value in net.info.items():
            lines.append(f"    c.info[{_format_value(key)}] = {_format_value(value)}")
        lines.append("")

    # Set name
    if net.name:
        lines.append(f"    c.name = {_format_value(net.name)}")
        lines.append("")

    # Return component
    lines.append("    return c")

    return "\n".join(lines)


def _generate_instance_code(
    name: str, inst: Any, raw_settings: dict[str, Any]
) -> list[str]:
    """Generate code for creating an instance.

    Args:
        name: Instance name.
        inst: Instance specification from netlist.
        raw_settings: Settings actually specified in YAML (before validation).

    Returns:
        List of code lines.
    """
    lines: list[str] = []

    # Build component getter - unpack settings as kwargs from YAML
    comp_str = _format_value(inst.component)

    # Only include settings that were actually specified in the YAML (not defaults)
    if raw_settings:
        settings_kwargs = ", ".join(
            f"{k}={_format_value(v)}" for k, v in raw_settings.items()
        )
        component_getter = f"pdk.get_component({comp_str}, {settings_kwargs})"
    else:
        component_getter = f"pdk.get_component({comp_str})"

    # Handle different instance types
    if isinstance(inst.array, OrthogonalGridArray):
        arr = inst.array
        # Build arguments list - only include what's specified
        args = [
            f"        {component_getter},",
            f"        rows={arr.rows},",
            f"        columns={arr.columns},",
            f"        name={_format_value(name)},",
        ]
        if arr.column_pitch is not None:
            args.append(f"        column_pitch={arr.column_pitch},")
        if arr.row_pitch is not None:
            args.append(f"        row_pitch={arr.row_pitch},")
        # Remove trailing comma from last arg
        args[-1] = args[-1].rstrip(",")

        lines.append(f"    {name} = c.add_ref(")
        lines.extend(args)
        lines.append("    )")

    elif isinstance(inst.array, GridArray):
        grid_arr = inst.array
        args = [
            f"        {component_getter},",
            f"        na={grid_arr.num_a},",
            f"        nb={grid_arr.num_b},",
            f"        a=kf.kdb.DVector({grid_arr.pitch_a[0]}, {grid_arr.pitch_a[1]}),",
            f"        b=kf.kdb.DVector({grid_arr.pitch_b[0]}, {grid_arr.pitch_b[1]}),",
        ]
        # Remove trailing comma from last arg
        args[-1] = args[-1].rstrip(",")

        lines.append(f"    {name} = c.create_inst(")
        lines.extend(args)
        lines.append("    )")

    else:
        # Regular instance
        if inst.virtual:
            lines.append(f"    {name} = c.add_ref_off_grid({component_getter})")
            lines.append(f"    {name}.name = {_format_value(name)}")
        else:
            lines.append(
                f"    {name} = c.add_ref({component_getter}, name={_format_value(name)})"
            )

    return lines


def _generate_placement_code(inst_name: str, placement: Placement) -> list[str]:
    """Generate code for placing an instance using chained rotate().mirror().move() pattern.

    The order is always: rotate -> mirror -> move
    This maps 1-1 to a KLayout DCplxTrans transformation.

    Args:
        inst_name: Instance name.
        placement: Placement specification.

    Returns:
        List of code lines.
    """
    lines: list[str] = []

    # Build chained transformation: rotate().mirror().move()
    chain_parts: list[str] = []

    # 1. Rotation (if specified)
    if placement.rotation and placement.rotation != 0:
        if placement.port:
            center_code = _get_anchor_point_code(inst_name, placement.port)
            chain_parts.append(f"rotate({placement.rotation}, center={center_code})")
        else:
            chain_parts.append(f"rotate({placement.rotation})")

    # 2. Mirror (if specified)
    if placement.mirror:
        if placement.mirror is True:
            if placement.port:
                anchor_x_code = _get_anchor_value_code(inst_name, placement.port, "x")
                chain_parts.append(
                    f"mirror(p1=({anchor_x_code}, 0), p2=({anchor_x_code}, 1))"
                )
            else:
                chain_parts.append("mirror()")
        elif isinstance(placement.mirror, str):
            port_x = f"{inst_name}.ports[{_format_value(placement.mirror)}].x"
            chain_parts.append(f"mirror(p1=({port_x}, 0), p2=({port_x}, 1))")
        else:
            # Numeric mirror
            chain_parts.append(
                f"mirror(p1=({placement.mirror}, 0), p2=({placement.mirror}, 1))"
            )

    # 3. Calculate final position
    # Start with port anchor offset if specified
    x_offset_parts: list[str] = []
    y_offset_parts: list[str] = []

    if placement.port:
        anchor_code = _get_anchor_point_code(inst_name, placement.port)
        x_offset_parts.append(f"-{anchor_code}[0]")
        y_offset_parts.append(f"-{anchor_code}[1]")

    # Add x/xmin/xmax positioning
    if placement.x is not None:
        x_code = _generate_position_code(placement.x, "x")
        x_offset_parts.append(x_code)
    elif placement.xmin is not None:
        xmin_code = _generate_position_code(placement.xmin, "x")
        # For xmin, we need to calculate: target_xmin - current_xmin
        lines.append(f"    _target_xmin = {xmin_code}")
        lines.append(f"    _x_offset = _target_xmin - {inst_name}.xmin")
        x_offset_parts.append("_x_offset")
    elif placement.xmax is not None:
        xmax_code = _generate_position_code(placement.xmax, "x")
        lines.append(f"    _target_xmax = {xmax_code}")
        lines.append(f"    _x_offset = _target_xmax - {inst_name}.xmax")
        x_offset_parts.append("_x_offset")

    # Add y/ymin/ymax positioning
    if placement.y is not None:
        y_code = _generate_position_code(placement.y, "y")
        y_offset_parts.append(y_code)
    elif placement.ymin is not None:
        ymin_code = _generate_position_code(placement.ymin, "y")
        lines.append(f"    _target_ymin = {ymin_code}")
        lines.append(f"    _y_offset = _target_ymin - {inst_name}.ymin")
        y_offset_parts.append("_y_offset")
    elif placement.ymax is not None:
        ymax_code = _generate_position_code(placement.ymax, "y")
        lines.append(f"    _target_ymax = {ymax_code}")
        lines.append(f"    _y_offset = _target_ymax - {inst_name}.ymax")
        y_offset_parts.append("_y_offset")

    # Add dx/dy offsets
    if placement.dx is not None:
        x_offset_parts.append(str(placement.dx))
    if placement.dy is not None:
        y_offset_parts.append(str(placement.dy))

    # Build final position expression
    if x_offset_parts or y_offset_parts:
        x_expr = " + ".join(x_offset_parts) if x_offset_parts else "0"
        y_expr = " + ".join(y_offset_parts) if y_offset_parts else "0"
        chain_parts.append(f"move(({x_expr}, {y_expr}))")

    # Generate the chained call
    if chain_parts:
        chain = ".".join(chain_parts)
        lines.append(f"    {inst_name}.{chain}")

    return lines


def _get_anchor_point_code(inst_name: str, anchor: str) -> str:
    """Generate code to get an anchor point.

    Args:
        inst_name: Instance name.
        anchor: Anchor name (port name or keyword like 'center', 'ce', etc.).

    Returns:
        Code string that evaluates to the anchor point.
    """
    if anchor in valid_anchor_point_keywords:
        return f"{inst_name}.dsize_info.{anchor}"
    return f"{inst_name}.ports[{_format_value(anchor)}].center"


def _get_anchor_value_code(inst_name: str, anchor: str, coord: str) -> str:
    """Generate code to get an anchor value (x or y coordinate).

    Args:
        inst_name: Instance name.
        anchor: Anchor name.
        coord: Either 'x' or 'y'.

    Returns:
        Code string that evaluates to the coordinate value.
    """
    if anchor in valid_anchor_value_keywords:
        return f"{inst_name}.dsize_info.{anchor}"
    if anchor in valid_anchor_point_keywords:
        return (
            f"{_get_anchor_point_code(inst_name, anchor)}[{0 if coord == 'x' else 1}]"
        )
    return f"{inst_name}.ports[{_format_value(anchor)}].{coord}"


def _generate_position_code(value: str | int | float, coord: str) -> str:
    """Generate code for a position value (handles references like 'inst,port').

    Args:
        value: Position value (number or reference string).
        coord: Either 'x' or 'y'.

    Returns:
        Code string.
    """
    if isinstance(value, str):
        inst_ref, anchor_ref = value.split(",", 1)
        inst_ref = inst_ref.strip()
        anchor_ref = anchor_ref.strip()

        if anchor_ref in valid_anchor_value_keywords:
            return _get_anchor_value_code(inst_ref, anchor_ref, coord)
        return f"{inst_ref}.ports[{_format_value(anchor_ref)}].{coord}"
    return str(value)


def _generate_connection_code(i1: str, i2: str, ports: tuple[str, str]) -> list[str]:
    """Generate code for connecting two instances.

    Args:
        i1: Source instance name (may include array indices).
        i2: Destination instance name (may include array indices).
        ports: Tuple of (port1, port2) names.

    Returns:
        List of code lines.
    """
    lines: list[str] = []
    p1, p2 = ports

    i1name, i1a, i1b = _parse_maybe_arrayed_instance(i1)
    i2name, i2a, i2b = _parse_maybe_arrayed_instance(i2)

    # Build port access code
    if i1a is not None and i1b is not None:
        port1_code = f"{i1name}.ports[{_format_value(p1)}, {i1a}, {i1b}]"
        if i2a is not None and i2b is not None:
            port2_code = f"{i2name}.ports[{_format_value(p2)}, {i2a}, {i2b}]"
            lines.append(f"    {i1name}.connect({port1_code}, {port2_code})")
        else:
            lines.append(
                f"    {i1name}.connect({port1_code}, other={i2}, "
                f"other_port_name={_format_value(p2)})"
            )
    else:
        if i2a is not None and i2b is not None:
            lines.append(
                f"    {i1}.connect({_format_value(p1)}, other={i2name}, "
                f"other_port_name=({_format_value(p2)}, {i2a}, {i2b}))"
            )
        else:
            lines.append(
                f"    {i1}.connect({_format_value(p1)}, other={i2}, "
                f"other_port_name={_format_value(p2)})"
            )

    return lines


def _generate_route_code(bundle_name: str, bundle: Any) -> list[str]:
    """Generate code for creating routes.

    Args:
        bundle_name: Route bundle name.
        bundle: Bundle specification.

    Returns:
        List of code lines.
    """
    lines: list[str] = []

    routing_strategy = bundle.routing_strategy
    lines.append(f"    # Route: {bundle_name}")

    # Collect port access code
    ports1_code: list[str] = []
    ports2_code: list[str] = []

    # Process links
    for ip1, ip2 in bundle.links.items():
        first1, middles1, last1 = _split_route_link(ip1)
        first2, middles2, last2 = _split_route_link(ip2)

        # Generate port gathering code
        for m1, m2 in zip(middles1, middles2, strict=False):
            ip1_full = first1 + m1 + last1
            ip2_full = first2 + m2 + last2

            # Parse instance and port
            i1, p1 = ip1_full.split(",", 1)
            i2, p2 = ip2_full.split(",", 1)

            i1name, i1a, i1b = _parse_maybe_arrayed_instance(i1)
            i2name, i2a, i2b = _parse_maybe_arrayed_instance(i2)

            # Generate port access
            if i1a is None or i1b is None:
                port1_code = f"{i1name}.ports[{_format_value(p1)}]"
            else:
                port1_code = f"{i1name}.ports[{_format_value(p1)}, {i1a}, {i1b}]"

            if i2a is None or i2b is None:
                port2_code = f"{i2name}.ports[{_format_value(p2)}]"
            else:
                port2_code = f"{i2name}.ports[{_format_value(p2)}, {i2a}, {i2b}]"

            ports1_code.append(port1_code)
            ports2_code.append(port2_code)

    # Build port list literals
    ports1_literal = "[" + ", ".join(ports1_code) + "]"
    ports2_literal = "[" + ", ".join(ports2_code) + "]"

    # Generate routing call
    # Build settings kwargs
    settings_str = ""
    if bundle.settings:
        settings_items = [f"{k}={_format_value(v)}" for k, v in bundle.settings.items()]
        settings_str = ", " + ", ".join(settings_items)

    lines.append(
        f"    pdk.routing_strategies[{_format_value(routing_strategy)}](c, ports1={ports1_literal}, ports2={ports2_literal}{settings_str})"
    )
    lines.append("")

    return lines


def _generate_port_code(port_name: str, port_spec: str) -> list[str]:
    """Generate code for exposing a port.

    Args:
        port_name: External port name.
        port_spec: Port specification (instance,port format).

    Returns:
        List of code lines.
    """
    lines: list[str] = []

    inst_name, port_name_ref = port_spec.split(",", 1)
    inst_name = inst_name.strip()
    port_name_ref = port_name_ref.strip()

    inst_name_parsed, ia, ib = _parse_maybe_arrayed_instance(inst_name)

    if ia is None or ib is None:
        port_code = f"{inst_name_parsed}.ports[{_format_value(port_name_ref)}]"
    else:
        port_code = (
            f"{inst_name_parsed}.ports[{_format_value(port_name_ref)}, {ia}, {ib}]"
        )

    lines.append(f"    c.add_port({_format_value(port_name)}, port={port_code})")

    return lines
