"""Extract netlist from component port connectivity."""

import re
import secrets
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain, product
from typing import Any, Protocol, TypeAlias, cast

import kfactory as kf

import gdsfactory as gf
import gdsfactory.schematic as scm
from gdsfactory.component import Component

Instance: TypeAlias = kf.DInstance | kf.VInstance | kf.Instance


class ComponentNamer(Protocol):
    """Protocol for naming components in netlists."""

    def __call__(self, cell: kf.ProtoTKCell[Any]) -> str:
        """Return the component name for the given cell."""
        ...


def factory_namer(cell: kf.ProtoTKCell[Any]) -> str:
    """Names components using their factory name."""
    try:
        return cell.factory_name
    except ValueError:
        return cell.name


def function_namer(cell: kf.ProtoTKCell[Any]) -> str:
    """Names components using their function name, falling back to factory name."""
    try:
        return cell.function_name or cell.factory_name
    except ValueError:
        return cell.name


def cell_namer(cell: kf.ProtoTKCell[Any]) -> str:
    """Names components using their cell name."""
    return cell.name


class InstanceNamer(Protocol):
    """Protocol for naming instances in netlists."""

    def __call__(self, inst: Instance) -> str:
        """Return the instance name for the given instance."""
        ...


class OriginalNamer:
    """Names instances using their original instance name."""

    def __init__(self) -> None:
        self._instance_names: dict[str | None, str] = {}
        self._rev_instance_names: dict[str, str | None] = {}

    def __call__(self, inst: Instance) -> str:
        if inst.name in self._instance_names:
            return self._instance_names[inst.name]
        name = self._instance_names[inst.name] = _clean_instname(inst.name)
        self._rev_instance_names[name] = inst.name
        return name


class CountedNamer:
    """Names instances using their component name with numeric suffixes."""

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._instance_names: dict[str | None, str] = {}
        self._rev_instance_names: dict[str, str | None] = {}

    def __call__(self, inst: Instance) -> str:
        if inst.name in self._instance_names:
            return self._instance_names[inst.name]
        compname = _short_component_name(
            cast(kf.ProtoTKCell[Any], inst.cell), self._component_namer
        )
        name = _instname_from_compname(
            inst, compname, self._instance_names, self._rev_instance_names
        )
        name = self._instance_names[inst.name] = _clean_instname(name)
        self._rev_instance_names[name] = inst.name
        return name


class SmartNamer:
    """Names instances using component name if auto-generated, otherwise instance name."""

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._instance_names: dict[str | None, str] = {}
        self._rev_instance_names: dict[str, str | None] = {}

    def __call__(self, inst: Instance) -> str:
        if inst.name in self._instance_names:
            return self._instance_names[inst.name]
        compname = _short_component_name(
            cast(kf.ProtoTKCell[Any], inst.cell), self._component_namer
        )
        if inst.name is not None and inst.name.startswith(f"{compname}_"):
            name: str | None = _instname_from_compname(
                inst, compname, self._instance_names, self._rev_instance_names
            )
        else:
            name = inst.name
        cleaned = self._instance_names[inst.name] = _clean_instname(name)
        self._rev_instance_names[cleaned] = inst.name
        return cleaned


class NetlistNamer(Protocol):
    """Protocol for naming cells in recursive netlists."""

    def __call__(self, cell: kf.ProtoTKCell[Any]) -> str:
        """Return the name for the given cell in the netlist."""
        ...


class CountedNetlistNamer:
    """Names cells with counting for uniqueness in recursive netlists.

    Uses component_namer for leaf cells (no instances) and counted naming
    for hierarchical cells (with instances).
    """

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._cell_names: dict[str, str] = {}  # cell.name -> assigned name
        self._used_names: set[str] = set()

    def __call__(self, cell: kf.ProtoTKCell[Any]) -> str:
        if cell.name in self._cell_names:
            return self._cell_names[cell.name]

        if not _has_instances(cell):
            # Leaf cell: use component_namer (typically function_name)
            name = self._component_namer(cell)
        else:
            # Hierarchical cell: use counted naming
            base = self._component_namer(cell)
            name = self._get_unique_name(base)

        self._cell_names[cell.name] = name
        self._used_names.add(name)
        return name

    def _get_unique_name(self, base: str) -> str:
        # Always use a numbered suffix for hierarchical cells to distinguish
        # netlist keys from function names (which are used for leaf cells)
        # If base ends with a number, add underscore
        basename = re.sub("[0-9]*$", "", base)
        if basename != base:
            basename = f"{base}_"
        index = 1
        while (name := f"{basename}{index}") in self._used_names:
            index += 1
        return name


# PortMatcher can return:
# - False: ports don't match
# - True: ports match with no metadata
# - dict: ports match with metadata (stored in net's 'settings' field)
MatchResult: TypeAlias = bool | dict[str, Any]


class PortMatcher(Protocol):
    """Protocol for determining if two ports are connected."""

    def __call__(
        self, port1: kf.DPort | kf.Port, port2: kf.DPort | kf.Port
    ) -> MatchResult:
        """Return match result: False, True, or dict with metadata."""
        ...


class PortCenterMatcher:
    """Matches ports based on center position only."""

    def __init__(self, tolerance_dbu: int = 2) -> None:
        self.tolerance_dbu = tolerance_dbu

    def __call__(self, port1: kf.DPort | kf.Port, port2: kf.DPort | kf.Port) -> bool:
        """Return True if two ports are at the same location (within tolerance)."""
        if port1.port_type != port2.port_type:
            return False
        x1, y1 = port1.to_itype().center
        x2, y2 = port2.to_itype().center
        return abs(x2 - x1) < self.tolerance_dbu and abs(y2 - y1) < self.tolerance_dbu


class SmartPortMatcher:
    """Matches ports based on position, width, and orientation.

    For electrical ports, orientation is not checked.
    For other port types (e.g. optical), ports must face each other (180° apart).
    """

    def __init__(
        self,
        position_tolerance_dbu: int = 2,
        width_tolerance: float = 0.001,
        angle_tolerance: float = 0.01,
    ) -> None:
        self.position_tolerance_dbu = position_tolerance_dbu
        self.width_tolerance = width_tolerance
        self.angle_tolerance = angle_tolerance

    def __call__(self, port1: kf.DPort | kf.Port, port2: kf.DPort | kf.Port) -> bool:
        """Return True if ports match in position, width, and orientation."""
        # Check position
        if port1.port_type != port2.port_type:
            return False
        x1, y1 = port1.to_itype().center
        x2, y2 = port2.to_itype().center
        if (
            abs(x2 - x1) >= self.position_tolerance_dbu
            or abs(y2 - y1) >= self.position_tolerance_dbu
        ):
            return False

        # Check width
        if abs(port1.width - port2.width) > self.width_tolerance:
            return False

        # Skip orientation check for electrical ports
        if port1.port_type == "electrical" or port2.port_type == "electrical":
            return True

        # Check orientation: ports should face each other (180° apart)
        angle_diff = abs(_angle_difference(port1.orientation, port2.orientation))
        if abs(angle_diff - 180) > self.angle_tolerance:  # noqa: SIM103
            return False

        return True


class FlexiblePortMatcher:
    """Matches ports based on position and orientation, recording width mismatches.

    Unlike SmartPortMatcher, this matcher does not reject connections with
    width mismatches. Instead, it returns metadata about the mismatch that
    can be used for post-processing (e.g., inserting interface components).
    """

    def __init__(
        self,
        position_tolerance_dbu: int = 2,
        angle_tolerance: float = 0.01,
    ) -> None:
        self.position_tolerance_dbu = position_tolerance_dbu
        self.angle_tolerance = angle_tolerance

    def __call__(
        self, port1: kf.DPort | kf.Port, port2: kf.DPort | kf.Port
    ) -> MatchResult:
        """Return match result with width mismatch metadata if applicable."""
        # Check port type
        if port1.port_type != port2.port_type:
            return False

        # Check position
        x1, y1 = port1.to_itype().center
        x2, y2 = port2.to_itype().center
        if (
            abs(x2 - x1) >= self.position_tolerance_dbu
            or abs(y2 - y1) >= self.position_tolerance_dbu
        ):
            return False

        # Skip orientation check for electrical ports
        if port1.port_type != "electrical":
            # Check orientation: ports should face each other (180° apart)
            angle_diff = abs(_angle_difference(port1.orientation, port2.orientation))
            if abs(angle_diff - 180) > self.angle_tolerance:
                return False

        # Check for width mismatch - record but don't reject
        width_diff = abs(port1.width - port2.width)
        if width_diff > 0.001:  # small tolerance for floating point
            return {
                "width1": port1.width,
                "width2": port2.width,
            }

        return True


def _angle_difference(angle1: float, angle2: float) -> float:
    """Return the difference between two angles, normalized to [-180, 180]."""
    diff = angle2 - angle1
    while diff < -180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff


def _flip_port(port: kf.DPort | kf.Port) -> kf.DPort:
    """Return a copy of the port with orientation flipped by 180°."""
    return kf.DPort(
        name=port.name,
        center=(port.x, port.y),
        orientation=port.orientation + 180,
        width=port.width,
        layer=port.layer,
        port_type=port.port_type,
    )


_default_port_matcher = SmartPortMatcher()


def get_netlist(
    cell: kf.ProtoTKCell[Any],
    *,
    allow_multiple: bool = False,
    instance_namer: InstanceNamer | None = None,
    component_namer: ComponentNamer = function_namer,
    port_matcher: PortMatcher | None = None,
) -> dict[str, Any]:
    """Extract netlist from a cell's port connectivity.

    Args:
        cell: The cell to extract the netlist from.
        allow_multiple: If True, allow more than two ports to overlap.
        instance_namer: Callable to name instances.
            Defaults to SmartNamer(component_namer).
        component_namer: Callable to name components.
            Defaults to function_namer.
        port_matcher: Callable to determine if two ports are connected.
            Defaults to SmartPortMatcher().

    Returns:
        A dictionary containing instances, placements, and nets.
    """
    recnet: dict[str, dict[str, Any]] = {}
    _insert_netlist(
        recnet,
        cell,
        allow_multiple,
        instance_namer or SmartNamer(component_namer),
        component_namer,
        component_namer,  # netlist_namer: use component_namer for non-recursive
        port_matcher or _default_port_matcher,
        recursive=False,
    )
    return recnet[next(iter(recnet))]


def get_netlist_recursive(
    cell: kf.ProtoTKCell[Any],
    *,
    allow_multiple: bool = False,
    instance_namer: InstanceNamer | None = None,
    component_namer: ComponentNamer = function_namer,
    netlist_namer: NetlistNamer | None = None,
    port_matcher: PortMatcher | None = None,
) -> dict[str, Any]:
    """Extract netlists recursively from a cell and all its subcells.

    Args:
        cell: The cell to extract the netlist from.
        allow_multiple: If True, allow more than two ports to overlap.
        instance_namer: Callable to name instances.
            Defaults to SmartNamer(component_namer).
        component_namer: Callable to name components in instance dicts.
            Defaults to function_namer.
        netlist_namer: Callable to name cells in the recursive netlist.
            Defaults to CountedNetlistNamer(component_namer).
        port_matcher: Callable to determine if two ports are connected.
            Defaults to SmartPortMatcher().

    Returns:
        A dictionary mapping cell names to their netlists.
    """
    recnet: dict[str, Any] = {}
    _insert_netlist(
        recnet,
        cell,
        allow_multiple,
        instance_namer or SmartNamer(component_namer),
        component_namer,
        netlist_namer or CountedNetlistNamer(component_namer),
        port_matcher or _default_port_matcher,
        recursive=True,
    )
    return recnet


def _insert_netlist(
    recnet: dict[str, Any],
    cell: kf.ProtoTKCell[Any],
    allow_multiple: bool,
    instance_namer: InstanceNamer,
    component_namer: ComponentNamer,
    netlist_namer: NetlistNamer,
    port_matcher: PortMatcher,
    recursive: bool,
) -> None:
    cell_name = netlist_namer(cell)
    if cell_name in recnet:
        return
    net = recnet[cell_name] = {
        "instances": {},
        "placements": {},
        "ports": {},
        "nets": {},
    }
    _all_ports: dict[str, kf.DPort | kf.Port] = {}
    for _inst in sorted(
        chain(cell.insts, cell.vinsts), key=lambda i: getattr(i, "name", "") or ""
    ):
        inst = cast(Instance, _inst)
        inst_name = instance_namer(inst)
        if _is_pure_vinst(inst):
            if _is_array_inst(inst):
                msg = (
                    "Cannot export netlist: virtual array instances are not supported."
                )
                raise ValueError(msg)
            array = None
            virtual = True
            transform = inst.dcplx_trans
            _all_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})
        elif _is_flattened_vinst(inst):
            if _is_array_inst(inst):
                msg = (
                    "Cannot export netlist: virtual array instances are not supported."
                )
            if hasattr(inst, "to_dtype"):  # always True - just making mypy happy
                inst = inst.to_dtype()
            array = None
            virtual = True
            if hasattr(inst.cell, "vtrans"):  # always True - just making mypy happy
                transform = inst.dcplx_trans * inst.cell.vtrans
            _all_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})
        elif _is_array_inst(inst):
            if hasattr(inst, "to_dtype"):  # always True - just making mypy happy
                inst = inst.to_dtype()
            array = _get_array_config(inst)
            virtual = False
            transform = inst.dcplx_trans
            _all_ports.update(
                {
                    f"{inst_name}<{a}.{b}>,{p.name}": inst.ports[p.name, a, b]
                    for p, a, b in product(
                        inst.cell.ports, range(inst.na), range(inst.nb)
                    )
                }
            )
        else:
            if hasattr(inst, "to_dtype"):  # always True - just making mypy happy
                inst = inst.to_dtype()
            array = None
            virtual = False
            transform = inst.dcplx_trans
            _all_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})

        net["instances"][inst_name] = _dump_instance(
            scm.Instance(
                component=component_namer(cast(kf.ProtoTKCell[Any], inst.cell)),
                array=array,
                settings=inst.cell.settings.model_dump(),
                info=inst.cell.info.model_dump(),
                virtual=virtual,
            ),
        )
        net["placements"][inst_name] = {
            "x": transform.disp.x,
            "y": transform.disp.y,
            "rotation": transform.angle,
            "mirror": transform.mirror,
        }

        if recursive and _has_instances(inst.cell):
            net["instances"][inst_name]["component"] = netlist_namer(
                cast(kf.ProtoTKCell[Any], inst.cell)
            )
            _insert_netlist(
                recnet,
                cast(kf.ProtoTKCell[Any], inst.cell),
                allow_multiple,
                instance_namer,
                component_namer,
                netlist_namer,
                port_matcher,
                recursive,
            )

    # Add top-level ports
    # we flip them so they face opposite to the instance ports
    # this is necessary as most PortMatchers will assume ports face each other
    for port in cell.ports:
        if port.name is not None:
            _all_ports[port.name] = _flip_port(cast(kf.DPort | kf.Port, port))

    all_nets = _get_nets(_all_ports, allow_multiple, port_matcher)
    net["ports"], net["nets"] = _split_nets_and_ports(all_nets)


def _has_instances(cell: Any) -> bool:
    """Return True if the cell has any instances."""
    return bool(getattr(cell, "insts", False)) or bool(getattr(cell, "vinsts", False))


def _get_array_config(inst: Instance) -> scm.Array:
    kcl = inst.cell.kcl
    match (
        abs(inst.a.x) == 0,
        abs(inst.a.y) == 0,
        abs(inst.b.x) == 0,
        abs(inst.b.y) == 0,
    ):
        case (_, True, True, _):
            return scm.OrthogonalGridArray(
                columns=inst.na,
                rows=inst.nb,
                column_pitch=kcl.dbu * inst.a.x,
                row_pitch=kcl.dbu * inst.b.y,
            )
        case (True, _, _, True):
            return scm.OrthogonalGridArray(
                columns=inst.nb,
                rows=inst.na,
                column_pitch=kcl.dbu * inst.b.x,
                row_pitch=kcl.dbu * inst.a.y,
            )
    return scm.GridArray(
        num_a=inst.na,
        num_b=inst.nb,
        pitch_a=(kcl.dbu * inst.a.x, kcl.dbu * inst.a.y),
        pitch_b=(kcl.dbu * inst.b.x, kcl.dbu * inst.b.y),
    )


def _is_array_inst(inst: Instance) -> bool:
    return getattr(inst, "na", 0) > 1 or getattr(inst, "nb", 0) > 1


def _is_pure_vinst(inst: Instance) -> bool:
    return isinstance(inst, kf.VInstance)


def _is_flattened_vinst(inst: Instance) -> bool:
    return getattr(inst.cell, "vtrans", None) is not None


def _dump_instance(
    instance: scm.Instance,
    instance_exclude: Iterable[str] = (),
) -> dict[str, Any]:
    instance_exclude = set(instance_exclude)
    dct = {}
    for k in sorted(scm.Instance.model_fields):
        v = getattr(instance, k)
        if (
            k in instance_exclude
            or (k == "array" and v is None)
            or (k == "virtual" and v is False)
        ):
            continue
        dct[k] = v
        if hasattr(dct[k], "model_dump"):
            dct[k] = dct[k].model_dump()
    return dct


def _short_component_name(
    cell: kf.ProtoTKCell[Any], component_namer: ComponentNamer
) -> str:
    """Get short component name, preferring function_name."""
    try:
        return cell.function_name or component_namer(cell)
    except ValueError:
        return component_namer(cell)


def _instname_from_compname(
    inst: Instance,
    compname: str,
    _instance_names: dict[str | None, str],
    _rev_instance_names: dict[str, str | None],
) -> str:
    if compname not in _rev_instance_names:
        _instance_names[inst.name] = compname
        _rev_instance_names[compname] = inst.name
        return compname

    # if the compname already ends on a number, we add an underscore.
    basename = re.sub("[0-9]*$", "", compname)
    if basename != compname:
        basename = f"{compname}_"

    index = 2
    while (compname := f"{basename}{index}") in _rev_instance_names:
        index += 1

    _instance_names[inst.name] = compname
    _rev_instance_names[compname] = inst.name
    return compname


def _clean_instname(name: str | None) -> str:
    if name is None:
        return f"unnamed_{secrets.token_hex(4)}"
    replace_map = {" ": "_", "!": "", "?": "", "#": "_", "%": "_", "(": "", ")": "", "*": "_", ",": "_", "-": "m", ".": "p", "/": "_", ":": "_", "=": "", "@": "_", "[": "", "]": "", "{": "", "}": "", "$": ""}  # fmt: skip
    for k, v in replace_map.items():
        name = name.replace(k, v)
    name = re.sub("[^a-zA-Z0-9]", "_", name)
    if name[0] in "0123456789":
        name = f"_{name}"
    return name


def _get_nets(
    all_ports: dict[str, kf.DPort | kf.Port],
    allow_multiple: bool,
    port_matcher: PortMatcher,
) -> list[dict[str, Any]]:
    """Extract connections between ports.

    NOTE: This is O(n²) in the number of ports. For very large circuits,
    consider adding a PortIndexer to bucket ports by position first (O(n)),
    then only compare within buckets using the PortMatcher.

    Returns:
        List of net dicts with keys 'p1', 'p2', and optionally 'settings'.
    """
    _port_names = list(all_ports)
    _num_ports = len(_port_names)

    # Track which ports are connected and their metadata
    # Key: sorted tuple of two port names, Value: settings dict or None
    _matched_pairs: dict[tuple[str, str], dict[str, Any] | None] = {}

    for i in range(_num_ports):
        pname = _port_names[i]
        p = all_ports[pname]
        for j in range(i + 1, _num_ports):
            qname = _port_names[j]
            q = all_ports[qname]
            result = port_matcher(p, q)
            if result is not False:
                pair = (pname, qname) if pname < qname else (qname, pname)
                if isinstance(result, dict):
                    _matched_pairs[pair] = result
                else:
                    _matched_pairs[pair] = None

    # Check for multiple connections at same location
    if not allow_multiple:
        # Group pairs by shared ports to detect multi-port overlaps
        port_connections: dict[str, set[str]] = defaultdict(set)
        for p1, p2 in _matched_pairs:
            port_connections[p1].add(p2)
            port_connections[p2].add(p1)
        for port, connected in port_connections.items():
            if len(connected) > 1:
                msg = (
                    f"More than two ports overlapping at {port}: {connected | {port}}."
                )
                raise ValueError(msg)

    # Build list of nets
    nets: list[dict[str, Any]] = []
    for (p1, p2), settings in _matched_pairs.items():
        net: dict[str, Any] = {"p1": p1, "p2": p2}
        if settings is not None:
            net["settings"] = settings
        nets.append(net)
    return nets


def _split_nets_and_ports(
    nets: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Split nets into top-level port mappings and instance-to-instance nets.

    Top-level ports never have a ',' in their name, whereas instance ports
    are formatted as '{instance_name},{port_name}'.

    Note: Settings from nets involving top-level ports are discarded since
    the ports dict only maps port names to instance ports.
    """
    ports: dict[str, str] = {}
    instance_nets: list[dict[str, Any]] = []
    for net in nets:
        p1, p2 = net["p1"], net["p2"]
        if "," not in p1:
            ports[p1] = p2
        elif "," not in p2:
            ports[p2] = p1
        else:
            instance_nets.append(net)
    return ports, instance_nets


def _sample_circuit() -> Component:
    c = Component()
    ring = c.add_ref(gf.c.ring_single(), name="ring").move((100, 0))
    mzi = c.add_ref_off_grid(gf.c.mzi()).rotate(33).move((0, 20))
    mzi.name = "mzi"
    s1 = c.add_ref_off_grid(gf.c.bend_euler_all_angle(angle=90 - 33)).connect(  # type: ignore[arg-type]
        "o1", mzi["o1"]
    )
    s1.name = "s1"
    s2 = c.add_ref_off_grid(gf.c.bend_euler_all_angle(angle=33)).connect(  # type: ignore[arg-type]
        "o2", mzi["o2"]
    )
    s2.name = "s2"
    gf.routing.route_bundle(c, [s1["o2"]], [ring["o1"]], cross_section="strip")
    arr = c.add_ref(gf.c.straight(), columns=1, rows=4, row_pitch=30).move((150, 0))
    arr.name = "arr"
    gf.routing.route_bundle(
        c,
        [arr["o1", 0, 0], arr["o1", 0, 3]],
        [ring["o2"], s2["o1"]],
        cross_section="strip",
    )
    gf.routing.route_bundle(
        c, [arr["o1", 0, 1]], [arr["o1", 0, 2]], cross_section="strip"
    )
    for i, p in enumerate(list(arr.ports)[1::2]):
        c.add_port(name=f"o{i + 1}", port=p)
    return c


def _width_mismatch_circuit() -> Component:
    """Create a simple circuit with a width mismatch for testing."""
    c = Component()
    s1 = c.add_ref(gf.c.straight(length=10, width=0.5), name="s1")
    s2 = c.add_ref(gf.c.straight(length=10, width=0.6), name="s2")
    s2.move((10, 0))  # Position s2 so its o1 aligns with s1's o2
    return c


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.gpdk import PDK

    PDK.activate()

    # Test sample circuit
    c = _sample_circuit()
    recnet = get_netlist_recursive(c, port_matcher=PortCenterMatcher())
    for name, netlist in recnet.items():
        print(f"Cell: {name}")
        for inst_name, inst in netlist["instances"].items():
            print(f"  Instance: {inst_name} -> {inst['component']}")
        print(f"  Ports: {netlist['ports']}")
        print(f"  Nets: {len(netlist['nets'])}")

    # Test width mismatch circuit
    print("\nWidth mismatch circuit:")
    c2 = _width_mismatch_circuit()
    netlist2 = get_netlist(c2, port_matcher=FlexiblePortMatcher())
    print(f"  Nets: {netlist2['nets']}")
