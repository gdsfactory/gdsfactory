"""Extract netlist from component port connectivity."""

import inspect
import re
import secrets
import warnings
from collections import defaultdict
from collections.abc import Iterable
from hashlib import md5
from itertools import chain, product
from typing import Any, Literal, Protocol, TypeAlias, cast

import kfactory as kf
from natsort import natsorted

import gdsfactory as gf
import gdsfactory.schematic as scm
from gdsfactory.component import Component
from gdsfactory.name import get_instance_name_from_alias as legacy_namer  # noqa: F401
from gdsfactory.serialization import clean_value_json

Instance: TypeAlias = kf.DInstance | kf.VInstance | kf.Instance
ErrorBehavior: TypeAlias = Literal["ignore", "warn", "error"]


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
        inst_name = _instname(inst)
        if inst_name in self._instance_names:
            return self._instance_names[inst_name]
        name = self._instance_names[inst_name] = _clean_instname(inst_name)
        self._rev_instance_names[name] = inst_name
        return name


class CountedNamer:
    """Names instances using their component name with numeric suffixes."""

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._instance_names: dict[str | None, str] = {}
        self._rev_instance_names: dict[str, str | None] = {}

    def __call__(self, inst: Instance) -> str:
        inst_name = _instname(inst)
        if inst_name in self._instance_names:
            return self._instance_names[inst_name]
        compname = _short_component_name(_instcell(inst), self._component_namer)
        name = _instname_from_compname(
            inst, compname, self._instance_names, self._rev_instance_names
        )
        name = self._instance_names[inst_name] = _clean_instname(name)
        self._rev_instance_names[name] = inst_name
        return name


class SmartNamer:
    """Names instances using component name if auto-generated, otherwise instance name."""

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._instance_names: dict[str | None, str] = {}
        self._rev_instance_names: dict[str, str | None] = {}

    def __call__(self, inst: Instance) -> str:
        inst_name = _instname(inst)
        if inst_name in self._instance_names:
            return self._instance_names[inst_name]
        compname = _short_component_name(_instcell(inst), self._component_namer)
        if inst_name is not None and inst_name.startswith(f"{compname}_"):
            name: str | None = _instname_from_compname(
                inst, compname, self._instance_names, self._rev_instance_names
            )
        else:
            name = inst_name
        cleaned = self._instance_names[inst_name] = _clean_instname(name)
        self._rev_instance_names[cleaned] = inst_name
        return cleaned


class NetlistNamer(Protocol):
    """Protocol for naming cells in recursive netlists."""

    def __call__(self, cell: kf.ProtoTKCell[Any]) -> str:
        """Return the name for the given cell in the netlist."""
        ...


class CountedNetlistNamer:
    """Names cells with counting for uniqueness in recursive netlists.

    Uses component_namer for leaf cells (no instances). For hierarchical cells,
    uses the component name as-is if there are no settings, otherwise uses
    counted naming to distinguish different parameterizations.
    """

    def __init__(self, component_namer: ComponentNamer) -> None:
        self._component_namer = component_namer
        self._cell_names: dict[str, str] = {}  # cell.name -> assigned name
        self._used_names: set[str] = set()

    def __call__(self, cell: kf.ProtoTKCell[Any]) -> str:
        if cell.name in self._cell_names:
            return self._cell_names[cell.name]

        base = self._component_namer(cell)

        if not _has_instances(cell):
            # Leaf cell: use component_namer (typically function_name)
            name = base
        elif not _has_non_default_settings(cell):
            # Hierarchical cell with default settings: use component name as-is
            name = base
        else:
            # Hierarchical cell with non-default settings: use counted naming
            name = self._get_unique_name(base)

        self._cell_names[cell.name] = name
        self._used_names.add(name)
        return name

    def _get_unique_name(self, base: str) -> str:
        # If name is not already used, use it as-is
        if base not in self._used_names:
            return base
        # Otherwise add a numbered suffix
        # If base ends with a number, add underscore
        basename = re.sub("[0-9]*$", "", base)
        if basename != base:
            basename = f"{base}_"
        index = 2
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
    on_multi_connect: ErrorBehavior = "error",
    on_dangling_port: ErrorBehavior = "warn",
    instance_namer: InstanceNamer | None = None,
    component_namer: ComponentNamer = function_namer,
    port_matcher: PortMatcher | None = None,
) -> dict[str, Any]:
    """Extract netlist from a cell's port connectivity.

    Args:
        cell: The cell to extract the netlist from.
        on_multi_connect: What to do when more than two ports overlap.
            "ignore": silently allow, "warn": allow with warning, "error": raise.
        on_dangling_port: What to do when an instance port is not connected.
            "ignore": silently allow, "warn": allow with warning, "error": raise.
        instance_namer: Callable to name instances.
            Defaults to SmartNamer(component_namer).
        component_namer: Callable to name components.
            Defaults to function_namer.
        port_matcher: Callable to determine if two ports are connected.
            Defaults to SmartPortMatcher().

    Returns:
        A dictionary containing instances, placements, ports, and nets.
    """
    recnet: dict[str, dict[str, Any]] = {}
    _insert_netlist(
        recnet,
        cell,
        on_multi_connect,
        on_dangling_port,
        instance_namer or SmartNamer(component_namer),
        component_namer,
        component_namer,  # netlist_namer: use component_namer for non-recursive
        port_matcher or _default_port_matcher,
        recursive=False,
    )
    return cast(dict[str, Any], clean_value_json(recnet[next(iter(recnet))]))


def get_netlist_recursive(
    cell: kf.ProtoTKCell[Any],
    *,
    on_multi_connect: ErrorBehavior = "error",
    on_dangling_port: ErrorBehavior = "warn",
    instance_namer: InstanceNamer | None = None,
    component_namer: ComponentNamer = function_namer,
    netlist_namer: NetlistNamer | None = None,
    port_matcher: PortMatcher | None = None,
) -> dict[str, Any]:
    """Extract netlists recursively from a cell and all its subcells.

    Args:
        cell: The cell to extract the netlist from.
        on_multi_connect: What to do when more than two ports overlap.
            "ignore": silently allow, "warn": allow with warning, "error": raise.
        on_dangling_port: What to do when an instance port is not connected.
            "ignore": silently allow, "warn": allow with warning, "error": raise.
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
        on_multi_connect,
        on_dangling_port,
        instance_namer or SmartNamer(component_namer),
        component_namer,
        netlist_namer or CountedNetlistNamer(component_namer),
        port_matcher or _default_port_matcher,
        recursive=True,
    )
    return cast(dict[str, dict[str, Any]], clean_value_json(recnet))


def _insert_netlist(
    recnet: dict[str, Any],
    cell: kf.ProtoTKCell[Any],
    on_multi_connect: ErrorBehavior,
    on_dangling_port: ErrorBehavior,
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
        inst_cell = _instcell(inst)
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
            transform = inst.dcplx_trans * inst.cell.vtrans  # type: ignore[union-attr]
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
                        inst_cell.ports, range(inst.na), range(inst.nb)
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
            {
                "component": component_namer(inst_cell),
                "array": array,
                "settings": inst_cell.settings.model_dump(),
                "info": inst_cell.info.model_dump(),
                "virtual": virtual,
            }
        )
        net["placements"][inst_name] = {
            "x": transform.disp.x,
            "y": transform.disp.y,
            "rotation": transform.angle,
            "mirror": transform.mirror,
        }

        if recursive and _has_instances(inst_cell):
            net["instances"][inst_name]["component"] = netlist_namer(inst_cell)
            _insert_netlist(
                recnet,
                inst_cell,
                on_multi_connect,
                on_dangling_port,
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

    all_nets = _get_nets(_all_ports, on_multi_connect, port_matcher)
    _handle_dangling_ports(_all_ports, all_nets, on_dangling_port)
    net["ports"], net["nets"] = _split_nets_and_ports(all_nets)


def _has_instances(cell: Any) -> bool:
    """Return True if the cell has any instances."""
    return bool(getattr(cell, "insts", False)) or bool(getattr(cell, "vinsts", False))


def _has_non_default_settings(cell: kf.ProtoTKCell[Any]) -> bool:
    """Return True if the cell has settings that differ from factory defaults."""
    settings = cell.settings.model_dump()
    if not settings:
        return False

    # Try to get the factory function to compare defaults
    try:
        factory_name = cell.function_name or cell.factory_name
    except ValueError:
        # No factory name, assume settings are non-default if present
        return bool(settings)

    # Get factory from active PDK
    pdk = gf.get_active_pdk()
    factory = pdk.cells.get(factory_name)
    if factory is None:
        # Factory not found, assume settings are non-default if present
        return bool(settings)

    # Get default parameter values from factory signature
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return bool(settings)

    defaults = {}
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default

    # Check if any setting differs from default
    for key, value in settings.items():
        if value is None:
            continue
        if key not in defaults:
            # Setting not in factory signature, consider it non-default
            return True
        if value != defaults[key]:
            return True

    return False


def _get_array_config(inst: Instance) -> scm.Array:
    kcl = inst.cell.kcl
    # inst.a and inst.b have the instance rotation baked in (but not mirror).
    # The netlist stores the array pitches in the local (pre-rotation) frame,
    # so we need to undo the rotation to get the original pitch vectors.
    trans = inst.dcplx_trans
    inv_rot = kf.kdb.DCplxTrans(1, trans.angle, False, 0, 0).inverted()
    a = inv_rot * kf.kdb.DVector(inst.a.x, inst.a.y)
    b = inv_rot * kf.kdb.DVector(inst.b.x, inst.b.y)
    ax = round(kcl.dbu * a.x, 6)
    ay = round(kcl.dbu * a.y, 6)
    bx = round(kcl.dbu * b.x, 6)
    by = round(kcl.dbu * b.y, 6)
    match (
        ax == 0,
        ay == 0,
        bx == 0,
        by == 0,
    ):
        case (_, True, True, _):
            return scm.OrthogonalGridArray(
                columns=inst.na,
                rows=inst.nb,
                column_pitch=ax,
                row_pitch=by,
            )
        case (True, _, _, True):
            return scm.OrthogonalGridArray(
                columns=inst.nb,
                rows=inst.na,
                column_pitch=bx,
                row_pitch=ay,
            )
    return scm.GridArray(
        num_a=inst.na,
        num_b=inst.nb,
        pitch_a=(ax, ay),
        pitch_b=(bx, by),
    )


def _is_array_inst(inst: Instance) -> bool:
    return getattr(inst, "na", 0) > 1 or getattr(inst, "nb", 0) > 1


def _is_pure_vinst(inst: Instance) -> bool:
    return isinstance(inst, kf.VInstance)


def _is_flattened_vinst(inst: Instance) -> bool:
    return getattr(inst.cell, "vtrans", None) is not None


def _dump_instance(
    instance: dict[str, Any],
    instance_exclude: Iterable[str] = (),
) -> dict[str, Any]:
    instance_exclude = set(instance_exclude)
    dct = {}
    for k, v in instance.items():
        if (
            k in instance_exclude
            or (k == "array" and v is None)
            or (k == "virtual" and v is False)
        ):
            continue
        if hasattr(v, "model_dump"):
            dct[k] = v.model_dump()
        else:
            dct[k] = v
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
    inst_name = _instname(inst)
    if compname not in _rev_instance_names:
        _instance_names[inst_name] = compname
        _rev_instance_names[compname] = inst_name
        return compname

    # if the compname already ends on a number, we add an underscore.
    basename = re.sub("[0-9]*$", "", compname)
    if basename != compname:
        basename = f"{compname}_"

    index = 2
    while (compname := f"{basename}{index}") in _rev_instance_names:
        index += 1

    _instance_names[inst_name] = compname
    _rev_instance_names[compname] = inst_name
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


def _handle_multi_connect(
    matched_pairs: dict[tuple[str, str], dict[str, Any] | None],
    on_multi_connect: ErrorBehavior,
) -> None:
    """Check for multiple connections at same location and warn/error as configured."""
    if on_multi_connect == "ignore":
        return

    # Group pairs by shared ports to detect multi-port overlaps
    port_connections: dict[str, set[str]] = defaultdict(set)
    for p1, p2 in matched_pairs:
        port_connections[p1].add(p2)
        port_connections[p2].add(p1)

    for port, connected in port_connections.items():
        if len(connected) > 1:
            msg = f"More than two ports overlapping at {port}: {connected | {port}}."
            if on_multi_connect == "error":
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=5)


def _get_nets(
    all_ports: dict[str, kf.DPort | kf.Port],
    on_multi_connect: ErrorBehavior,
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

    _handle_multi_connect(_matched_pairs, on_multi_connect)

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
            p1 = net["p1"]
            p2 = net["p2"]
            meta = {k: v for k, v in net.items() if k != "p1" and k != "p2"}
            if p1 < p2:
                instance_nets.append({"p1": p1, "p2": p2, **meta})
            else:
                instance_nets.append({"p1": p2, "p2": p1, **meta})
    ports = {k: ports[k] for k in natsorted(ports)}
    nets = natsorted(instance_nets, key=lambda n: f"{n['p1']}:{n['p2']}")
    return ports, nets


def _handle_dangling_ports(
    all_ports: dict[str, kf.DPort | kf.Port],
    nets: list[dict[str, Any]],
    on_dangling_port: ErrorBehavior,
) -> None:
    """Check for unconnected instance ports and warn/error as configured.

    Top-level ports (without ',' in name) are not considered dangling.
    """
    if on_dangling_port == "ignore":
        return

    # Collect all connected ports
    connected: set[str] = set()
    for net in nets:
        connected.add(net["p1"])
        connected.add(net["p2"])

    # Find dangling instance ports (exclude top-level ports)
    dangling = [p for p in all_ports if "," in p and p not in connected]

    if dangling:
        msg = f"Unconnected ports: {dangling}"
        if on_dangling_port == "error":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=4)


@gf.cell
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
    s1 = c.add_ref(gf.c.straight(length=10, width=0.5), name="s1")  # noqa: F841
    s2 = c.add_ref(gf.c.straight(length=10, width=0.6), name="s2")
    s2.move((10, 0))  # Position s2 so its o1 aligns with s1's o2
    return c


def _instname(inst: Instance) -> str:
    cell = _instcell(inst)
    h = md5(
        repr(
            (
                cell.name,
                (
                    inst.trans.disp.x,
                    inst.trans.disp.y,
                    inst.trans.angle,
                    inst.trans.mirror,
                ),
                inst.a,
                inst.b,
                inst.na,
                inst.nb,
            )
        ).encode()
    ).hexdigest()[:8]
    return inst.name or f"{cell.name}__{h}"


def _instcell(inst: Instance) -> kf.ProtoTKCell[Any]:
    if _is_flattened_vinst(inst):
        inst_cell = gf.get_component(
            str(inst.cell.function_name or inst.cell.factory_name),
            **inst.cell.settings.model_dump(),
        )
        return cast(kf.ProtoTKCell[Any], inst_cell)
    return cast(kf.ProtoTKCell[Any], inst.cell)


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.gpdk import PDK

    PDK.activate()

    # Test sample circuit
    c = _sample_circuit()
    netlist = c.get_netlist()
    # gf.clear_cache()
    c2 = gf.read.from_yaml(netlist)
    c2.show()
