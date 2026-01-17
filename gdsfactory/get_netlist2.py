"""Extract netlist from component port connectivity."""

import re
import secrets
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain, combinations, product
from typing import Literal, TypeAlias

import kfactory as kf

import gdsfactory as gf
import gdsfactory.schematic as scm
from gdsfactory.component import Component

ComponentNameFrom: TypeAlias = Literal["function_name", "factory_name"]
InstanceNameFrom: TypeAlias = Literal[
    "instance_name", "component_name", "simple_instance_name"
]


def _insert_netlist(
    recnet: dict,
    cell: kf.KCell,
    instance_name_from: InstanceNameFrom,
    component_name_from: ComponentNameFrom,
    instance_exclude: Iterable[str],
    allow_multiple: bool = True,
) -> None:
    instance_exclude = set(instance_exclude)
    net = recnet[cell.name] = {
        "instances": {},
        "placements": {},
        "nets": {},
    }
    _instance_names = {}
    _rev_instance_names = {}
    _instance_ports = {}
    for inst in sorted(chain(cell.insts, cell.vinsts), key=lambda i: i.name):
        inst_name = _instname(
            inst,
            instance_name_from,
            component_name_from,
            _instance_names,
            _rev_instance_names,
        )
        if _is_pure_vinst(inst):
            if _is_array_inst(inst):
                msg = (
                    "Cannot export netlist: virtual array instances are not supported."
                )
                raise ValueError(msg)
            array = None
            virtual = True
            transform = inst.dcplx_trans
            _instance_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})
        elif _is_flattened_vinst(inst):
            if _is_array_inst(inst):
                msg = (
                    "Cannot export netlist: virtual array instances are not supported."
                )
            inst = inst.to_dtype()
            array = None
            virtual = True
            transform = inst.dcplx_trans * inst.cell.vtrans
            _instance_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})
        elif _is_array_inst(inst):
            inst = inst.to_dtype()
            array = _get_array_config(inst)
            virtual = False
            transform = inst.dcplx_trans
            _instance_ports.update(
                {
                    f"{inst_name}<{a}.{b}>,{p.name}": inst.ports[p.name, a, b]
                    for p, a, b in product(
                        inst.cell.ports, range(inst.na), range(inst.nb)
                    )
                }
            )
        else:
            inst = inst.to_dtype()
            array = None
            virtual = False
            transform = inst.dcplx_trans
            _instance_ports.update({f"{inst_name},{p.name}": p for p in inst.ports})

        net["instances"][inst_name] = _dump_instance(
            scm.Instance(
                component=_component_name(inst, component_name_from),
                array=array,
                settings=inst.cell.settings.model_dump(),
                info=inst.cell.info.model_dump(),
                virtual=virtual,
            ),
            instance_exclude,
        )
        net["placements"][inst_name] = {
            "x": transform.disp.x,
            "y": transform.disp.y,
            "rotation": transform.angle,
            "mirror": transform.mirror,
        }
    net["nets"] = _get_nets(_instance_ports, allow_multiple)
    return _instance_ports


def _get_array_config(inst: kf.Instance) -> scm.Array:
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


def _is_array_inst(inst: kf.Instance) -> bool:
    return getattr(inst, "na", 0) > 1 or getattr(inst, "nb", 0) > 1


def _is_pure_vinst(inst: kf.Instance) -> bool:
    return isinstance(inst, kf.VInstance)


def _is_flattened_vinst(inst: kf.Instance) -> bool:
    return inst.cell.vtrans is not None


def _sample_circuit() -> Component:
    c = Component()
    ring = c.add_ref(gf.c.ring_single(), name="ring").move((100, 0))
    mzi = c.add_ref_off_grid(gf.c.mzi()).rotate(33).move((0, 20))
    mzi.name = "mzi"
    s1 = c.add_ref_off_grid(gf.c.bend_euler_all_angle(angle=90 - 33)).connect(
        "o1", mzi["o1"]
    )
    s1.name = "s1"
    s2 = c.add_ref_off_grid(gf.c.bend_euler_all_angle(angle=33)).connect(
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


def _component_name(
    inst: kf.Instance,
    component_name_from: ComponentNameFrom,
) -> str:
    try:
        match component_name_from:
            case "factory_name":
                return inst.cell.factory_name
            case "function_name":
                return inst.cell.function_name or inst.cell.factory_name
            case _:
                msg = f"Invalid value for 'component_name_from'. Expected: 'factory_name' | 'function_name'. Got: {component_name_from!r}."
                raise TypeError(msg)
    except ValueError:
        return f"unknown_{secrets.token_hex(4)}"


def _short_component_name(
    inst: kf.Instance,
    component_name_from: ComponentNameFrom,
) -> str:
    return inst.cell.function_name or _component_name(inst, component_name_from)


def _dump_instance(instance: scm.Instance, instance_exclude: set[str]) -> dict:
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


def _instname(
    inst: kf.Instance,
    instance_name_from: InstanceNameFrom,
    component_name_from: ComponentNameFrom,
    _instance_names: dict[str, str],
    _rev_instance_names: dict[str, str],
) -> str:
    if inst.name in _instance_names:
        return _instance_names[inst.name]
    compname = _short_component_name(inst, component_name_from)
    match instance_name_from:
        case "instance_name":
            name = inst.name
        case "component_name":
            name = _instname_from_compname(
                inst, compname, _instance_names, _rev_instance_names
            )
        case "simple_instance_name":
            if inst.name.startswith(f"{compname}_"):
                name = _instname_from_compname(
                    inst, compname, _instance_names, _rev_instance_names
                )
            else:
                name = inst.name
        case _:
            msg = (
                "Invalid value for 'instance_name_from'. "
                "Expected: 'instance_name' | 'component_name' | 'short_instance_name'. "
                f"Got: {instance_name_from}."
            )
            raise ValueError(msg)
    name = _instance_names[inst.name] = _clean_instname(name)
    _rev_instance_names[name] = inst.name
    return name


def _instname_from_compname(
    inst: kf.Instance,
    compname: str,
    _instance_names: dict[str, str],
    _rev_instance_names: dict[str, str],
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


def _clean_instname(name: str) -> str:
    replace_map = {" ": "_", "!": "", "?": "", "#": "_", "%": "_", "(": "", ")": "", "*": "_", ",": "_", "-": "m", ".": "p", "/": "_", ":": "_", "=": "", "@": "_", "[": "", "]": "", "{": "", "}": "", "$": ""}  # fmt: skip
    for k, v in replace_map.items():
        name = name.replace(k, v)
    name = re.sub("[^a-zA-Z0-9]", "_", name)
    if name[0] in "0123456789":
        name = f"_{name}"
    return name


def _get_nets(
    instance_ports: dict[str, kf.DPort | kf.Port],
    allow_multiple: bool,
) -> list[dict[str, str]]:
    _instance_port_names = list(instance_ports)
    _num_instance_ports = len(_instance_port_names)
    _nets = defaultdict(set)
    for i in range(_num_instance_ports):
        pname = _instance_port_names[i]
        for j in range(i + 1, _num_instance_ports):
            qname = _instance_port_names[j]
            p = instance_ports[pname]
            q = instance_ports[qname]
            if _ports_equal(p, q):
                _nets[i].add(pname)
                _nets[i].add(qname)
    nets = []
    for net in _nets.values():
        if len(net) > 2 and not allow_multiple:
            msg = f"More than two ports overlapping: {net}."
            raise ValueError(msg)
        for p1, p2 in combinations(net, 2):
            nets.append({"p1": p1, "p2": p2})
    return nets


def _ports_equal(port1: kf.DPort | kf.Port, port2: kf.DPort | kf.Port) -> bool:
    tolerance_dbu = 2
    x1, y1 = port1.to_itype().center
    x2, y2 = port2.to_itype().center
    return abs(x2 - x1) < tolerance_dbu and abs(y2 - y1) < tolerance_dbu


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.gpdk import PDK

    PDK.activate()
    c = _sample_circuit()
    cell = kf.KCell(base=c.base)
    recnet: dict = {}
    _insert_netlist(recnet, cell, "simple_instance_name", "function_name", ())
    netlist = recnet[next(iter(recnet))]
    netlist["placements"]["mzi"]
    for net in netlist["nets"]:
        print(net["p1"], net["p2"])
    c2 = gf.read.from_yaml(netlist)
    c2.show()
