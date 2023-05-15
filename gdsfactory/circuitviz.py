from collections import defaultdict
from functools import partial
from typing import Dict, List, NamedTuple, Union

import bokeh.events as be
import numpy as np
import pandas as pd
import yaml
from bokeh import io as bio
from bokeh import models as bm
from bokeh import plotting as bp
from natsort import natsorted

import gdsfactory as gf
from gdsfactory.picmodel import PicYamlConfiguration, Placement, SchematicConfiguration

data = {
    "srcs": defaultdict(lambda: defaultdict(lambda: [])),
    "dss": {},
}

COLORS_BY_PORT_TYPE = {
    "optical": "#0000ff",
    "electrical": "#00ff00",
    "placement": "white",
    None: "gray",  # default
}


def save_netlist(netlist, filename):
    with open(filename, mode="w") as f:
        d = netlist.dict(exclude_none=True)
        if "placements" in d:
            placements_dict = d["placements"]
        elif "schematic_placements" in d:
            placements_dict = d["schematic_placements"]
        else:
            raise ValueError("No placements attribute found in netlist")
        pkeys = list(placements_dict.keys())
        p: dict = placements_dict
        for pk in pkeys:
            pv = p[pk]
            if pv:
                for kk in ["x", "y"]:
                    if kk in pv:
                        try:
                            pv[kk] = float(pv[kk])
                        except Exception:
                            pass
            else:
                p.pop(pk)
        yaml.dump(d, f, sort_keys=False, default_flow_style=None)


class Rect(NamedTuple):
    tag: str
    x: float
    y: float
    w: float
    h: float
    c: str


class LayerPolygons(NamedTuple):
    tag: str
    xs: List[List[List[float]]]
    ys: List[List[List[float]]]
    c: str
    alpha: float


class LineSegment(NamedTuple):
    tag: str
    x0: float
    y0: float
    x1: float
    y1: float
    name: str


def _enlarge_limits(ax, x, y, w=0.0, h=0.0):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_min, x_max = xlim if xlim != (0.0, 1.0) else (np.inf, -np.inf)
    y_min, y_max = ylim if ylim != (0.0, 1.0) else (np.inf, -np.inf)
    x_min = min(x_min, x)
    x_max = max(x + w, x_max)
    y_min = min(y_min, y)
    y_max = max(y + h, y_max)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _get_sources(objs):
    srcs = defaultdict(lambda: defaultdict(lambda: []))
    for obj in objs:
        if isinstance(obj, LineSegment):
            src = srcs["MultiLine"]
            src["tag"].append(obj.tag)
            src["x"].append(np.array([obj.x0, obj.x1]))
            src["y"].append(np.array([obj.y0, obj.y1]))
            src["line_color"].append("#000000")
            src["name"].append(obj.name)
        elif isinstance(obj, Rect):
            src = srcs["Rect"]
            src["tag"].append(obj.tag)
            src["x"].append(obj.x + obj.w / 2)
            src["y"].append(obj.y + obj.h / 2)
            src["width"].append(obj.w)
            src["height"].append(obj.h)
            src["fill_color"].append(obj.c)
            src["fill_alpha"].append(0.1)
        elif isinstance(obj, gf.Port):
            src = srcs["Port"]
            src["tag"].append(obj.tag)
            src["x"].append(obj.x)
            src["y"].append(obj.y)
            # src["width"].append(obj.w)
            # src["height"].append(obj.h)
            color = COLORS_BY_PORT_TYPE.get(obj.port_type, COLORS_BY_PORT_TYPE[None])
            src["fill_color"].append(color)
            src["fill_alpha"].append(0.5)
        elif isinstance(obj, LayerPolygons):
            src = srcs["Polygons"]
            src["tag"].append(obj.tag)
            src["xs"].append(obj.xs)
            src["ys"].append(obj.ys)
            src["fill_color"].append(obj.c)
            src["fill_alpha"].append(obj.alpha)
    return srcs


def _get_column_data_sources(srcs):
    _srcs = {}
    for k, src in srcs.items():
        ds = bm.ColumnDataSource(dict(src.items()))
        _srcs[k] = ds
    return _srcs


def viz_bk(
    netlist: Union[SchematicConfiguration, PicYamlConfiguration],
    instances,
    netlist_filename,
    fig=None,
    **kwargs,
):
    global data
    if fig is None:
        fig = bp.figure()

    if isinstance(netlist, (PicYamlConfiguration, SchematicConfiguration)):
        objs = viz_netlist(netlist, instances, **kwargs)
    else:
        objs = netlist
        netlist = None

    if not isinstance(objs, list):
        raise ValueError("viz_bk can only visualize a list of objects.")

    # data['srcs'] = _viz_bk_srcs(objs)

    srcs = _get_sources(objs)
    dss = data["dss"] = _get_column_data_sources(srcs)
    netlist = data["netlist"]

    def cb_rect_on_change_data(attr, old, new):
        tags = np.array(old["tag"], dtype=object)
        xy_old = np.stack([old["x"], old["y"]], 1)
        xy_new = np.stack([new["x"], new["y"]], 1)
        if xy_old.shape != xy_new.shape:
            dss["Rect"].data.__dict__.update(old)
            return
        dxs, dys = (xy_new - xy_old).T  # type: ignore
        idxs = np.where(dxs**2 + dys**2 > 1.0)[0]
        tags = tags[idxs]
        dxs = dxs[idxs]
        dys = dys[idxs]

        for tag, dx, dy in zip(tags, dxs, dys):  # loop over all displaced rectangles
            if netlist is not None:
                if tag not in netlist.placements:
                    netlist.placements[tag] = Placement(x=0, y=0, dx=0, dy=0)
                dx_, dy_ = netlist.placements[tag].dx, netlist.placements[tag].dy
                dx_ = 0.0 if dx_ is None else dx_
                dy_ = 0.0 if dy_ is None else dy_
                netlist.placements[tag].dx = float(dx_ + dx)
                netlist.placements[tag].dy = float(dy_ + dy)
            for k, v in dss.items():
                if k == "Rect":
                    continue
                elif k == "MultiLine":
                    data = dict(v.data)
                    for i, tag_ in enumerate(data["tag"]):
                        if "," in tag_:
                            tag1, tag2 = tag_.split(",")
                            if tag == tag1:
                                data["x"][i][0] += dx
                                data["y"][i][0] += dy
                            elif tag == tag2:
                                data["x"][i][-1] += dx
                                data["y"][i][-1] += dy
                            else:
                                continue
                        else:
                            if tag_ == tag:
                                data["x"][i][:] += dx
                                data["y"][i][:] += dy
                            else:
                                continue
                    v.data = data
                elif k == "Polygons":
                    data = dict(v.data)
                    for i, tag_ in enumerate(data["tag"]):
                        if tag_ == tag:
                            for i_poly in range(len(data["xs"][i])):
                                for i_boundary in range(len(data["xs"][i][i_poly])):
                                    data["xs"][i][i_poly][i_boundary] += dx
                                    data["ys"][i][i_poly][i_boundary] += dy
                    v.data = data
                elif k == "Port":
                    data = dict(v.data)
                    for i, tag_ in enumerate(data["tag"]):
                        if tag_ == tag:
                            data["x"][i] += dx
                            data["y"][i] += dy
                        else:
                            continue
                    v.data = data
        save_netlist(netlist, netlist_filename)

    def cb_rect_selected_on_change_indices(attr, old, new):
        if len(new) > 1:
            data["dss"]["Rect"].selected.indices = [new[0]]

    def cp_double_tap(event):
        # only works on 'hierarchical netlists...'
        if netlist is None:
            return
        df = pd.DataFrame(data["dss"]["Rect"].data)
        mask = np.ones_like(df.x, dtype=bool)
        mask &= df.x - df.width / 2 < event.x
        mask &= event.x < df.x + df.width / 2
        mask &= df.y - df.height / 2 < event.y
        mask &= event.y < df.y + df.width / 2
        df = df[mask]

        tags = df.tag.values
        if tags.shape[0] != 1:
            return

        tag = tags[0]
        if tag in netlist.placements:
            cur_rotation = netlist.placements[tag].rotation or 0
            netlist.placements[tag].rotation = (cur_rotation + 90) % 360
        else:
            return

        update_schematic_plot(schematic=netlist, instances=instances)

    data["dss"]["Rect"].on_change("data", cb_rect_on_change_data)
    data["dss"]["Rect"].selected.on_change(
        "indices", cb_rect_selected_on_change_indices
    )
    fig.on_event(be.DoubleTap, cp_double_tap)

    fig.add_glyph(
        data["dss"]["Rect"],
        bm.Rect(
            x="x",
            y="y",
            width="width",
            height="height",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
        ),
        name="instances",
    )
    if "Polygons" in data["dss"]:
        fig.add_glyph(
            data["dss"]["Polygons"],
            bm.MultiPolygons(
                xs="xs", ys="ys", fill_color="fill_color", fill_alpha="fill_alpha"
            ),
        )
    if "MultiLine" in data["dss"]:
        fig.add_glyph(
            data["dss"]["MultiLine"],
            bm.MultiLine(xs="x", ys="y"),
            name="nets",
        )  # , line_color="line_color"))
    fig.add_glyph(
        data["dss"]["Port"], glyph=bm.Circle(x="x", y="y", fill_color="fill_color")
    )
    del fig.tools[:]
    draw_tool = bm.PointDrawTool(
        renderers=[r for r in fig.renderers if isinstance(r.glyph, bm.Rect)],
        empty_value="black",
    )
    hover_tool = bm.HoverTool(
        names=["instances"],
        tooltips=[("Instance", "@tag")],
    )
    hover_tool_nets = bm.HoverTool(
        names=["nets"],
        tooltips=[("Net", "@name")],
        show_arrow=True,
        line_policy="interp",
    )
    # pan_tool = bm.PanTool()
    tap_tool = bm.TapTool()
    zoom = bm.WheelZoomTool()
    fig.add_tools(draw_tool, hover_tool, hover_tool_nets, tap_tool, zoom)
    fig.toolbar.active_scroll = zoom
    fig.toolbar.active_tap = tap_tool
    fig.toolbar.active_drag = draw_tool
    fig.toolbar.logo = None
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.yaxis.major_label_text_font_size = "0pt"
    fig.match_aspect = True

    def bkapp(doc):
        doc.add_root(fig)
        data["doc"] = doc

    return bkapp


def get_ports(component):
    comp = component
    return natsorted(comp.ports.keys())


def is_output_port(port):
    if "," in port:
        return is_output_port(port.split(",")[-1])
    return port.startswith("out")


def is_input_port(port):
    return not is_output_port(port)


def get_input_ports(component):
    ports = get_ports(component)
    return [p for p in ports if is_input_port(p)]


def get_output_ports(component):
    ports = get_ports(component)
    return [p for p in ports if is_output_port(p)]


def ports_ys(ports, instance_height):
    h = instance_height
    if len(ports) < 1:
        return [h / 2]
    ys = np.linspace(0, h, len(ports) + 1)
    dy = ys[1] - ys[0]
    return ys[1:] - dy / 2


def viz_instance(
    netlist: Union[PicYamlConfiguration, SchematicConfiguration],
    instance_name,
    component,
    instance_size,
):
    # inst_spec = netlist.instances[instance_name].dict()
    inst_ref = component.named_references[instance_name]
    bbox = inst_ref.bbox
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    x0 = bbox[0][0]
    y0 = bbox[0][1]
    # pl = w / 10
    # input_ports = get_input_ports(component)
    # output_ports = get_output_ports(component)
    # y_inputs = ports_ys(input_ports, h)
    # y_outputs = ports_ys(output_ports, h)
    # x, y = get_placements(netlist).get(instance_name, (0, 0))
    x, y = x0, y0
    polys_by_layer = inst_ref.get_polygons(by_spec=True, as_array=False)
    layer_polys = []
    layer_views = gf.pdk.get_layer_views()

    for layer, polys in polys_by_layer.items():
        if layer not in layer_views.get_layer_tuples():
            print(f"layer {layer} not found")
            continue
        lv = layer_views.get_from_tuple(layer)
        if lv:
            xs = [[p.points[:, 0]] for p in polys]
            ys = [[p.points[:, 1]] for p in polys]
            lp = LayerPolygons(
                tag=instance_name,
                xs=xs,
                ys=ys,
                c=lv.get_color_dict()["fill_color"],
                alpha=lv.get_alpha(),
            )
            layer_polys.append(lp)

    ports: List[gf.Port] = inst_ref.ports.values()
    ports = [p.copy() for p in ports]
    for p in ports:
        # p.move((x, y))
        p.tag = instance_name
    c = "#000000"

    r = Rect(tag=instance_name, x=x, y=y, w=w, h=h, c=c)
    return [r, *ports] + layer_polys


def split_port(port, netlist):
    if "," not in port:
        port = netlist.ports[port]
    *instance_name, port = port.split(",")
    return ",".join(instance_name), port


def viz_connection(netlist, p_in, p_out, instance_size, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    tag = f"{p_in.split(',')[0]},{p_out.split(',')[0]}"
    name = f"{p_in} ➔ {p_out}"
    line = LineSegment(tag, x1, y1, x2, y2, name=name)
    return [line]


def viz_netlist(netlist, instances, instance_size=20):
    schematic_dict = netlist.dict()
    schematic_as_layout = {
        "instances": schematic_dict["instances"],
        "placements": schematic_dict["schematic_placements"],
    }
    schematic_component = gf.read.from_yaml(schematic_as_layout, mode="schematic")

    els = []
    port_coords = {}
    for instance_name in netlist.instances:
        els += viz_instance(netlist, instance_name, schematic_component, instance_size)
        for el in els:
            if isinstance(el, gf.Port):
                port_name = f"{instance_name},{el.name}"
                port_coords[port_name] = el.center

    for net in netlist.nets:
        p_in, p_out = net
        point1 = port_coords[p_in]
        point2 = port_coords[p_out]
        els += viz_connection(netlist, p_in, p_out, instance_size, point1, point2)
    return els


def show_netlist(schematic: SchematicConfiguration, instances: Dict, netlist_filename):
    global data
    data["netlist"] = schematic
    fig = bp.figure(width=800, height=500)
    app = viz_bk(
        schematic,
        instances=instances,
        fig=fig,
        instance_size=50,
        netlist_filename=netlist_filename,
    )
    bio.show(app)


def update_schematic_plot(
    schematic: SchematicConfiguration, instances: Dict, *args, **kwargs
):
    global data

    if "doc" in data:
        doc = data["doc"]
        doc.add_next_tick_callback(
            partial(
                _update_schematic_plot,
                schematic=schematic,
                instances=instances,
            )
        )


def _update_schematic_plot(
    schematic: SchematicConfiguration, instances: Dict, *args, **kwargs
):
    srcs = _get_sources(viz_netlist(schematic, instances=instances))
    for k in srcs:
        data["dss"][k].data = srcs[k]


def add_instance(name: str, component):
    inst_viz = viz_instance(
        data["netlist"], instance_name=name, component=component, instance_size=0
    )
    srcs = _get_sources([inst_viz])
    for k, src in srcs.items():
        cds: bm.ColumnDataSource = data["dss"][k]
        cds.stream(src)


def get_deltas(netlist):
    return {
        k: {"dx": p.dx or 0, "dy": p.dy or 0} for k, p in netlist.placements.items()
    }


def apply_deltas(netlist, deltas):
    for k, d in deltas.items():
        netlist.placements[k].dx = d["dx"]
        netlist.placements[k].dy = d["dy"]
