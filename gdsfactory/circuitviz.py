###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2021 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################

# hide
import pathlib
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional

import bokeh.events as be
import numpy as np
import pandas as pd
import yaml
from bokeh import io as bio
from bokeh import models as bm
from bokeh import plotting as bp
from natsort import natsorted

import gdsfactory as gf

from .picmodel import PicYamlConfiguration, Placement

data = {
    "srcs": defaultdict(lambda: defaultdict(lambda: [])),
    "dss": {},
}


NETLIST_FILENAME = pathlib.Path("interactive_movement.pic.yml")

# netlist = create_pic()
# # netlist.move_instance('i3', 150, 200)


def save_netlist(netlist, filename):
    with open(filename, mode="w") as f:
        d = netlist.dict(exclude_none=True)
        pkeys = list(d["placements"].keys())
        p: dict = d["placements"]
        for pk in pkeys:
            pv = p[pk]
            if pv:
                for kk in ["x", "y"]:
                    if kk in pv:
                        try:
                            pv[kk] = float(pv[kk])
                        except:
                            pass
            else:
                p.pop(pk)
        yaml.dump(d, f)


# save_netlist(netlist, NETLIST_FILENAME)

# export
class Rect(NamedTuple):
    tag: str
    x: float
    y: float
    w: float
    h: float
    c: str


# export
class LineSegment(NamedTuple):
    tag: str
    x0: float
    y0: float
    x1: float
    y1: float


# exporti
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


# exporti
def _get_sources(objs):
    srcs = defaultdict(lambda: defaultdict(lambda: []))
    for obj in objs:
        if isinstance(obj, LineSegment):
            src = srcs["MultiLine"]
            src["tag"].append(obj.tag)
            src["x"].append(np.array([obj.x0, obj.x1]))
            src["y"].append(np.array([obj.y0, obj.y1]))
            src["line_color"].append("#000000")
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
            src["fill_color"].append("green")
            src["fill_alpha"].append(0.5)
    return srcs


def _get_column_data_sources(srcs):
    _srcs = {}
    for k, src in srcs.items():
        ds = bm.ColumnDataSource({kk: v for kk, v in src.items()})
        _srcs[k] = ds
    return _srcs


# export
def viz_bk(netlist, instances, fig=None, **kwargs):
    global data
    if fig is None:
        fig = bp.figure()

    if isinstance(netlist, PicYamlConfiguration):
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
                elif k == "Port":
                    data = dict(v.data)
                    for i, tag_ in enumerate(data["tag"]):
                        if tag_ == tag:
                            data["x"][i] += dx
                            data["y"][i] += dy
                        else:
                            continue
                    v.data = data
        save_netlist(netlist, NETLIST_FILENAME)

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
        if tag not in netlist.instances:
            return

        instance = netlist.instances[tag]
        if "component" in instance:
            return

        del fig.renderers[:]
        return viz_bk(instance, fig=fig, **kwargs)

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
    )
    if "MultiLine" in data["dss"]:
        fig.add_glyph(
            data["dss"]["MultiLine"], bm.MultiLine(xs="x", ys="y")
        )  # , line_color="line_color"))
    fig.add_glyph(
        data["dss"]["Port"], glyph=bm.Circle(x="x", y="y", fill_color="fill_color")
    )
    del fig.tools[:]
    fig.add_tools(
        draw_tool := bm.PointDrawTool(
            renderers=[r for r in fig.renderers if isinstance(r.glyph, bm.Rect)],
            empty_value="black",
        ),
        hover_tool := bm.HoverTool(
            renderers=[r for r in fig.renderers if isinstance(r.glyph, bm.Rect)],
        ),
        # pan_tool := bm.PanTool(),
        tap_tool := bm.TapTool(),
        zoom := bm.WheelZoomTool(),
    )
    fig.toolbar.active_scroll = zoom
    fig.toolbar.active_tap = tap_tool
    fig.toolbar.active_drag = draw_tool
    fig.toolbar.logo = None
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.yaxis.major_label_text_font_size = "0pt"
    fig.match_aspect = True
    hover_tool.tooltips = [("", "@tag"), ("xy", "$x{0.000} , $y{0.000}")]

    def bkapp(doc):
        doc.add_root(fig)

    return bkapp


def _resolve_x(netlist, x, dx):
    # TODO: this function 'works' but is not 100% correct I think...
    if x is None:
        x = 0.0
    if dx is None:
        dx = 0.0
    elif isinstance(x, str):
        try:
            x = float(x)
        except:
            inst, port = x.split(",")  # TODO: use port...
            x = _resolve_x(
                netlist, netlist.placements[inst].x, netlist.placements[inst].dx
            )
    return float(x) + float(dx)


def _resolve_y(netlist, y, dy):
    # TODO: this function 'works' but is not 100% correct I think...
    if y is None:
        y = 0.0
    if dy is None:
        dy = 0.0
    elif isinstance(y, str):
        inst, port = y.split(",")  # TODO: use port...
        y = _resolve_y(netlist, netlist.placements[inst].y, netlist.placements[inst].dy)
    return float(y) + float(dy)


def get_placements(netlist):
    ret = {}
    for k, v in netlist.placements.items():
        x = _resolve_x(netlist, v.x, v.dx)
        y = _resolve_x(netlist, v.y, v.dy)
        ret[k] = (x, y)
    return ret


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


# get_input_ports(netlist, 'i1')

# export
def ports_ys(ports, instance_height):
    h = instance_height
    if len(ports) < 1:
        return [h / 2]
    ys = np.linspace(0, h, len(ports) + 1)
    dy = ys[1] - ys[0]
    return ys[1:] - dy / 2


# export
def viz_instance(
    netlist: PicYamlConfiguration, instance_name, component, instance_size
):
    # inst_spec = netlist.instances[instance_name].dict()
    c = component
    bbox = c.bbox
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    x0 = bbox[0][0]
    y0 = bbox[0][1]
    pl = w / 10
    input_ports = get_input_ports(component)
    output_ports = get_output_ports(component)
    y_inputs = ports_ys(input_ports, h)
    y_outputs = ports_ys(output_ports, h)
    x, y = get_placements(netlist).get(instance_name, (0, 0))

    ports: List[gf.Port] = c.ports.values()
    ports = [p.copy() for p in ports]
    for p in ports:
        p.move((x, y))
        p.tag = instance_name
    if False:  # hierarchical:
        c = "#FF0000"
    else:
        c = "#000000"

    r = Rect(tag=instance_name, x=x + x0, y=y + y0, w=w, h=h, c=c)
    input_ports = [
        LineSegment(instance_name, x, y + yi, x + pl, y + yi) for yi in y_inputs
    ]
    output_ports = [
        LineSegment(instance_name, x + w - pl, y + yi, x + w, y + yi)
        for yi in y_outputs
    ]
    ret = [r, *ports]
    return ret


# export
def split_port(port, netlist):
    if "," not in port:
        port = netlist.ports[port]
    *instance_name, port = port.split(",")
    return ",".join(instance_name), port


# export
def get_port_location(netlist, port, instance_size, component):
    if "," not in port:
        port = netlist.ports[port]
    w, h = instance_size, instance_size
    instance_name, port = split_port(port, netlist)
    placements = get_placements(netlist)
    x, y = placements[instance_name]
    if is_input_port(port):
        ports = get_input_ports(component)
        idx = ports.index(port)
        ys = ports_ys(ports, h)
        y += ys[idx]
    else:
        ports = get_output_ports(component)
        idx = ports.index(port)
        ys = ports_ys(ports, h)
        y += ys[idx]
        x = x + w
    return x, y


# export
def viz_connection(netlist, p_in, p_out, instance_size, component):
    x1, y1 = get_port_location(netlist, p_in, instance_size, component)
    x2, y2 = get_port_location(netlist, p_out, instance_size, component)
    tag = f"{p_in.split(',')[0]},{p_out.split(',')[0]}"
    line = LineSegment(tag, x1, y1, x2, y2)
    return [line]


# export
def viz_netlist(netlist, instances, instance_size=20):
    els = []
    for instance_name in netlist.instances:
        els += viz_instance(
            netlist, instance_name, instances[instance_name], instance_size
        )

    # HOW TO RESOLVE CONNECTIONS?
    # for p_in, p_out in netlist.connections.items():
    #     els += viz_connection(netlist, p_in, p_out, instance_size)
    return els


def show_netlist(netlist: Dict, instances: Dict):
    global data
    picmodel = PicYamlConfiguration(
        instances=netlist["instances"],
        placements=netlist.get("schematic_placements"),
        ports=netlist.get("ports"),
    )
    data["netlist"] = picmodel
    fig = bp.figure(width=800, height=500)
    app = viz_bk(picmodel, instances=instances, fig=fig, instance_size=50)
    bio.show(app)


def edit_netlist(netlist):
    global data
    data["netlist"] = netlist
    fig = bp.figure(width=800, height=500)
    app = viz_bk(netlist, fig=fig, instance_size=50)
    bio.show(app)
    save_netlist(netlist, NETLIST_FILENAME)


def add_instance(
    name: str, component: gf.Component, placement: Optional[Placement] = None
):
    data["netlist"].add_instance(
        name=name,
        component=component,
        placement=placement,
    )
    inst_viz = viz_instance(data["netlist"], name, 0)
    srcs = _get_sources([inst_viz])
    for k, src in srcs.items():
        cds: bm.ColumnDataSource = data["dss"][k]
        cds.stream(src)


def get_deltas(netlist):
    deltas = {}
    for k, p in netlist.placements.items():
        deltas[k] = {"dx": p.dx or 0, "dy": p.dy or 0}
    return deltas


def apply_deltas(netlist, deltas):
    for k, d in deltas.items():
        netlist.placements[k].dx = d["dx"]
        netlist.placements[k].dy = d["dy"]


# def move_instance(name: str, dx: float, dy: float):
#     data['netlist'].move_instance(name=name,
#                                   dx=dx,
#                                   dy=dy)

# fig = bp.figure(width=1500, height=1500)
# app = viz_bk(netlist, fig=fig, instance_size=50)
# app(curdoc())
