from typing import Dict, Optional, List, Any, Union

import networkx as nx
import pandas as pd
import warnings

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import row
from bokeh.plotting import figure, from_networkx
from bokeh.models import (
    BoxSelectTool,
    Circle,
    HoverTool,
    MultiLine,
    NodesAndLinkedEdges,
    TapTool,
    WheelZoomTool,
    ColumnDataSource,
    Patches,
    TableColumn,
    DataTable,
    HTMLTemplateFormatter,
)
from bokeh.palettes import Spectral4, Category10
import numpy as np
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import CrossSectionSpec
from pathlib import Path


DEFAULT_CS_COLORS = {
    "rib": "red",
    "strip": "blue",
    "r2s": "purple",
    "m1": "#00FF92",
    "m2": "gold",
}


def get_internal_netlist_attributes(
    route_inst_def: Dict[str, Dict], route_info: Optional[Dict], component: Component
):
    if route_info:
        link = _get_link_name(component)
        component_name = route_inst_def["component"]
        attrs = route_info
        attrs["component"] = component_name
        return {link: attrs}
    else:
        return None


def _get_link_name(component: Component):
    ports = sorted(component.ports.keys())
    if len(ports) != 2:
        raise ValueError("routing components must have two ports")
    link = ":".join(ports)
    return link


def _node_to_inst_port(node: str):
    ip = node.split(",")
    if len(ip) == 2:
        inst, port = ip
    elif len(ip) == 1:
        port = ip[0]
        inst = ""
    else:
        raise ValueError(
            f"did not expect a connection name with more than one comma: {node}"
        )
    return inst, port


def _is_scalar(val):
    return isinstance(val, float) or isinstance(val, int)


def _expand_bbox(bbox):
    if len(bbox) == 2:
        bbox = [bbox[0], (bbox[0][0], bbox[1][1]), bbox[1], (bbox[1][0], bbox[0][1])]
    return bbox


def sum_route_attrs(records):
    totals = {}
    for record in records:
        for k, v in record.items():
            if _is_scalar(v):
                if k not in totals:
                    totals[k] = v
                else:
                    totals[k] += v
    return totals


def report_pathlengths(
    pic: Component, result_dir, visualize=False, component_connectivity=None
):
    print(f"Reporting pathlengths for {pic.name}...")
    pathlength_graph = get_edge_based_route_attr_graph(
        pic, recursive=True, component_connectivity=component_connectivity
    )
    route_records = get_paths(pathlength_graph)

    if route_records:
        if not result_dir.is_dir():
            result_dir.mkdir()
        pathlength_table_filename = result_dir / f"{pic.name}.pathlengths.csv"

        df = pd.DataFrame.from_records(route_records)
        df.to_csv(pathlength_table_filename)
        print(f"Success! Wrote pathlength table to {pathlength_table_filename}")

    if visualize:
        visualize_graph(pic, pathlength_graph, route_records, result_dir)


def get_paths(pathlength_graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    Gets a list of dictionaries from the pathlength graph describing each of the aggregate paths.

    Args:
        pathlength_graph: a graph representing a circuit
    """
    paths = nx.connected_components(pathlength_graph)
    route_records = []
    for path in paths:
        node_degrees = pathlength_graph.degree(path)
        end_nodes = [n for n, deg in node_degrees if deg == 1]
        end_ports = []
        for node in end_nodes:
            inst, port = _node_to_inst_port(node)
            end_ports.append((inst, port))
        if len(end_ports) > 1:
            node_pairs = []
            for n1 in end_nodes:
                for n2 in end_nodes:
                    if n1 != n2:
                        s = {n1, n2}
                        if s not in node_pairs:
                            node_pairs.append(s)
            for node_pair in node_pairs:
                end_nodes = list(node_pair)

                all_paths = nx.all_shortest_paths(pathlength_graph, *end_nodes)
                for path in all_paths:
                    record = {}
                    record["src_inst"], record["src_port"] = end_ports[0]
                    record["src_node"] = end_nodes[0]
                    record["dst_inst"], record["dst_port"] = end_ports[1]
                    record["dst_node"] = end_nodes[1]
                    insts = [n.partition(",")[0] for n in path]
                    valid_path = True
                    for i in range(1, len(insts) - 1):
                        if insts[i - 1] == insts[i] == insts[i + 1]:
                            if i == len(insts) - 2:
                                path.pop()
                            elif i == 1:
                                path.pop(0)
                            else:
                                valid_path = False
                                break
                            end_nodes = [path[0], path[-1]]
                            end_ports2 = []
                            for node in end_nodes:
                                inst, port = _node_to_inst_port(node)
                                end_ports2.append((inst, port))
                            record["src_inst"], record["src_port"] = end_ports2[0]
                            record["src_node"] = end_nodes[0]
                            record["dst_inst"], record["dst_port"] = end_ports2[1]
                            record["dst_node"] = end_nodes[1]
                    if not valid_path:
                        continue
                    edges = pathlength_graph.edges(nbunch=path, data=True)
                    edge_data = [e[2] for e in edges if e[2]]
                    summed_route_attrs = sum_route_attrs(edge_data)
                    if "weight" in summed_route_attrs:
                        summed_route_attrs.pop("weight")
                    if summed_route_attrs:
                        record.update(summed_route_attrs)
                        route_records.append(record)
                    record["edges"] = edges
                    record["nodes"] = path
    return route_records


def _get_subinst_node_name(node_name, inst_name):
    if "," in node_name:
        new_node_name = f"{inst_name}.{node_name}"
    else:
        # for top-level ports
        new_node_name = f"{inst_name},{node_name}"
    return new_node_name


def idealized_mxn_connectivity(inst_name: str, ref: ComponentReference, g: nx.Graph):
    """
    Connects all input ports to all output ports of m x n components, with idealized routes

    Args:
        inst_name: The name of the instance we are providing internal routing for.
        ref: The component reference.
        g: The main graph we are adding connectivity to.
    Returns:
        None (graph is modified in-place)
    """
    warnings.warn(f"using idealized links for {inst_name} ({ref.parent.name})")
    in_ports = [p for p in ref.ports if p.startswith("in")]
    out_ports = [p for p in ref.ports if p.startswith("out")]
    for in_port in in_ports:
        for out_port in out_ports:
            inst_in = f"{inst_name},{in_port}"
            inst_out = f"{inst_name},{out_port}"
            g.add_edge(inst_in, inst_out, weight=0.0001, component=ref.parent.name)


def _get_edge_based_route_attr_graph(
    component: Component,
    recursive=False,
    component_connectivity=None,
    netlist=None,
    netlists=None,
):
    connections = netlist["connections"]
    top_level_ports = netlist["ports"]
    g = nx.Graph()
    inst_route_attrs = {}

    # connect all ports from connections between devices
    node_attrs = {}
    inst_refs = {}

    for inst_name in netlist["instances"]:
        ref = component.named_references[inst_name]
        inst_refs[inst_name] = ref
        if "route_info" in ref.parent.info:
            inst_route_attrs[inst_name] = ref.parent.info["route_info"]
        for port_name, port in ref.ports.items():
            ploc = port.center
            pname = f"{inst_name},{port_name}"
            n_attrs = {
                "x": ploc[0],
                "y": ploc[1],
            }
            node_attrs[pname] = n_attrs
            g.add_node(pname, **n_attrs)
    # nx.set_node_attributes(g, node_attrs)
    g.add_edges_from(connections.items(), weight=0.0001)

    # connect all internal ports for devices with connectivity defined
    # currently we only do this for routing components, but could do it more generally in the future
    for inst_name, inst_dict in netlist["instances"].items():
        route_info = inst_route_attrs.get(inst_name)
        inst_component = component.named_references[inst_name]
        route_attrs = get_internal_netlist_attributes(
            inst_dict, route_info, inst_component
        )
        if route_attrs:
            for link, attrs in route_attrs.items():
                in_port, out_port = link.split(":")
                inst_in = f"{inst_name},{in_port}"
                inst_out = f"{inst_name},{out_port}"
                g.add_edge(inst_in, inst_out, **attrs)
        elif recursive:
            sub_inst = inst_refs[inst_name]
            if sub_inst.parent.name in netlists:
                sub_netlist = netlists[sub_inst.parent.name]
                sub_graph = _get_edge_based_route_attr_graph(
                    sub_inst.parent,
                    recursive=True,
                    component_connectivity=component_connectivity,
                    netlist=sub_netlist,
                    netlists=netlists,
                )
                sub_edges = []
                sub_nodes = []
                for edge in sub_graph.edges(data=True):
                    s, e, d = edge
                    new_edge = []
                    for node_name in [s, e]:
                        new_node_name = _get_subinst_node_name(node_name, inst_name)
                        new_edge.append(new_node_name)
                    new_edge.append(d)
                    sub_edges.append(new_edge)
                for node in sub_graph.nodes(data=True):
                    n, d = node
                    new_name = _get_subinst_node_name(n, inst_name)
                    x = d["x"]
                    y = d["y"]
                    new_pt = sub_inst._transform_point(
                        np.array([x, y]),
                        sub_inst.origin,
                        sub_inst.rotation,
                        sub_inst.x_reflection,
                    )
                    d["x"] = new_pt[0]
                    d["y"] = new_pt[1]
                    new_node = (new_name, d)
                    sub_nodes.append(new_node)
                g.add_nodes_from(sub_nodes)
                g.add_edges_from(sub_edges)
            else:
                if component_connectivity:
                    component_connectivity(inst_name, sub_inst, g)
                else:
                    warnings.warn(
                        f"ignoring any links in {inst_name} ({sub_inst.parent.name})"
                    )

    # connect all top level ports
    if top_level_ports:
        edges = []
        for port, sub_port in top_level_ports.items():
            p_attrs = dict(node_attrs[sub_port])
            e_attrs = {"weight": 0.0001}
            edge = [port, sub_port, e_attrs]
            edges.append(edge)
            g.add_node(port, **p_attrs)
        g.add_edges_from(edges)
    return g


def get_edge_based_route_attr_graph(
    pic: Component, recursive=False, component_connectivity=None
) -> nx.Graph:
    """
    Gets a connectivity graph for the circuit, with all path attributes on edges and ports as nodes.

    Args:
        pic: the pic to generate a graph from
        recursive: True to expand all hierarchy. False to only report top-level connectivity.
        component_connectivity: a function to report connectivity for base components. None to treat as black boxes with no internal connectivity.
    Returns:
        A NetworkX Graph
    """
    from gdsfactory.get_netlist import get_netlist, get_netlist_recursive

    if recursive:
        netlists = get_netlist_recursive(pic, component_suffix="", full_settings=True)
        netlist = netlists[pic.name]
    else:
        netlist = get_netlist(pic, full_settings=True)
        netlists = None

    graph = _get_edge_based_route_attr_graph(
        pic, recursive, component_connectivity, netlist=netlist, netlists=netlists
    )
    return graph


def get_pathlength_widgets(
    pic: Component,
    G: nx.Graph,
    paths: List[Dict[str, Any]],
    cs_colors: Optional[Dict[str, str]] = None,
    default_color: str = "#CCCCCC",
) -> Dict[str, Any]:
    """
    Gets a dictionary of bokeh widgets which can be used to visualize pathlength.

    Args:
        pic: the component to analyze
        G: the connectivity graph
        paths: a list of dictionaries of path attributes
        cs_colors: a dictionary mapping cross-section names to colors to use in the plot
        default_color: the default color to use for unmapped cross-section types

    Returns:
        A dictionary of linked bokeh widgets: the pathlength_table and the pathlength_plot
    """
    inst_infos = {}
    node_positions = {}
    if cs_colors is None:
        cs_colors = DEFAULT_CS_COLORS
    for node, data in G.nodes(data=True):
        node_positions[node] = (data["x"], data["y"])
    instances = pic.named_references
    for inst_name, ref in instances.items():
        ref: ComponentReference
        inst_info = {"bbox": ref.bbox}
        inst_infos[inst_name] = inst_info
    pic_bbox = pic.bbox
    for port_name, port in pic.ports.items():
        p = port.center
        node_positions[port_name] = (p[0], p[1])

    edge_attrs = {}

    for start_node, end_node, edge_data in G.edges(data=True):
        edge_type = edge_data.get("type")
        edge_color = cs_colors.get(edge_type, default_color)
        edge_attrs[(start_node, end_node)] = edge_color
        edge_data["start_name"] = start_node
        edge_data["end_name"] = end_node

    nx.set_edge_attributes(G, edge_attrs, "edge_color")

    plot = figure(
        title=f"{pic.name} -- Connectivity Graph",
        x_range=(pic_bbox[0][0], pic_bbox[1][0]),
        y_range=(pic_bbox[0][1], pic_bbox[1][1]),
    )
    wheel_zoom = WheelZoomTool()
    plot.toolbar.active_scroll = wheel_zoom
    graph_to_viz = nx.convert_node_labels_to_integers(G, label_attribute="name")
    node_positions_by_int_label = {
        k: node_positions[d["name"]] for k, d in graph_to_viz.nodes(data=True)
    }
    graph_renderer = from_networkx(graph_to_viz, node_positions_by_int_label)
    graph_renderer.node_renderer.glyph = Circle(size=5, fill_color=Spectral4[0])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="edge_color", line_alpha=0.8, line_width=5
    )

    graph_renderer.inspection_policy = NodesAndLinkedEdges()
    plot.renderers.append(graph_renderer)

    path_data = {
        "src_inst": [],
        "dst_inst": [],
        "src_port": [],
        "dst_port": [],
        "xs": [],
        "ys": [],
        "rib_length": [],
        "length": [],
        "strip_length": [],
        "r2s_length": [],
        "m2_length": [],
        "color": [],
        "n_bends": [],
    }
    for i, path in enumerate(paths):
        if "dst_node" in path:
            pp = path["nodes"]
            xs = []
            ys = []
            for n in pp:
                n_data = G.nodes[n]
                xs.append(n_data["x"])
                ys.append(n_data["y"])
            path_data["xs"].append(xs)
            path_data["ys"].append(ys)
            path_data["color"].append(Category10[10][i % 10])
            for key in path_data:
                if key not in ["xs", "ys", "color"]:
                    path_data[key].append(path.get(key, 0.0))

    inst_patches = {"xs": [], "ys": [], "names": [], "xl": [], "yt": []}
    for inst_name, inst_info in inst_infos.items():
        bbox = np.array(_expand_bbox(inst_info["bbox"]))
        xs = bbox[:, 0]
        ys = bbox[:, 1]
        inst_patches["xs"].append(xs)
        inst_patches["ys"].append(ys)
        inst_patches["names"].append(inst_name)
        inst_patches["xl"].append(bbox[0][0])
        inst_patches["yt"].append(bbox[0][1])
    inst_source = ColumnDataSource(inst_patches)
    inst_glyph = Patches(xs="xs", ys="ys", fill_color="blue", fill_alpha=0.05)
    plot.add_glyph(inst_source, inst_glyph)

    paths_ds = ColumnDataSource(path_data)
    paths_glyph = MultiLine(
        xs="xs", ys="ys", line_color="black", line_width=0, line_alpha=0.0
    )
    paths_glyph_selected = MultiLine(xs="xs", ys="ys", line_color="color", line_width=8)

    paths_renderer = plot.add_glyph(paths_ds, glyph=paths_glyph)
    paths_renderer.selection_glyph = paths_glyph_selected
    paths_renderer.hover_glyph = MultiLine(
        xs="xs", ys="ys", line_color="color", line_width=8, line_alpha=0.3
    )

    template = """
                <div style="background:<%= color %>;">
                    &ensp;
                </div>
                """
    formatter = HTMLTemplateFormatter(template=template)
    columns = [
        TableColumn(field="color", title="Key", formatter=formatter, width=2),
        TableColumn(field="src_inst", title="Source"),
        TableColumn(field="src_port", title="Port"),
        TableColumn(field="dst_inst", title="Dest"),
        TableColumn(field="dst_port", title="Port"),
        TableColumn(field="length", title="Length"),
    ]
    for cs_name in cs_colors:
        columns.append(TableColumn(field=f"{cs_name}_length", title=cs_name))
    columns.append(TableColumn(field="n_bends", title="# bends"))
    table = DataTable(
        source=paths_ds,
        columns=columns,
        sizing_mode="stretch_height",
        selectable="checkbox",
    )
    hover_tool_ports = HoverTool(
        tooltips=[("port", "@name"), ("x", "@x"), ("y", "@y")],
        renderers=[graph_renderer.node_renderer],
    )
    edge_hover_tool = HoverTool(
        tooltips=[
            ("Start", "@start_name"),
            ("End", "@end_name"),
            ("Length", "@length"),
        ],
        renderers=[graph_renderer.edge_renderer],
        anchor="center",
    )
    plot.add_tools(
        TapTool(), BoxSelectTool(), wheel_zoom, hover_tool_ports, edge_hover_tool
    )
    return {
        "pathlength_table": table,
        "pathlength_plot": plot,
    }


def visualize_graph(
    pic: Component,
    G: nx.Graph,
    paths: List[Dict[str, Any]],
    result_dir: Union[str, Path],
    cs_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Visualizes a pathlength graph with bokeh and shows the output html.

    Args:
        pic: the circuit component
        G: the connectivity graph
        paths: the path statistics
        result_dir: the directory (name or Path) in which to store results
        cs_colors: a mapping of cross-section names to colors to use in the visualization
    """
    widgets = get_pathlength_widgets(pic, G, paths, cs_colors=cs_colors)
    plot = widgets["pathlength_plot"]
    table = widgets["pathlength_table"]
    layout = row(plot, table, sizing_mode="stretch_both")
    curdoc().add_root(layout)
    result_dir = Path(result_dir)
    output_file(result_dir / f"{pic.name}.html")
    show(layout)


def route_info(
    cs_type: str, length: float, length_eff: float = None, taper: bool = False, **kwargs
):
    """
    Gets a dictionary of route info, used by pathlength analysis.

    Args:
        cs_type: cross section type
        length: length
        length_eff: effective length (i.e. an equivalent straight length of a bend)
        taper: True if this component is a taper
        kwargs: other attributes to track
    Returns:
        A dictionary of routing attributes
    """
    if length_eff is None:
        length_eff = length

    d = {
        "type": cs_type,
        "length": length_eff,
        f"{cs_type}_length": length_eff,
        "weight": length_eff,
    }
    if taper:
        d[f"{cs_type}_taper_length"] = length
    d.update(kwargs)
    return d


def route_info_from_cs(
    cs: CrossSectionSpec, length: float, length_eff: float = None, **kwargs
):
    """
    Gets a dictionary of route info, used by pathlength analysis.

    Args:
        cs: cross section object or spec
        length: length
        length_eff: effective length (i.e. an equivalent straight length of a bend)
        kwargs: other attributes to track
    Returns:
        A dictionary of routing attributes
    """
    from gdsfactory import get_cross_section

    x = get_cross_section(cs)
    cs_type = x.info.get("type", str(cs))
    return route_info(cs_type, length=length, length_eff=length_eff, **kwargs)
