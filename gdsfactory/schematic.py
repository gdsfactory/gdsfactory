import json
import warnings
from math import sqrt
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from graphviz import Digraph
from pydantic import BaseModel, Field, model_validator

import gdsfactory
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.typings import Anchor, Delta


class Instance(BaseModel):
    """Instance of a component.

    Parameters:
        component: component name.
        settings: input variables.
        info: information (polarization, wavelength ...).
        columns: number of columns.
        rows: number of rows.
        column_pitch: column pitch.
        row_pitch: row pitch.
    """

    component: str
    settings: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict, exclude=True)
    columns: int = 1
    rows: int = 1
    column_pitch: float = 0
    row_pitch: float = 0

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def update_settings_and_info(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validator to update component, settings and info based on the component."""
        component = values.get("component")
        settings = values.get("settings", {})
        info = values.get("info", {})

        import gdsfactory as gf

        c = gf.get_component(component, settings=settings)
        component_info = c.info.model_dump(exclude_none=True)
        component_settings = c.settings.model_dump(exclude_none=True)
        values["info"] = {**component_info, **info}
        values["settings"] = {**component_settings, **settings}
        values["component"] = c.function_name or component
        return values


class Placement(BaseModel):
    x: str | float | None = None
    y: str | float | None = None
    xmin: str | float | None = None
    ymin: str | float | None = None
    xmax: str | float | None = None
    ymax: str | float | None = None
    dx: Delta = 0
    dy: Delta = 0
    port: str | Anchor | None = None
    rotation: float = 0
    mirror: bool | str | float = False

    def __getitem__(self, key: str) -> Any:
        """Allows to access the placement attributes as a dictionary."""
        return getattr(self, key, 0)

    model_config = {"extra": "forbid"}


class Bundle(BaseModel):
    links: dict[str, str]
    settings: dict[str, Any] = Field(default_factory=dict)
    routing_strategy: str = "route_bundle"

    model_config = {"extra": "forbid"}


class Net(BaseModel):
    """Net between two ports.

    Parameters:
        p1: instance_name,port 1.
        p2: instance_name,port 2.
        name: route name.
    """

    p1: str
    p2: str
    settings: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize the net."""
        global _route_counter
        super().__init__(**data)
        # If route name is not provided, generate one automatically
        if self.name is None:
            self.name = f"route_{_route_counter}"
            _route_counter += 1


class Netlist(BaseModel):
    """Netlist defined component.

    Parameters:
        instances: dict of instances (name, settings, component).
        placements: dict of placements.
        connections: dict of connections.
        routes: dict of routes.
        name: component name.
        info: information (polarization, wavelength ...).
        ports: exposed component ports.
        settings: input variables.
        nets: list of nets.
        warnings: warnings.
    """

    pdk: str = ""
    instances: dict[str, Instance] = Field(default_factory=dict)
    placements: dict[str, Placement] = Field(default_factory=dict)
    connections: dict[str, str] = Field(default_factory=dict)
    routes: dict[str, Bundle] = Field(default_factory=dict)
    name: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    ports: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict, exclude=True)
    nets: list[Net] = Field(default_factory=list)
    warnings: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


_route_counter = 0


def to_yaml_graph_networkx(
    netlist: Netlist, nets: list[Net]
) -> tuple[nx.Graph, dict[str, str], dict[str, tuple[float, float]]]:
    """Generates a netlist graph using NetworkX."""
    connections = netlist.connections
    placements = netlist.placements
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (",".join(k.split(",")[:-1]), ",".join(v.split(",")[:-1]))
            for k, v in connections.items()
        ]
    )
    pos = {k: (v["x"], v["y"]) for k, v in placements.items()}
    labels = {k: ",".join(k.split(",")[:1]) for k in placements.keys()}

    for node, placement in placements.items():
        if not graph.has_node(
            node
        ):  # Check if the node is already in the graph (from connections), to avoid duplication.
            graph.add_node(node)
            pos[node] = (placement.x, placement.y)

    for net in nets:
        graph.add_edge(net.p1.split(",")[0], net.p2.split(",")[0])

    return graph, labels, pos


def to_graphviz(
    instances: dict[str, Instance],
    placements: dict[str, Placement],
    nets: list[Net],
    show_ports: bool = True,
) -> Digraph:
    """Generates a netlist graph using Graphviz."""
    from graphviz import Digraph

    dot = Digraph(comment="Netlist Diagram")

    canvas_width = 10  # in inches
    canvas_height = 10  # in inches
    dpi = 300
    node_reduction_factor = 0.4  # to prevent nodes from overlapping
    vertical_conflict_factor = (
        3  # minimum vertical separation between two nodes (in multiples of width)
    )
    horizontal_conflict_factor = (
        2  # minimum horizontal separation between two nodes (in multiples of height)
    )

    dot.attr(
        dpi=str(dpi),
        layout="neato",
        overlap="scale",
        size=f"{canvas_width},{canvas_height}!",
    )

    # Retrieve all the ports in the component
    all_ports = []
    for name, instance in instances.items():
        if hasattr(instance, "component"):
            instance = instance.component
        else:
            instance = instance["component"]
        ports = gdsfactory.get_component(instance).ports
        all_ports.append((name, ports))

    # Check the range of positions
    x_values = [
        placement.x if hasattr(placement, "x") else placement["x"]
        for placement in placements.values()
    ]
    y_values = [
        placement.y if hasattr(placement, "y") else placement["y"]
        for placement in placements.values()
    ]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    position_tracker = {}

    for node, placement in placements.items():
        ports = dict(all_ports).get(node)

        # Define a subgraph for each component without an outer frame
        with dot.subgraph(name=f"cluster_{node}") as sub:
            sub.attr(style="invis")

            x = placement.x if hasattr(placement, "x") else placement["x"]
            y = placement.y if hasattr(placement, "y") else placement["y"]
            range_x = max_x - min_x
            range_y = max_y - min_y
            effective_canvas_area = canvas_height * canvas_width
            effective_canvas_area *= (
                range_x * range_y / (max(range_x, range_y)) ** 2
                if range_y > 0 and range_x > 0
                else 1
            )
            node_density = len(instances) / (effective_canvas_area * dpi)
            node_size = 1 / (node_density**0.5)
            node_width = node_size * node_reduction_factor
            node_height = node_size * node_reduction_factor
            font_size = min(node_width, node_height) * 20

            # Normalize and scale
            scaling_factor = (
                300 / max(range_x, range_y) if max(range_x, range_y) > 0 else 1
            )
            x = (
                scaling_factor * (x - min_x) if range_x > 0 else 150
            )  # Center if no range
            y = scaling_factor * (y - min_y) if range_y > 0 else 150

            pos = (x, y)

            # Check for exact position overlap and proximity
            attempts = 0
            while any(
                abs(pos[0] - tracked_pos[0]) < horizontal_conflict_factor * node_width
                and abs(pos[1] - tracked_pos[1])
                < vertical_conflict_factor * node_height
                for tracked_pos in position_tracker
            ):
                conflicting_positions = [
                    tracked_pos
                    for tracked_pos in position_tracker
                    if abs(pos[0] - tracked_pos[0])
                    < horizontal_conflict_factor * node_width
                    and abs(pos[1] - tracked_pos[1])
                    < vertical_conflict_factor * node_height
                ]

                attempts += 1
                if attempts > 10:  # reduce node size
                    node_width *= 0.9
                    node_height *= 0.9

                closest_pos = min(
                    conflicting_positions,
                    key=lambda tracked_pos: sqrt(
                        (pos[0] - tracked_pos[0]) ** 2 + (pos[1] - tracked_pos[1]) ** 2
                    ),
                )

                y_diff = pos[1] - closest_pos[1]
                y += (
                    node_height * vertical_conflict_factor * 0.1
                    if y_diff > 0
                    else -node_height * vertical_conflict_factor * 0.1
                )
                pos = (x, y)

            position_tracker[pos] = node
            sub.node(
                node,
                label=node,
                shape="rectangle",
                pos=f"{x},{y}!",
                width=str(node_width),
                height=str(node_height),
                fontsize=str(font_size),
                style="filled",
                fillcolor="white",
            )

            # Create ports for the components
            if ports and show_ports:
                right_ports, bottom_ports, left_ports, top_ports = [], [], [], []

                for port in ports:
                    orientation = port.orientation
                    if 0 <= orientation < 45 or 315 <= orientation < 360:
                        right_ports.append(port)
                    elif 45 <= orientation < 135:
                        bottom_ports.append(port)
                    elif 135 <= orientation < 225:
                        left_ports.append(port)
                    elif 225 <= orientation < 315:
                        top_ports.append(port)

                def position_ports(port_list, side) -> None:
                    port_count = len(port_list)

                    if port_count == 0:
                        return

                    # Reverse the port list if side is 'left' or 'bottom' to achieve the anticlockwise effect
                    if side in ["left", "bottom"]:
                        port_list = port_list[::-1]

                    if side in ["top", "bottom"]:
                        port_width = node_width / port_count
                        port_height = node_height / 2
                    else:
                        port_width = node_width / 2
                        port_height = node_height / port_count

                    port_font_size = min(port_width, port_height) * 15

                    for i, port in enumerate(port_list):
                        port_name = f"{node}_{port.name.replace('_', '-')}"  # Prevent ambiguity with subgraph notation

                        if side == "right":
                            port_x = x + node_width / 2 + port_width / 2
                            port_y = y + node_height / 2 - (i + 0.5) * port_height
                        elif side == "bottom":
                            port_x = x - node_width / 2 + (i + 0.5) * port_width
                            port_y = y - node_height / 2 - port_height / 2
                        elif side == "left":
                            port_x = x - node_width / 2 - port_width / 2
                            port_y = y + node_height / 2 - (i + 0.5) * port_height
                        elif side == "top":
                            port_x = x - node_width / 2 + (i + 0.5) * port_width
                            port_y = y + node_height / 2 + port_height / 2

                        port_pos = f"{port_x},{port_y}!"
                        sub.node(
                            port_name,
                            label=port_name,
                            shape="rectangle",
                            pos=port_pos,
                            fontsize=str(port_font_size),
                            width=str(port_width),
                            height=str(port_height),
                            style="filled",
                            fillcolor="white",
                        )

                position_ports(right_ports, "right")
                position_ports(bottom_ports, "bottom")
                position_ports(left_ports, "left")
                position_ports(top_ports, "top")

    for net in nets:
        p1 = net.p1 if hasattr(net, "p1") else net["p1"]
        p2 = net.p2 if hasattr(net, "p2") else net["p2"]

        p1_instance = p1.split(",")[0]
        p1_port = p1.split(",")[1]

        p2_instance = p2.split(",")[0]
        p2_port = p2.split(",")[1]

        p1_port = p1_port.replace("_", "-")  # Prevent ambiguity with subgraph notation
        p2_port = p2_port.replace("_", "-")

        dot.edge(f"{p1_instance}_{p1_port}", f"{p2_instance}_{p2_port}", dir="none")

    return dot


class Link(BaseModel):
    """Link between instances.

    Parameters:
        instance1: instance name 1.
        instance2: instance name 2.
        port1: port name 1.
        port2: port name 2.
    """

    instance1: str
    instance2: str
    port1: str
    port2: str


class Schematic(BaseModel):
    """Schematic."""

    netlist: Netlist = Field(default_factory=Netlist)
    nets: list[Net] = Field(default_factory=list)
    placements: dict[str, Placement] = Field(default_factory=dict)
    links: list[Link] = Field(default_factory=list)

    def add_instance(
        self, name: str, instance: Instance, placement: Placement | None = None
    ) -> None:
        self.netlist.instances[name] = instance
        if placement:
            self.add_placement(name, placement)

    def add_placement(
        self,
        instance_name: str,
        placement: Placement,
    ) -> None:
        """Add placement to the netlist.

        Args:
            instance_name: instance name.
            placement: placement.
        """
        self.placements[instance_name] = placement
        self.netlist.placements[instance_name] = placement

    def from_component(self, component: Component) -> None:
        n = component.to_yaml()
        self.netlist = Netlist.model_validate(n)

    def add_net(self, net: Net) -> None:
        """Add a net between two ports."""
        self.nets.append(net)
        if net.name not in self.netlist.routes:
            self.netlist.routes[net.name] = Bundle(
                links={net.p1: net.p2}, settings=net.settings
            )
        else:
            self.netlist.routes[net.name].links[net.p1] = net.p2

    def to_graphviz(self, show_ports: bool = True) -> Digraph:
        """Generates a netlist graph using Graphviz.

        Args:
            show_ports: whether to show ports or not.
        """
        return to_graphviz(
            self.netlist.instances, self.placements, self.nets, show_ports
        )

    def to_yaml_graph_networkx(
        self,
    ) -> tuple[nx.Graph, dict[str, str], dict[str, tuple[float, float]]]:
        return to_yaml_graph_networkx(self.netlist, self.nets)

    def plot_graphviz(self, interactive: bool = False, splines: str = "ortho") -> None:
        """Plots the netlist graph (Automatic fallback to networkx).

        Args:
            interactive: whether to plot the graph interactively or not.
            splines: type of splines to use for the graph.

        """
        dot = self.to_graphviz()
        plot_graphviz(dot, interactive=interactive, splines=splines)

    def plot_schematic_networkx(self) -> None:
        """Plots the netlist graph (Automatic fallback to networkx)."""
        warnings.warn(
            "plot_schematic_networkx is deprecated. Use plot_graphviz instead",
            DeprecationWarning,
        )
        self.plot_graphviz()


def plot_graphviz(
    graph: Digraph, interactive: bool = False, splines: str = "ortho"
) -> None:
    """Plots the netlist graph (Automatic fallback to networkx)."""
    from IPython.display import Image, display

    valid_splines = ["ortho", "spline", "line", "polyline", "curved"]
    if splines not in valid_splines:
        raise ValueError(f"Invalid splines value. Choose from {valid_splines}")

    graph.graph_attr["splines"] = splines

    if interactive:
        graph.view()
    else:
        png_data = graph.pipe(format="png")
        display(Image(data=png_data))


def write_schema(
    model: BaseModel = Netlist, schema_path_json: Path = PATH.schema_netlist
) -> None:
    s = model.model_json_schema()
    schema_path_yaml = schema_path_json.with_suffix(".yaml")

    with open(schema_path_json, "w") as f:
        json.dump(s, f, indent=2)
    with open(schema_path_yaml, "w") as f:
        yaml.dump(s, f)


if __name__ == "__main__":
    # write_schema()
    import gdsfactory as gf
    import gdsfactory.schematic as gt

    s = gt.Schematic()
    s.add_instance("s11", gt.Instance(component=gf.c.mmi1x2()))
    s.add_instance("s21", gt.Instance(component=gf.c.mmi1x2()))
    s.add_instance("s22", gt.Instance(component=gf.c.mmi1x2()))
    s.add_placement("s11", gt.Placement(x=000, y=0))
    s.add_placement("s21", gt.Placement(x=100, y=+50))
    s.add_placement("s22", gt.Placement(x=100, y=-50))
    s.add_net(gt.Net(p1="s11,o2", p2="s21,o1"))
    s.add_net(gt.Net(p1="s11,o3", p2="s22,o1"))
    g = s.plot_graphviz()
