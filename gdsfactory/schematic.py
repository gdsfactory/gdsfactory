import json
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from graphviz import Digraph
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

import gdsfactory
from gdsfactory._deprecation import deprecate
from gdsfactory.component import Component
from gdsfactory.config import PATH
from gdsfactory.typings import Anchor, Delta
from gdsfactory.utils import is_component_spec


class OrthogonalGridArray(BaseModel):
    """Orthogonal grid array config.

    Parameters:
        columns: number of columns.
        rows: number of rows.
        column_pitch: column pitch.
        row_pitch: row pitch.
    """

    columns: int = 1
    rows: int = 1
    column_pitch: float = 0
    row_pitch: float = 0


class GridArray(BaseModel):
    """Orthogonal grid array config.

    Parameters:
        columns: number of columns.
        rows: number of rows.
        column_pitch: column pitch.
        row_pitch: row pitch.
    """

    num_a: int = 1
    num_b: int = 1
    pitch_a: tuple[float, float] = (1.0, 0.0)
    pitch_b: tuple[float, float] = (0.0, 1.0)


class Instance(BaseModel):
    """Instance of a component.

    Parameters:
        component: component name.
        settings: input variables.
        info: information (polarization, wavelength ...).
        array: array config to make create an array reference for this instance
    """

    component: str
    settings: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict, exclude=True)
    array: OrthogonalGridArray | GridArray | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def update_settings_and_info(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validator to update component, settings and info based on the component."""
        component = values.get("component")
        settings = values.get("settings", {})
        info = values.get("info", {})

        assert is_component_spec(component)

        import gdsfactory as gf

        c = gf.get_component(component, settings=settings)
        component_info = c.info.model_dump(exclude_none=True)
        component_settings = c.settings.model_dump(exclude_none=True)
        values["info"] = {**component_info, **info}
        values["settings"] = {**component_settings, **settings}
        values["component"] = c.function_name or component
        return values

    @model_validator(mode="after")
    def validate_array(self) -> Self:
        if isinstance(self.array, OrthogonalGridArray):
            if self.array.columns < 2:
                self.array.columns = 1
            if self.array.rows < 2:
                self.array.rows = 1
            if self.array.columns == 1 and self.array.rows == 1:
                self.array = None
        elif isinstance(self.array, GridArray):
            if self.array.num_a < 2:
                self.array.num_a = 1
            if self.array.num_b < 2:
                self.array.num_b = 1
            if self.array.num_a == 1 and self.array.num_b == 1:
                self.array = None
        return self


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

    @model_validator(mode="after")
    def validate_instance_names(self) -> Self:
        self.instances = {
            _validate_instance_name(k): v for k, v in self.instances.items()
        }
        return self


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

    # Graphviz implementation
    dot = Digraph(comment="Netlist Diagram")
    dot.attr(dpi="300", layout="neato", overlap="false")

    all_ports = []

    # Retrieve all the ports in the component
    for name, instance in instances.items():
        if hasattr(instance, "component"):
            instance_component = instance.component
        else:
            instance_component = instance["component"]  # type: ignore
        ports = gdsfactory.get_component(instance_component).ports
        all_ports.append((name, ports))

    for node, placement in placements.items():
        _ports = dict(all_ports).get(node)
        assert _ports is not None
        ports = _ports

        if not ports or not show_ports:
            label = node
        else:
            top_ports, right_ports, bottom_ports, left_ports = [], [], [], []

            for port in ports:
                if 0 <= port.orientation < 45 or 315 <= port.orientation < 360:
                    right_ports.append(port)
                elif 45 <= port.orientation < 135:
                    bottom_ports.append(port)
                elif 135 <= port.orientation < 225:
                    left_ports.append(port)
                elif 225 <= port.orientation < 315:
                    top_ports.append(port)

            # Format ports for Graphviz record structure in anticlockwise order
            port_labels = []

            if left_ports:
                left_ports_label = " | ".join(
                    f"<{port.name}> {port.name}" for port in reversed(left_ports)
                )
                port_labels.append(f"{{ {left_ports_label} }}")

            middle_row = []

            if top_ports:
                top_ports_label = " | ".join(
                    f"<{port.name}> {port.name}" for port in top_ports
                )
                middle_row.append(f"{{ {top_ports_label} }}")

            middle_row.append(node)

            if bottom_ports:
                bottom_ports_label = " | ".join(
                    f"<{port.name}> {port.name}" for port in reversed(bottom_ports)
                )
                middle_row.append(f"{{ {bottom_ports_label} }}")

            port_labels.append(f"{{ {' | '.join(middle_row)} }}")

            if right_ports:
                right_ports_label = " | ".join(
                    f"<{port.name}> {port.name}" for port in right_ports
                )
                port_labels.append(f"{{ {right_ports_label} }}")

            label = " | ".join(port_labels)

        x = placement.x if hasattr(placement, "x") else placement["x"]
        y = placement.y if hasattr(placement, "y") else placement["y"]
        pos = f"{x},{y}!"
        dot.node(node, label=label, pos=pos, shape="record")

    for net in nets:
        p1 = net.p1 if hasattr(net, "p1") else net["p1"]  # type: ignore
        p2 = net.p2 if hasattr(net, "p2") else net["p2"]  # type: ignore

        p1_instance = p1.split(",")[0]
        p1_port = p1.split(",")[1]

        p2_instance = p2.split(",")[0]
        p2_port = p2.split(",")[1]

        dot.edge(f"{p1_instance}:{p1_port}", f"{p2_instance}:{p2_port}", dir="none")

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
        raise NotImplementedError
        n = component.to_yaml()
        self.netlist = Netlist.model_validate(n)

    def add_net(self, net: Net) -> None:
        """Add a net between two ports."""
        self.nets.append(net)
        if net.name not in self.netlist.routes:
            assert net.name is not None
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
        deprecate("plot_schematic_networkx", "plot_graphviz")
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
        display(Image(data=png_data))  # type: ignore


def write_schema(
    model: type[BaseModel] = Netlist, schema_path_json: Path = PATH.schema_netlist
) -> None:
    s = model.model_json_schema()
    schema_path_yaml = schema_path_json.with_suffix(".yaml")

    with open(schema_path_json, "w") as f:
        json.dump(s, f, indent=2)
    with open(schema_path_yaml, "w") as f:
        yaml.dump(s, f)


def _validate_instance_name(name: str) -> str:
    if "," in name:
        raise ValueError(
            f"Having a ',' in an instance name is not supported. The ',' is used for port-delineation. Got: {name!r}."
        )
    if "-" in name:
        raise ValueError(
            f"Having a '-' in an instance name is not supported. The '-' is used for bundle routing. Got: {name!r}."
        )
    if ":" in name:
        raise ValueError(
            f"Having a ':' in an instance name is not supported. The ':' is used for bundle routing. Got: {name!r}."
        )
    return name


if __name__ == "__main__":
    # write_schema()
    import gdsfactory as gf
    import gdsfactory.schematic as gt

    s = Schematic()
    s.add_instance("mzi1", gt.Instance(component=gf.c.mzi(delta_length=10)))  # type: ignore
    s.add_instance("mzi2", gt.Instance(component=gf.c.mzi(delta_length=100)))  # type: ignore
    s.add_instance("mzi3", gt.Instance(component=gf.c.mzi(delta_length=200)))  # type: ignore
    s.add_placement("mzi1", gt.Placement(x=000, y=0))
    s.add_placement("mzi2", gt.Placement(x=100, y=100))
    s.add_placement("mzi3", gt.Placement(x=200, y=0))
    s.add_net(gt.Net(p1="mzi1,o2", p2="mzi2,o2"))
    s.add_net(gt.Net(p1="mzi2,o2", p2="mzi3,o1"))
    dot = s.to_graphviz()
    s.plot_graphviz()
