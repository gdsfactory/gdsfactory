# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% [markdown]
# # gdsfactory downloads
#
# You can plot the downloads for gdsfactory over the last year.

# %%
import requests
import datetime
import plotly.graph_objects as go


def get_total_downloads(package_name):
    statistics = []
    end_date = datetime.date.today()

    while True:
        url = f"https://pypistats.org/api/packages/{package_name}/overall"
        response = requests.get(url, params={"last_day": end_date})
        data = response.json()

        if response.status_code != 200:
            return None

        statistics.extend(
            [(entry["date"], entry["downloads"]) for entry in data["data"]]
        )
        if "next_day" in data:
            end_date = data["next_day"]
        else:
            break
    statistics.sort(key=lambda x: x[0])  # Sort by date
    dates, downloads = zip(*statistics)
    cumulative_downloads = [sum(downloads[: i + 1]) for i in range(len(downloads))]

    return dates, cumulative_downloads


# Replace 'gdsfactory' with the package you want to check
package_name = "gdsfactory"
dates, cumulative_downloads = get_total_downloads(package_name)

if dates and cumulative_downloads:
    fig = go.Figure(data=go.Scatter(x=dates, y=cumulative_downloads))
    fig.update_layout(
        xaxis=dict(title="Date", tickformat="%Y-%m-%d", tickangle=45, showgrid=True),
        yaxis=dict(title="Total Downloads", showgrid=True),
        title=f"Total Downloads - {package_name}",
    )
    fig.show()
else:
    print(f"Failed to retrieve download statistics for package '{package_name}'.")


# %% [markdown]
# ## dependencies

# %%
import contextlib
import pkg_resources
import networkx as nx
import matplotlib.pyplot as plt


def build_dependency_graph(package_name):
    graph = nx.DiGraph()
    visited = set()

    def traverse_dependencies(package):
        if package not in visited:
            visited.add(package)
            graph.add_node(package)

            with contextlib.suppress(pkg_resources.DistributionNotFound):
                dependencies = pkg_resources.get_distribution(package).requires()
                for dependency in dependencies:
                    graph.add_edge(package, dependency)
                    traverse_dependencies(dependency)

    traverse_dependencies(package_name)
    return graph


# Specify the name of the package you want to build the dependency graph for
package_name = "gdsfactory"

# Build the dependency graph
dependency_graph = build_dependency_graph(package_name)

# Customize the graph layout
pos = nx.spring_layout(dependency_graph, k=0.25)

# Increase the figure size to accommodate the graph
plt.figure(figsize=(12, 8))

# Draw nodes with different styles for the main package and its dependencies
nx.draw_networkx_nodes(
    dependency_graph, pos, node_color="lightblue", node_size=1000, alpha=0.9
)
nx.draw_networkx_nodes(
    dependency_graph,
    pos,
    nodelist=[package_name],
    node_color="salmon",
    node_size=1200,
    alpha=0.9,
)

# Draw edges with different styles for the main package and its dependencies
nx.draw_networkx_edges(dependency_graph, pos, edge_color="gray", alpha=0.5)
nx.draw_networkx_edges(
    dependency_graph,
    pos,
    edgelist=[(package_name, dep) for dep in dependency_graph[package_name]],
    edge_color="red",
    alpha=0.7,
    width=2,
)

# Draw node labels
nx.draw_networkx_labels(dependency_graph, pos, font_size=10, font_weight="bold")

# Customize plot appearance
plt.title(f"Dependency Graph for {package_name}")
plt.axis("off")
plt.tight_layout()

# Show the graph
plt.show()


# %%
import pkg_resources
import networkx as nx
import matplotlib.pyplot as plt


def build_dependency_graph(package_name):
    graph = nx.DiGraph()
    visited = set()

    def traverse_dependencies(package):
        if package not in visited:
            visited.add(package)
            graph.add_node(package)

            try:
                dependencies = pkg_resources.get_distribution(package).requires()
                for dependency in dependencies:
                    graph.add_edge(package, dependency)
                    traverse_dependencies(dependency)
            except pkg_resources.DistributionNotFound:
                # Package is not installed or cannot be found
                pass

    traverse_dependencies(package_name)
    return graph


# Specify the name of the package you want to build the dependency graph for
package_name = "gdsfactory"

# Build the dependency graph
dependency_graph = build_dependency_graph(package_name)

# Customize the graph layout
pos = nx.spring_layout(dependency_graph, k=0.25)

# Increase the figure size to accommodate the graph
plt.figure(figsize=(12, 8))

# Draw nodes with different styles for the main package and its dependencies
nx.draw_networkx_nodes(
    dependency_graph, pos, node_color="lightblue", node_size=1000, alpha=0.9
)
nx.draw_networkx_nodes(
    dependency_graph,
    pos,
    nodelist=[package_name],
    node_color="salmon",
    node_size=1200,
    alpha=0.9,
)

# Draw edges with different styles for the main package and its dependencies
nx.draw_networkx_edges(dependency_graph, pos, edge_color="gray", alpha=0.5)
nx.draw_networkx_edges(
    dependency_graph,
    pos,
    edgelist=[(package_name, dep) for dep in dependency_graph[package_name]],
    edge_color="red",
    alpha=0.7,
    width=2,
)

# Draw node labels
nx.draw_networkx_labels(dependency_graph, pos, font_size=10, font_weight="bold")

# Customize plot appearance
plt.title(f"Dependency Graph for {package_name}")
plt.axis("off")
plt.tight_layout()

# Show the graph
plt.show()


# %%
import pkg_resources
import networkx as nx
import plotly.graph_objects as go


def build_dependency_graph(package_name):
    graph = nx.DiGraph()
    visited = set()

    def traverse_dependencies(package):
        if package not in visited:
            visited.add(package)
            graph.add_node(package)

            try:
                dependencies = pkg_resources.get_distribution(package).requires()
                for dependency in dependencies:
                    graph.add_edge(package, dependency)
                    traverse_dependencies(dependency)
            except pkg_resources.DistributionNotFound:
                # Package is not installed or cannot be found
                pass

    traverse_dependencies(package_name)
    return graph


# Specify the name of the package you want to build the dependency graph for
package_name = "gdsfactory"

# Build the dependency graph
dependency_graph = build_dependency_graph(package_name)

# Create nodes and edges
nodes = dependency_graph.nodes()
edges = dependency_graph.edges()

# Create Plotly nodes
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode="markers",
    hoverinfo="text",
    marker=dict(size=15, color="lightblue", line_width=2),
)

# Create Plotly edges
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1, color="gray"),
    hoverinfo="none",
    mode="lines",
)

# Assign positions to nodes
pos = nx.spring_layout(dependency_graph, k=0.2)
for node in nodes:
    x, y = pos[node]
    node_trace["x"] += (x,)
    node_trace["y"] += (y,)
    node_trace["text"] += (node,)

# Assign positions to edges
for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace["x"] += (x0, x1, None)
    edge_trace["y"] += (y0, y1, None)

# Create the layout for the graph
layout = go.Layout(
    title=f"Dependency Graph for {package_name}",
    title_font=dict(size=20),
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

# Combine nodes and edges into a data list
data = [edge_trace, node_trace]

# Create the figure and plot the graph
fig = go.Figure(data=data, layout=layout)
fig.show()


# %%
len(nodes)
