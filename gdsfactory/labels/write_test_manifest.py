"""Converts CSV of test site labels into a CSV test manifest."""

import itertools
import json
import pathlib

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def write_test_manifest(csvpath: str | pathlib.Path) -> pd.DataFrame:
    """Converts CSV of test site labels into a CSV test manifest."""
    df_in = pd.read_csv(csvpath)

    # Initialize an empty list to collect the rows
    rows = []

    for _, label in df_in.iterrows():
        x, y, orientation = label.x, label.y, label.rotation
        text = label.text
        d = json.loads(text)
        electrical_ports = d.get("electrical_ports", {})
        optical_component_ports = d.get("optical_component_ports", {})
        optical_alignment_ports = d.get("optical_alignment_ports", {})
        component = d["name"] + f"_{int(x)}_{int(y)}"
        optical_alignment_ports_names = list(optical_alignment_ports.keys())

        for port_in, port_out in itertools.combinations(electrical_ports, 2):
            row = [
                component,
                x + electrical_ports[port_in]["x"],
                y + electrical_ports[port_in]["y"],
                orientation,
                port_in,
                port_out,
                "electrical",
            ]
            rows.append(row)

        if optical_alignment_ports:
            row = [
                component,
                x + optical_alignment_ports[optical_alignment_ports_names[0]]["x"],
                y + optical_alignment_ports[optical_alignment_ports_names[0]]["y"],
                orientation,
                optical_alignment_ports_names[0],
                optical_alignment_ports_names[1],
                "optical",
            ]
            rows.append(row)

        for port_in in optical_component_ports:
            ports_out = set(optical_component_ports) - {port_in}
            for port_out in ports_out:
                row = [
                    component,
                    x + optical_component_ports[port_in]["x"],
                    y + optical_component_ports[port_in]["y"],
                    orientation,
                    port_in,
                    port_out,
                    "optical",
                ]
                rows.append(row)

    # Create a DataFrame from the collected rows
    return pd.DataFrame(
        rows,
        columns=[
            "component",
            "x",
            "y",
            "orientation",
            "port_in",
            "port_out",
            "port_type",
        ],
    )


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    c.show(show_ports=True)
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=("{",), layer_label="TEXT"
    )
    df_in = pd.read_csv(csvpath)

    # Initialize an empty list to collect the rows
    rows = []

    for _, label in df_in.iterrows():
        x, y, orientation = label.x, label.y, label.rotation
        x = int(x)
        y = int(y)
        text = label.text
        d = json.loads(text)
        electrical_ports = d.get("electrical_ports", {})
        optical_component_ports = d.get("optical_component_ports", {})
        optical_alignment_ports = d.get("optical_alignment_ports", {})
        component = d["name"] + f"_{x}_{y}"
        optical_alignment_ports_names = list(optical_alignment_ports.keys())

        for port_in, port_out in itertools.combinations(electrical_ports, 2):
            row = [
                component,
                x + electrical_ports[port_in]["x"],
                y + electrical_ports[port_in]["y"],
                orientation,
                port_in,
                port_out,
                "electrical",
            ]
            rows.append(row)

        if optical_alignment_ports:
            row = [
                component,
                x + optical_alignment_ports[optical_alignment_ports_names[0]]["x"],
                y + optical_alignment_ports[optical_alignment_ports_names[0]]["y"],
                orientation,
                optical_alignment_ports_names[0],
                optical_alignment_ports_names[1],
                "optical-alignment",
            ]
            rows.append(row)

        for port_in in optical_component_ports:
            ports_out = set(optical_component_ports) - {port_in}
            for port_out in ports_out:
                row = [
                    component,
                    x + optical_component_ports[port_in]["x"],
                    y + optical_component_ports[port_in]["y"],
                    orientation,
                    port_in,
                    port_out,
                    "optical",
                ]
                rows.append(row)

    # Create a DataFrame from the collected rows
    df = pd.DataFrame(
        rows,
        columns=[
            "component",
            "x",
            "y",
            "orientation",
            "port_in",
            "port_out",
            "port_type",
        ],
    )

    # df = write_test_manifest(csvpath)
    df.to_csv("test_manifest.csv", index=False)
