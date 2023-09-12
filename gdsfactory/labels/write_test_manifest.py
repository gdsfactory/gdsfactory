"""Converts CSV of test site labels into a CSV test manifest."""

import json
import pathlib

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def write_test_manifest(csv_in: str | pathlib.Path) -> pd.DataFrame:
    """Converts CSV of test site labels into a CSV test manifest."""
    pd.read_csv(csv_in)
    pass


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    c.show(show_ports=True)
    gdspath = c.write_gds()
    csvpath = gf.labels.write_labels.write_labels_gdstk(
        gdspath, prefixes=("{",), layer_label="TEXT"
    )
    df_in = pd.read_csv(csvpath)
    df = pd.DataFrame(
        columns=[
            "component",
            "x",
            "y",
            "orientation",
            "port_in",
            "port_out",
            "port_type",
        ]
    )

    i = 0
    for _, label in df_in.iterrows():
        x, y, orientation = label.x, label.y, label.rotation
        text = label.text
        d = json.loads(text)
        electrical_ports = d.get("electrical_ports", {})
        optical_ports = d.get("optical_ports", {})

        component = d["name"] + f"_{int(x)}_{int(y)}"

        for port_name, _port in electrical_ports.items():
            if d["with_loopback"]:
                df.iloc[i] = [component, x, y, orientation, port_name, "", False, False]
            i += 1

        # for port_name, port in electrical_ports.items():
        #     row = pd.DataFrame(
        #         {
        #             "component": d["name"] + f"_{int(x)}_{int(y)}",
        #             "x": port["x"] + x,
        #             "y": port["y"] + y,
        #             "orientation": port["orientation"],
        #             "port_name": port_name,
        #         }
        #     )
        #     df = pd.concat([df, row], ignore_index=True)

    # df = write_test_manifest(csvpath)
    # df.to_csv("test_manifest.csv", index=False)
