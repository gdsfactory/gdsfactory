"""Converts CSV of test site labels into a CSV test manifest."""


from collections import defaultdict

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def get_test_manifest(component: gf.Component) -> pd.DataFrame:
    """Returns a pandas DataFrame with test manifest.

    Args:
        component: component to extract test manifest from.
    """
    rows = []
    columns = [
        "cell",
        "cell_settings",
        "ports",
        "measurement",
        "measurement_settings",
        "analysis",
        "analysis_settings",
        "doe",
    ]
    ports = component.get_ports_list(sort_by_name=True)
    name_to_ports = defaultdict(dict)
    name_to_settings = {}

    for port in ports:
        port_settings = port.to_dict()

        p = port.name.split("-")
        port_name = p[-1]
        instance_name = "-".join(p[:-1])

        name_to_ports[instance_name][port_name] = {
            key: port_settings[key] for key in ["center", "orientation", "port_type"]
        }
        name_to_settings[instance_name] = component.info.get("components", {}).get(
            instance_name, {}
        )

    for name, d in name_to_settings.items():
        row = [
            name,
            d,
            name_to_ports[name],
            d.get("info", {}).get("measurement", None),
            d.get("info", {}).get("measurement_settings", None),
            d.get("info", {}).get("analysis", None),
            d.get("info", {}).get("analysis_settings", None),
            d.get("info", {}).get("doe", None),
        ]
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=columns,
    )


if __name__ == "__main__":
    c = sample_reticle(grid=True)
    # c = c.mirror()
    c.show(show_ports=False)
    df = get_test_manifest(c)
    df.to_csv("test_manifest.csv")
    print(df["ports"][0])
    print(df)
