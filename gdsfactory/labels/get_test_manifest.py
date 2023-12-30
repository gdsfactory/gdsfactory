"""Converts CSV of test site labels into a CSV test manifest."""

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def get_test_manifest(
    component: gf.Component,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with test manifest.

    Args:
        component: component to extract test manifest from.
    """
    rows = []
    columns = [
        "cell",
        "cell_settings",
        "measurement",
        "measurement_settings",
        "analysis",
        "analysis_settings",
        "doe",
    ]
    ports = component.get_ports_list(sort_by_name=True)
    name_to_settings = {}

    for port in ports:
        try:
            instance_name = port.info.parent
        except AttributeError:
            raise AttributeError(f"port {port.name} has no `parent` in info dict.")
        name_to_settings[instance_name] = port.info

    for name, d in name_to_settings.items():
        row = [
            name,
            d,
            d.get("measurement", None),
            d.get("measurement_settings", None),
            d.get("analysis", None),
            d.get("analysis_settings", None),
            d.get("doe", None),
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
    print(df)
