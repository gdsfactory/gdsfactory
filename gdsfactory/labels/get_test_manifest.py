"""Converts CSV of test site labels into a CSV test manifest."""

import pandas as pd

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def get_test_manifest(
    component: gf.Component, one_setting_per_column: bool = True
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame with test manifest.

    Args:
        component: Component to extract test manifest from.
        one_setting_per_column: If True, puts each cell setting in a separate column.
    """
    rows = []
    ports = component.get_ports_list(sort_by_name=True)
    name_to_settings = {}

    for port in ports:
        try:
            instance_name = port.info.parent
        except AttributeError:
            raise AttributeError(f"port {port.name} has no `parent` in info dict.")
        name_to_settings[instance_name] = port.info.model_dump()

    if one_setting_per_column:
        # Gather all unique settings keys
        all_settings_keys = {key for d in name_to_settings.values() for key in d.keys()}
        columns = [
            "cell",
            "measurement",
            "measurement_settings",
            "analysis",
            "analysis_settings",
            "doe",
        ] + list(all_settings_keys)

        for name, d in name_to_settings.items():
            row = [
                name,
                d.get("measurement", None),
                d.get("measurement_settings", None),
                d.get("analysis", None),
                d.get("analysis_settings", None),
                d.get("doe", None),
            ] + [d.get(setting, None) for setting in all_settings_keys]
            rows.append(row)

    else:
        columns = [
            "cell",
            "measurement",
            "measurement_settings",
            "analysis",
            "analysis_settings",
            "doe",
            "cell_settings",
        ]

        for name, d in name_to_settings.items():
            row = [
                name,
                d.get("measurement", None),
                d.get("measurement_settings", None),
                d.get("analysis", None),
                d.get("analysis_settings", None),
                d.get("doe", None),
                d,
            ]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    c = sample_reticle(grid=False)
    c = gf.pack([c])[0]
    c.show(show_ports=False)
    df = get_test_manifest(c, one_setting_per_column=False)
    df.to_csv("test_manifest.csv", index=False)
    print(df)
