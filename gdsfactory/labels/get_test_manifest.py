"""Converts CSV of test site labels into a CSV test manifest."""

import gdsfactory as gf
from gdsfactory.samples.sample_reticle import sample_reticle


def get_test_manifest(component: gf.Component, one_setting_per_column: bool = True):
    """Returns a pandas DataFrame with test manifest.

    Args:
        component: Component to extract test manifest from.
        one_setting_per_column: If True, puts each cell setting in a separate column.
    """
    import pandas as pd

    rows = []
    ports = component.get_ports_list()
    name_to_settings = {port.name: port.info for port in ports}

    if one_setting_per_column:
        # Gather all unique settings keys
        all_settings_keys = {key for d in name_to_settings.values() for key in d}
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
    c = sample_reticle()
    # c = gf.pack([c])[0]
    c.show()
    df = get_test_manifest(c)
    df.to_csv("test_manifest.csv", index=False)
    print(df)
