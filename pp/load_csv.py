import csv

import numpy as np


def load_csv(csv_path):
    """ loads csv and returs a dict
    the key for each column is taken from the first row
    """
    dict_data = {}
    with open(csv_path, "r", encoding="utf-8-sig") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        fields = next(csv_reader)
        dict_data = {field: [] for field in fields}
        for row in csv_reader:
            for i, x in enumerate(row):
                field = fields[i]
                dict_data[field].append(float(x))

    dict_data = {k: np.array(v) for k, v in list(dict_data.items())}
    return dict_data


def load_csv_pandas(csv_path):
    import pandas as pd

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    from pp.config import CONFIG

    csv_path = CONFIG["components_path"] / "csv_data" / "taper_strip_0p5_3_36.csv"

    d = load_csv_pandas(csv_path)
    print(d)
