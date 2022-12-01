from __future__ import annotations

import gdsfactory.simulation.modes as gm


def test_neff_vs_width(dataframe_regression) -> None:
    df = gm.find_neff_vs_width(steps=1, resolution=10, cache=None)
    if dataframe_regression:
        dataframe_regression.check(df)
    else:
        print(df)


if __name__ == "__main__":
    test_neff_vs_width(None)
