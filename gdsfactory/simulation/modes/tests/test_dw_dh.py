from __future__ import annotations

from gdsfactory.simulation.modes.find_neff_ng_dw_dh import find_neff_ng_dw_dh


def test_dw_dh(dataframe_regression) -> None:
    df = find_neff_ng_dw_dh(steps=1, resolution=10, cache=None)
    if dataframe_regression:
        dataframe_regression.check(df, default_tolerance=dict(atol=1e-2, rtol=1e-2))
    else:
        print(df)


if __name__ == "__main__":
    test_dw_dh(None)
