from gdsfactory.simulation.modes.find_neff_ng_dw_dh import find_neff_ng_dw_dh


def test_dw_dh(dataframe_regression):
    df = find_neff_ng_dw_dh(steps=3)
    dataframe_regression.check(df)


if __name__ == "__main__":
    test_dw_dh()
