from gdsfactory.simulation.modes.find_neff_ng_dw_dh import find_neff_ng_dw_dh


def test_dw_dh(dataframe_regression):
    df = find_neff_ng_dw_dh(steps=1, resolution=10)
    if dataframe_regression:
        dataframe_regression.check(df)
    else:
        print(df)


if __name__ == "__main__":
    test_dw_dh(None)
