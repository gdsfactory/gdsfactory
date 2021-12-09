import gdsfactory.simulation.modes as gm


def test_neff_vs_width(dataframe_regression):
    df = gm.find_neff_vs_width(steps=3)
    dataframe_regression.check(df)


if __name__ == "__main__":
    test_neff_vs_width()
