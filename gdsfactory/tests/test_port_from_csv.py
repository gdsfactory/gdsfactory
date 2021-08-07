from gdsfactory.port import csv2port


def test_csv2port(data_regression):
    import gdsfactory as gf

    name = "straight"
    csvpath = gf.CONFIG["gdsdir"] / f"{name}.ports"
    ports = csv2port(csvpath)
    data_regression.check(ports)
