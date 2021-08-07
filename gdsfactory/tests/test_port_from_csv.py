from gdsfactory.port import csv2port


def test_csv2port(data_regression):
    import gdsfactory

    name = "straight"
    csvpath = gdsfactory.CONFIG["gdsdir"] / f"{name}.ports"
    ports = csv2port(csvpath)
    data_regression.check(ports)
