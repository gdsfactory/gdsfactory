from pp.port import csv2port


def test_csv2port(data_regression):
    import pp

    name = "waveguide"
    csvpath = pp.CONFIG["gdsdir"] / f"{name}.ports"
    ports = csv2port(csvpath)
    data_regression.check(ports)
