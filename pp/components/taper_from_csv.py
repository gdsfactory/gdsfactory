""" adiabatic tapers from CSV files
"""
import pathlib

import pp

data_path = pathlib.Path(__file__).parent / "csv_data"


def taper_from_csv(csv_path, wg_layer=1, clad_offset=3, clad_layer=pp.LAYER.WGCLAD):
    taper_data = pp.load_csv(csv_path)
    # taper_data = pd.read_csv(csv_path)
    # print(taper_data)
    xs = taper_data["x"] * 1e6
    ys = taper_data["width"] * 1e6 / 2.0
    ys_trench = ys + clad_offset

    c = pp.Component()
    c.add_polygon(list(zip(xs, ys)) + list(zip(xs, -ys))[::-1], layer=wg_layer)
    c.add_polygon(
        list(zip(xs, ys_trench)) + list(zip(xs, -ys_trench))[::-1], layer=clad_layer
    )

    c.add_port(
        name="W0",
        midpoint=(xs[0], 0),
        width=2 * ys[0],
        orientation=180,
        port_type="optical",
    )
    c.add_port(
        name="E0",
        midpoint=(xs[-1], 0),
        width=2 * ys[-1],
        orientation=0,
        port_type="optical",
    )
    return c


@pp.autoname
def taper_w3_l36(**kwargs):
    csv_path = data_path / "taper_strip_0p5_3_36.csv"
    return taper_from_csv(csv_path, **kwargs)


@pp.autoname
def taper_w10_l100(**kwargs):
    csv_path = data_path / "taper_strip_0p5_10_100.csv"
    return taper_from_csv(csv_path, **kwargs)


@pp.autoname
def taper_w10_l150(**kwargs):
    csv_path = data_path / "taper_strip_0p5_10_150.csv"
    return taper_from_csv(csv_path, **kwargs)


@pp.autoname
def taper_w10_l200(**kwargs):
    csv_path = data_path / "taper_strip_0p5_10_200.csv"
    return taper_from_csv(csv_path, **kwargs)


@pp.autoname
def taper_w11_l200(**kwargs):
    csv_path = data_path / "taper_strip_0p5_11_200.csv"
    return taper_from_csv(csv_path, **kwargs)


@pp.autoname
def taper_w12_l200(**kwargs):
    csv_path = data_path / "taper_strip_0p5_12_200.csv"
    return taper_from_csv(csv_path, **kwargs)


if __name__ == "__main__":
    c = taper_w3_l36()
    # c = taper_w10_l100()
    # c = taper_w11_l200()
    pp.show(c)
