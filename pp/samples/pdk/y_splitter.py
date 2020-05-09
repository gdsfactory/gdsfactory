import pp

from import_gds import import_gds


def y_splitter():
    c = import_gds("ebeam_y_1550")
    return c


if __name__ == "__main__":
    c = y_splitter()
    pp.show(c)
