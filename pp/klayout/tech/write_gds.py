import pathlib

from pp.components import component_factory

if __name__ == "__main__":
    gdsdir = pathlib.Path(__file__).parent / "gds"
    gdsdir.mkdir(exist_ok=True)

    for factory in component_factory.values():
        c = factory()
        gdspath = gdsdir / (c.name + ".gds")
        c.write_gds(gdspath=gdspath)
