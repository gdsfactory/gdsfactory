import pathlib
import pp
from pp.components import component_factory

gdsdir = pathlib.Path(__file__).parent / "gds"
gdsdir.mkdir(exist_ok=True)

for cf in component_factory.values():
    c = cf()
    gdspath = gdsdir / (c.name + ".gds")
    pp.write_gds(c, gdspath=gdspath)
