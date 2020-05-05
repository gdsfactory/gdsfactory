import pathlib
import pp
from pp.components import component_type2factory

gdsdir = pathlib.Path(__file__).parent / "gds"
gdsdir.mkdir(exist_ok=True)

for cf in component_type2factory.values():
    c = cf()
    gdspath = gdsdir / (c.name + ".gds")
    pp.write_gds(c, gdspath=gdspath)
