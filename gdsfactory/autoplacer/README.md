# Klayout placer

This klayout placer is quite fast when dealing with GDS files.

You have different options for putting together a mask:

1. python code (gf.pack and gf.grid)
2. YAML for the netlist driven flow (see gf.read.from_yaml)
3. YAML for defining DOEs and using Klayout placer, which is getting deprecated
4. gf.placer, which is deprecated
