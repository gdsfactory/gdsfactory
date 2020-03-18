# Klayout YAML placer

- It assumes that each DOE is provided in a folder together with a text file containing the list of each GDS file. (pp.placer does that)
- The placing instructions are taken from a YAML file. Typically the same as the one
  used for specifying the DOEs

The YAML Placer addresses the following requirements:

- Fast mask assembly
- Absolute or relative placement
- Adjustable margins between DOEs and within a DOE
- multirow / multi-column packing

The YAML Placer does not check for collisions and does not guarantee a valid placing.
It is a tool to put together a mask and rapidly iterate whether floorplan changes or
as new devices or DOEs are added.

```eval_rst
.. autofunction:: pp.autoplacer.place_from_yaml
```

## Workflow

While the YAML placer can work as a standalone tool, it is designed to be used with gds_factory (pp)
A single YAML file is used for both DOE specification and placing instructions.

In a typical workflow, this file is parsed twice:
The first time, by `generate_does` from pp library
The second time, by `place_from_yaml` from the YAML placer to read the placing instructions for every DOE

Example:

```
import os
import pp
from pp.config import CONFIG
from pp.generate_does import generate_does
from pp.autoplacer.yaml_placer import place_from_yaml


# =========
# GDS factories used for the Devices/DOEs
# =========
from chipedge import CHIPEDGE_ILOTS
from ilots import ILOT01
from coupler import CP2x2

factories = [
    CHIPEDGE_ILOTS,
    ILOT01,
	CP2x2,
]

# This maps the factory names for the factories, enabling them to be used in the YAML file
component_type2factory = {f.__name__ : f for f in factories}

def top_level(mask_name="TEG_ILOT_PCM"):
    print(CONFIG["cache_doe_directory"])
    c = pp.Component()

    folder = os.path.dirname(os.path.abspath(__file__))
    filepath_yml = os.path.join(folder, "{}.yml".format(mask_name))
    generate_does(filepath_yml, component_type2factory=component_type2factory)

    top_level = place_from_yaml(
        filepath_yml,
        root_does=CONFIG["cache_doe_directory"]
    )

    filepath_gds = os.path.join(CONFIG["mask_directory"], "{}.gds".format(mask_name))
    top_level.name = mask_name
    top_level.write(filepath_gds)
    return filepath_gds


if __name__ == "__main__":
    gdspath = top_level()
    pp.show(gdspath)

```

Corresponding YAML

```
mask:
    width: 24090
    height: 4500
    name: TEG_ILOT_PCM
    layer_doe_label: [102, 6]

	# Setting cache to `true`: By default, all generated GDS are cached and won't be regenerated
	# This default behaviour can be overwritten within each DOE.

	# To rebuild the full mask from scratch, just set this to `false`, and ensure there is no
	# cache: true specified in any other component
    cache: true


## =======================================================================
## Templates - DOEs which refer to templates inherit the template settings
## =======================================================================
template_east_align_S:
    type: template
    placer:
        type: pack_row
        x0: E
        y0: S
        align_x: W
        align_y: S
        margin: 0

template_south_align_W:
    type: template
    placer:
        x0: W
        y0: S
        align_x: W
        align_y: N
        margin: 0

ilot_template:
    type: template
    add_doe_label: true
    with_doe_name: false

CHIPEDGE_ILOTS:
    component: CHIPEDGE_ILOTS
    settings:
        width: 24090
        height: 4500
    placer:
        x0: 0
        y0: 0

## =============================
## ILOTS
## =============================
ILOT01:
    component: ILOT01
    template: ilot_template
	cache: false # Assuming I am working on this ILOT, it is convenient to
		# set cache to false here. The full mask rebuilds quickly thanks to
		# the default being true for all other DOEs, but all the changes to
		# this ILOT are captured at every iteration

    placer:
		# Absolute coordinate placing
        x0: 0
        y0: 4500

		# The west side of the device is aligned to x0
		# The north side of the device is aligned to y0
        align_x: W
        align_y: N

COUPLERS:

	component: CP2x2 # Using CP2x2 factory, generate the DOE with

	# Uses the combination of settings to produce 9 devices
	settings:
		gap: [0.3, 0.4, 0.5]
		length: [0.1, 5., 10.2]

	placer:

		type: pack_col 	# These devices are packed in a column
		# Relative placing: this DOE is using the South East of the previous DOE as the origin
		x0: E
		y0: S

		# The West side of the device is aligned to x0 + margin
		# The South side of the device is aligned to y0 + margin

		align_x: W
		align_y: S
		margin: 10. # 10um between every device
```

## Placer arguments

```


DOE003:
	component: MMI1x2
	settings:
		L: [5, 10, 15]
	placer:
		type: pack_row / pack_col # placer type
		row_ids / col_ids: list of integers, if specified, should have the same length as the total number of components within the DOE
			by defaults all devices are in the first column/row (index 0)
			If we want multiple rows, columns, we need to specify in which column they each go
			e.g [0, 0, 1]

		x0: <float> /  `E` / `W`
		y0: <float> / `S` / `N`
		align_x: `E` / `W`
		align_y: `S` / `N`
		margin: <float>
		margin_x: <float>
		margin_y: <float>
		inter_margin_x: <float>
		inter_margin_y: <float>
		next_to: <DOE_NAME>

```
