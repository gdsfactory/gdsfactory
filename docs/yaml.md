# YAML

You can define components, netlists (for circuits) and masks in YAML syntax.

## Netlist


```{eval-rst}

.. autofunction:: gdsfactory.read.from_yaml
```


## Mask

- It assumes that each DOE is provided in a folder together with a text file containing the list of each GDS file. (gdsfactory.placer)
- The placing instructions are taken from a YAML file. Typically the same as the one used for specifying the DOEs

The YAML Placer addresses the following requirements:

- Fast mask assembly
- Absolute or relative placement
- Adjustable margins between DOEs and within a DOE
- multirow / multi-column packing

The YAML Placer does not check for collisions and does not guarantee a valid placing.
It just puts together a mask allows you to change components placements.


You can specify DOEs (Design Of Experiments) and placement in a single YAML file.

In a typical workflow, this file is parsed twice:
The first time, `generate_does` builds the GDS files for each Design of Experiment variation.
The second time, `place_from_yaml` reads the placing instructions for every DOE and places them in a new GDS file.

Example:

```

import pathlib
import gdsfactory as gf
from gdsfactory.autoplacer.yaml_placer import place_from_yaml
from gdsfactory.generate_does import generate_does
from gdsfactory.mask.merge_metadata import merge_metadata


def test_mask():
    """Returns gdspath for a Mask

    - Write GDS files defined in does.yml (with JSON metadata)
    - place them into a mask following placer information in does.yml
    - merge mask JSON metadata into a combined JSON file

    """
    cwd = pathlib.Path.cwd()
    does_path = cwd / "does.yml"
    doe_root_path = cwd / "build" / "cache_doe_directory"
    mask_path = cwd / "build" / "mask"
    gdspath = mask_path / "mask.gds"
    mask_path.mkdir(parents=True, exist_ok=True)

    generate_does(
        str(does_path), doe_root_path=doe_root_path,
    )
    top_level = place_from_yaml(does_path, root_does=doe_root_path)
    top_level.write(str(gdspath))
    merge_metadata(gdspath)
    return gdspath


if __name__ == "__main__":
    gdspath = test_mask()
    gf.show(gdspath)

```

Corresponding YAML

```yaml

mask:
    width: 10000
    height: 10000
    name: mask2

    # Setting cache to `true`: By default, all generated GDS are cached and won't be regenerated
    # This default behaviour can be overwritten within each DOE.
    # To rebuild the full mask from scratch, just set this to `false`, and ensure there is no
    # cache: true specified in any other component
    cache: true

## =======================================================================
## Templates - global settings for DOEs (Optional)
## =======================================================================
template_align_east_south:
    type: template
    placer:
        type: pack_row
        x0: E
        y0: S
        align_x: W
        align_y: S
        margin: 0

template_align_south_west:
    type: template
    placer:
        x0: W
        y0: S
        align_x: W
        align_y: N
        margin: 0

template_add_labels:
    type: template
    add_doe_label: true
    with_doe_name: false

## =============================
## Does (Design Of exeriments)
## =============================

mmi2x2_width:
    component: mmi2x2
    settings: # Uses the combination of settings to produce 9 devices
        width_mmi: [4.5, 5.6]
        length_mmi: 10
    placer:
        type: pack_row
        x0: 0 # Absolute coordinate placing
        y0: 0
        align_x: W # x origin is west
        margin: 25. # x and y margin between the components within this DOE
        align_y: S # y origin is south

mmi1x2_width_length:
    component: mmi1x2
    do_permutation: False
    settings:
        length_mmi: [10, 20]
        width_mmi: [5, 10]

    placer:
        type: pack_row
        next_to: mmi2x2_width
# Relative placing: this DOE is using the West South of the previous DOE as the origin
        x0: W # x0 is the west of the DOE specified in next_to
        y0: S # y0 is the south of the DOE specified in next_to
# The West side of the component is aligned to x0 + margin
# The North side of the component is aligned to y0 + margin
        align_x: W
        align_y: N
        inter_margin_y: 100 # y margin between this DOE and the one used for relative placement
        margin_x: 50. # x margin between the components within this DOE
        margin_y: 20. # y margin between the components within this DOE

bend_south_west:
    component: bend_circular
    template: template_align_south_west
    settings:
        radius: [5, 10]
    placer:
        type: pack_col # These devices are packed in a column

ring_with_labels:
    component: ring_single
    template: template_add_labels
    cache: False
    settings:
        bend_radius: [5, 10]
    placer:
        next_to: mmi1x2_width_length
# Assuming I am working on this DOE, it is convenient to set cache to false here.
# The full mask rebuilds quickly thanks to
# the default cahe=true for all other DOEs, but all the changes to
# this DOE are captured at every iteration
```

**Placer arguments**

```


DOE003:
	component: mmi1x2
	settings:
		length: [5, 10, 15]
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
