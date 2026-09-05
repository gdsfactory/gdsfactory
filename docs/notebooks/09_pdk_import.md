<!-- #region -->
# Import PDK

## Importing a PDK from GDS Files  

For foundry PDKs, we highly recommend using GDSFactory PDKs. [See available PDKs here](https://gdsfactory.com/).  

If your foundry does not yet have a GDSFactory PDK, you can import a PDK from GDS files into GDSFactory. To do this, you will need:  

- A GDS file containing all the cells you want to import into the PDK (or multiple GDS files, each containing a separate design).  

Ideally, you should also obtain:  

- **KLayout layer properties files** – to define the layers available for custom components. This allows you to create a **LayerMap** that associates layer names with `(GDS_LAYER, GDS_PURPOSE)`.  
- **Layer stack information** – including material index, thickness, and the Z positions of each layer.  
- **DRC rules** – If these are not provided, you can build them using KLayout.  


GDS files efficiently describe geometry using **References**, which store each geometry only once in memory. However, most GDS tools lack a clear standard for storing **device metadata** such as settings, port locations, port widths, and port angles.  


`gdsfactory` addresses this limitation by storing metadata inside the GDS and provide built-in functions to add pins.  

To save a GDS file in `gdsfactory`, use:  

```python
Component.write_gds()
```

However, if you want to avoid the port and settings metadata you can use:


```python
Component.write_gds(with_metadata=False)
```
<!-- #endregion -->

```python
import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.technology import lyp_to_dataclass

gf.gpdk.PDK.activate()
```

```python
c = gf.components.mzi()
c.plot()
```

You can write **GDS**  files.

```python
gdspath = c.write_gds("extra/mzi.gds")
```

```python
c.pprint_ports()
```

You can import GDS files into gdsfactory thanks to the `import_gds` function.

`import_gds` reads the same GDS file from the disk without losing any information. 

Gdsfactory stores the settings and the ports as part of the GDS metadata.

```python
c = gf.import_gds(gdspath)
c.pprint_ports()
c.plot()
```

<!-- #region -->
### Add ports from pins

Sometimes the GDS does not have YAML metadata, therefore you need to figure out the port locations, widths and orientations.

gdsfactory provides you with functions that will add ports to the component by looking for the shapes of pins on a specific layer (port_markers or pins).

There are different pin standards supported to automatically add ports to components:

- PINs towards the inside of the port (port at the outer part of the PIN).
- PINs with half of the pin inside and half outside (port at the center of the PIN).
- PIN with only labels (no shapes). You have to manually specify the width of the port.


Now let us save a GDS and then import it back.
<!-- #endregion -->

```python
c = gf.components.straight()
c_with_pins = gf.add_pins.add_pins_container(component=c)
c_with_pins.plot()
```

```python
c_with_pins.ports = []  # Let us rely on the pins to extract the ports.
gdspath = c_with_pins.write_gds("extra/wg.gds")
```

```python
c2 = gf.import_gds(gdspath)
c2.plot()
```

```python
c2.ports  # Import_gds does not automatically add the pins.
```

```python
# This line reads a GDSII file from the location specified by the gdspath variable and converts its geometry into a gdsfactory component.
c3 = gf.import_gds(gdspath)

# This is a utility function that adds ports to a component by looking for special marker shapes.
# It searches the component for small shapes (often squares or paths) on the specified pin_layer (in this case, a layer named "PORT").
# It then uses the location, width, and orientation of these markers to create and add proper gdsfactory ports to the component.
c3 = gf.add_ports.add_ports_from_markers_inside(c3, pin_layer="PORT")
c3.plot()
```

```python
c3.pprint_ports()
```

<!-- #region -->
Foundries provide PDKs in different formats and commercial tools.

The easiest way to import a PDK into gdsfactory is to:

1. have each GDS cell import into a separate GDS file.
2. have one GDS file with all the cells inside.
3. Have a KLayout layermap. Makes it easier to create the layermap.

With that you can easily create the PDK as as python package.

Thanks to having a gdsfactory PDK as a python package you can:

- Version control your PDK using GIT to keep track of changes and work on it as a team.
    - Write tests of your pdk components to avoid unwanted changes from one component to another.
    - Ensure you maintain the quality of the PDK with continuous integration checks.
    - Pin the version of gdsfactory, so new updates of gdsfactory will not affect your code.
- Name your PDK version using [semantic versioning](https://semver.org/). For example patches increase the last number (0.0.1 -> 0.0.2).
- Install your PDK easily `pip install pdk_fab_a` and easily access an interface with other tools.



To create a **Python** package you can start from a customizable template (thanks to cookiecutter).

You can create a python package by running these 2 commands inside a terminal:

```
pip install cookiecutter
cookiecutter gh:joamatab/python
```

It will ask you some questions to fill into the template for the python package.


Then you can add the information about the GDS files and the Layers inside that package.
<!-- #endregion -->

```python
# lyp_to_dataclass(...): This gdsfactory utility function parses the .lyp file and converts its information into the text format of a Python dataclass.
print(lyp_to_dataclass(PATH.klayout_lyp))
```

```python
# Let us now create a sample PDK (for demo purposes only) using GDSfactory.
# If the PDK is in a commercial tool you can also do this. Make sure you save a single pdk.gds.

sample_pdk_cells = gf.grid(
    (
        gf.components.straight,
        gf.components.bend_euler,
        gf.components.grating_coupler_elliptical,
    )
)
sample_pdk_cells.write_gds("extra/pdk.gds")
sample_pdk_cells
```

```python
gf.clear_cache()

# write_cells_recursively(...): This function recursively goes through the entire hierarchy of the input GDSII file,
# finds every unique cell definition, and writes each one to a new .gds file in the output directory.
# gdspath="extra/pdk.gds": This is the input file. It is a GDSII file that contains a full design or Process Design Kit (PDK),
# which is typically a hierarchical structure of many cells referencing other cells.
# # dirpath="extra/gds": This is the output directory. The function will create this folder if it does not exist.
gf.write_cells.write_cells_recursively(gdspath="extra/pdk.gds", dirpath="extra/gds")
```

```python
print(gf.write_cells.get_import_gds_script("extra/gds"))
```

You can also include the code to plot each fix cell in the docstring.

```python
print(gf.write_cells.get_import_gds_script("extra/gds", module="samplepdk.components"))
```

## Import PDK from other python packages

You can Write the cells into GDS and use the

Ideally you also start transitioning your legacy code Pcells into gdsfactory syntax. It is a great way to learn the gdsfactory way!

Here is some advice:

- Ask your foundry for the gdsfactory PDK.
- Leverage the generic pdk cells available in gdsfactory.
- Write tests for your cells.
- Break the cells into small reusable functions.
- Use GIT to track changes.
- Review your code with your colleagues and other gdsfactory developers to get feedback. This is key to get better at coding in gdsfactory.
- Get rid of any warnings you see.

<!-- #region -->
## Import PDK from YAML uPDK

gdsfactory supports read and write to [uPDK YAML definition](https://openepda.org/index.html)

Let us write a PDK into uPDK YAML definition and then convert it back to a gdsfactory script.

the uPDK extracts the code from the docstrings.

```python

def evanescent_coupler_sample() -> None:
    """Evanescent coupler example.

    Args:
      coupler_length: length of coupling (min: 0.0, max: 200.0, um).
    """
    pass

```
<!-- #endregion -->

```python
from gdsfactory.samples.pdk.fab_c import PDK

PDK.activate()

# The .to_updk() method converts the PDK's settings into a YAML string that follows the universal PDK (uPDK) standard.
# The uPDK is an open-source format designed to make PDKs interoperable between different design and simulation tools.
yaml_pdk = PDK.to_updk()
print(yaml_pdk)
```


```python
from gdsfactory.component import GDSDIR_TEMP
from gdsfactory.read.from_updk import from_updk

yamlpath = GDSDIR_TEMP / "pdk.yml"
yamlpath.write_text(yaml_pdk)
gdsfactory_script = from_updk(yamlpath)
print(gdsfactory_script)
```

<!-- #region -->
## Build your own PDK

You can create a PDK as a python library using a cookiecutter template. For example, you can use this one:

```
pip install cookiecutter
cookiecutter gh:joamatab/python
```

Or you can fork the ubcpdk and create new PCell functions that use the correct layers for your foundry. For example:

```python

from gdsfactory.technology import LayerMap


class LayerMap(LayerMap):
    WGCORE = (3, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)  # PinRec
    PORTE: Layer = (1, 11)  # PinRecM
    FLOORPLAN: Layer = (99, 0)

    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    LABEL_INSTANCE: Layer = (66, 0)


LAYER = LayerMap

```
<!-- #endregion -->
