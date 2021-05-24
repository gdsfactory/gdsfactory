Now that you have installed gdsfactory lets create your first component.

You'll need to keep 3 windows open:

- A text editor or IDE (Visual Studio Code, Pycharm, Spyer, neovim, Atom, Jupyterlab ...)
- A python / Ipython terminal / jupyter notebook (interactive python to run).
- Klayout to Visualize the GDS files.

![windows](https://i.imgur.com/DyVL6IE.png)

# Component

A Component contains:

- a list of elements
  - add_polygon
    - boundary: defines the area inside a shape's points (requires shape and layer)
    - path: defines a line with a certain width combining a shape’s points (requires shape, layer and a line_width)
  - add_ref (single reference)
  - add_array (array of references)
- a dictionary of ports
  - add_port(): adds a port to the dictionary
- convenient methods:
  - write_gds(): saves to GDS

![gds](images/gds.png)

The first thing to learn about is how to create a new component.
We do that by creating a function which returns a pp.Component instance.
Here is a step by step example below generating a waveguide crossing

```eval_rst

.. plot::
    :include-source:

    import pp


    @pp.cell
    def crossing_arm(wg_width=0.5, r1=3.0, r2=1.1, w=1.2, L=3.4):
        c = pp.Component()

        # We need an ellipse, this is an existing primitive
        c << pp.components.ellipse(radii=(r1, r2), layer=pp.LAYER.SLAB150)

        a = L + w / 2
        h = wg_width / 2

        # Generate a polygon from scratch
        taper_pts = [
            (-a, h),
            (-w / 2, w / 2),
            (w / 2, w / 2),
            (a, h),
            (a, -h),
            (w / 2, -w / 2),
            (-w / 2, -w / 2),
            (-a, -h),
        ]

        # Add the polygon to the component on a specific layer
        c.add_polygon(taper_pts, layer=pp.LAYER.WG)

        # Add ports (more on that later)
        c.add_port(
            name="W0", midpoint=(-a, 0), orientation=180, width=wg_width, layer=pp.LAYER.WG
        )
        c.add_port(
            name="E0", midpoint=(a, 0), orientation=0, width=wg_width, layer=pp.LAYER.WG
        )
        return c


    @pp.port.deco_rename_ports  # This decorator will auto-rename the ports
    @pp.cell  # This decorator will generate a good name for the component
    def crossing():
        c = pp.Component()
        arm = crossing_arm()

        # Create two arm references. One has a 90Deg rotation
        arm_h = arm.ref(position=(0, 0))
        arm_v = arm.ref(position=(0, 0), rotation=90)

        # Add each arm to the component
        # Also add the ports
        port_id = 0
        for ref in [arm_h, arm_v]:
            c.add(ref)
            for p in c.ports.values():
                # Here we don't care too much about the name we give to the ports
                # since they will be renamed. We just want the names to be unique
                c.add_port(name="{}".format(port_id), port=p)
                port_id += 1

        return c


    c = crossing()
    c.plot()

```

## Types


What types of data do you have in gdsfactory?
Where is the gdsfactory name come from?


```eval_rst
.. automodule:: pp.types
```


## Layers

Each foundry uses different GDS numbers for each process step.

We follow the generic layer numbers from the book "Silicon Photonics Design: From Devices to Systems Lukas Chrostowski, Michael Hochberg". See `pp.LAYER`

| GDS (layer, purpose) | layer_name | Description                                                 |
| -------------------- | ---------- | ----------------------------------------------------------- |
| 1 , 0                | WG         | 220 nm Silicon core                                         |
| 2 , 0                | SLAB150    | 150nm Silicon slab (70nm shallow Etch for grating couplers) |
| 3 , 0                | SLAB90     | 90nm Silicon slab (for modulators)                          |
| 47, 0                | MH         | heater                                                      |
| 41, 0                | M1         | metal 1                                                     |
| 45, 0                | M2         | metal 2                                                     |
| 40, 0                | VIA1       | VIA1                                                        |
| 44, 0                | VIA2       | VIA2                                                        |
| 46, 0                | PADOPEN    | Bond pad opening                                            |
| 51, 0                | UNDERCUT   | Undercut                                                    |
| 52, 0                | DEEPTRENCH | Deep trench                                                 |
| 66, 0                | TEXT       | Text markup                                                 |
| 64, 0                | FLOORPLAN  | Mask floorplan                                              |

Layers are available in `pp.LAYER` as `pp.LAYER.WG`, `pp.LAYER.WGCLAD`

You can build PDKs for different foundries using gdsfactory, the PDKs contain some foundry IP such as layer numbers, minimum CD, layer stack, so you need to keep them in a separate private repo. See [UBC PDK](https://github.com/gdsfactory/ubc) as an example.

I recommend that you create the PDK repo using a cookiecutter template. For example, you can use this one.

```
pip install cookiecutter
cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal
```

Or you can fork the UBC PDK and create new cell functions that use the correct layers for your foundry. For example.

```

import dataclasses
import pp


@dataclasses.dataclass(frozen=True)
class Layer:
    WGCORE = (3, 0)
    LABEL = (100, 0)


LAYER = Layer()


```


## Port



You can define ports to:

 - facilitate positioning of components with respect to one another
 - connect components between each other using routing sub-routines
 - find ports by a particular layer or port name prefix



```eval_rst
.. plot::
    :include-source:

    import pp

    y = 0.5
    x = 2
    layer = (1, 0) # a GDS layer is a tuple of 2 integers
    c = pp.Component()
    c.add_polygon([(0, 0), (x, 0), (x, y), (0, y)], layer=layer)
    c.add_port(name='W0', midpoint=(0, y/2), width=y, orientation=180, layer=layer)
    c.plot()

```


```eval_rst
.. plot::
    :include-source:

    import pp

    coupler = pp.components.coupler()
    c = pp.Component()

    # Instantiate a reference to `_cpl`, positioning 'W0' port at coords (0, 0)
    coupler1 = coupler.ref(port_id='W0', position=(0, 0))

    # Instantiate another reference to `_cpl`, positioning 'W0' port at
    # the position of the 'E0' port from cpl1
    coupler2 = coupler.ref(port_id='W0', position=coupler1.ports['E0'].position)

    c.add(coupler1)
    c.add(coupler2)

    # add the ports of the child cells into the parent cell
    c.add_port(port=coupler1.ports['W0'])
    c.add_port(port=coupler1.ports['W1'])
    c.add_port(port=coupler2.ports['E0'])
    c.add_port(port=coupler2.ports['E1'])
    c.plot()

```


`pp.Component.ref` also accepts arguments such as:

 - `h_mirror` (True / False),
 - `v_mirror` (True / False)
 - `rotation` (0 / 90 / 180 / 270)

 They implement the transformation w.r.t the port position given by port_id If no port_id is given, transformation is done w.r.t (0,0)


Ports can have flexible labelling and by default, the user chooses how to label the ports
in the component with the constraint of giving name unique names within this component.

A function `auto_rename_ports` is provided to automatically label ports according to the following convention:

 - photonics ports are labelled with strings that refer to the orientation
 - other ports are labelled from a prefix and numbered counter-clockwise starting from the east port
