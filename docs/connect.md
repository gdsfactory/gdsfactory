# Connect

## References

You can connect:

- As a component sequence
- Defining a netlist
- Connecting the references butt to butt

![component levels](images/lib_example.png)


```eval_rst

.. plot::
    :include-source:

    import gdsfactory as gf

    @gf.cell
    def ring(
        coupler90=gf.components.coupler90,
        coupler_straight=gf.components.coupler_straight,
        straight=gf.components.straight,
        bend=gf.components.bend_euler,
        length_y=2.0,
        length_x=4.0,
        gap=0.2,
    ):
        """ single bus ring
        """
        c = gf.Component()

        # define subcells
        coupler90 = gf.call_if_func(coupler90, gap=gap)
        straight_x = gf.call_if_func(straight, length=length_x)
        straight_y = gf.call_if_func(straight, length=length_y)
        bend = gf.call_if_func(bend)
        coupler_straight = gf.call_if_func(coupler_straight, gap=gap, length=length_x)

        # add references to subcells
        cbl = c << coupler90
        cbr = c << coupler90
        cs = c << coupler_straight
        wyl = c << straight_y
        wyr = c << straight_y
        wx = c << straight_x
        btl = c << bend
        btr = c << bend


        # connect references
        wyr.connect(port='o2', destination=cbr.ports['o2'])
        cs.connect(port='o2', destination=cbr.ports['o1'])

        cbl.reflect(p1=(0, coupler90.y), p2=(1, coupler90.y))
        cbl.connect(port='o1', destination=cs.ports['o1'])
        wyl.connect(port='o2', destination=cbl.ports['o2'])

        btl.connect(port='o2', destination=wyl.ports['o1'])
        btr.connect(port='o1', destination=wyr.ports['o1'])
        wx.connect(port='o1', destination=btl.ports['o1'])
        return c


    c = ring()
    c.show()
    c.plot()
```


## Netlist

- Level 0 components have geomtry and port information
- Level 1 components are defined by netlist connectivity of level 0 components


Sometimes, when a component is mostly composed of sub-components adjacent to each
other, it can be easier to define the component by only saying what the
sub-component are, how they are connected, and which ports are part of the
new components.

This can be done using a netlist based approach where these 3 parts are defined:

- components: a dictionary of `{component reference name: (component, transform)}`
- connections: a list of `(component ref name 1, port name A, component ref name 2, port name B)`
- ports_map: a dictionary of which ports are being exposed together with their new name `{port_name: (component ref name, port name)}`

The code below illustrates how a simple MZI can be formed using this method.

```eval_rst

.. plot::
    :include-source:

    import gdsfactory as gf

    yaml = """
    instances:
        mmi_long:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 10
        mmi_short:
          component: mmi1x2
          settings:
            width_mmi: 4.5
            length_mmi: 5

    placements:
        mmi_long:
            rotation: 180
            x: 100
            y: 100

    routes:
        route_name1:
            links:
                mmi_short,E1: mmi_long,E0

    ports:
        E0: mmi_short,W0
        W0: mmi_long,W0
    """

    c = gf.component_from_yaml(yaml)
    c.show()
    c.plot()
```

Exporting connectivity map from a GDS is the first step towards verification.

- Adding ports to *every* cells in the GDS
- Generating the netlist


```eval_rst
.. plot::
    :include-source:

    import gdsfactory as gf
    c = gf.components.mzi()
    c.plot()
```

```eval_rst
.. plot::
    :include-source:

    import gdsfactory as gf
    c = gf.components.mzi()
    c.plot_netlist()
```



## component_sequence

This is a convenience function for cascading components such as cutbacks

The idea is to associate one symbol per type of section.
A section is uniquely defined by the component, its selected input and its selected output.

The mapping between symbols and components is supplied by a dictionnary.
The actual chain of components is supplied by a string or a list



- **Cutback heater**

```eval_rst

.. plot::
    :include-source:

    import gdsfactory as gf
    from gdsfactory.components import bend_circular
    from gdsfactory.components.straight import straight
    from gdsfactory.components.straight_heater import straight_heater
    from gdsfactory.components.component_sequence import component_sequence

    @gf.cell
    def test():
        # Define sub components
        radius=10.0
        bend180 = bend_circular(radius=radius, angle=180)
        wg = straight(length=5.0)
        wg_heater = straight_heater(length=20.0)

        # Define a map between symbols and (component, input port, output port)
        symbol_to_component = {
            "A": (bend180, 1, 2),
            "B": (bend180, 2, 1),
            "H": (wg_heater, 1, 2),
            "-": (wg, 1, 2),
        }

        # Generate a sequence
        # This is simply a chain of characters. Each of them represents a component
        # with a given input and and a given output

        sequence = "AB-H-H-H-H-BA"
        component = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)

        return component

    c = test()
    c.plot()

```

- **Cutback phase**

```eval_rst

.. plot::
    :include-source:

    import gdsfactory as gf
    from gdsfactory.components import bend_circular
    from gdsfactory.components.straight import straight
    from gdsfactory.components.straight_heater import straight_heater
    from gdsfactory.components.taper import taper_strip_to_ridge as _taper
    from gdsfactory.components.straight_pin import straight_pin
    from gdsfactory.layers import LAYER
    from gdsfactory.components.component_sequence import component_sequence



    def phase_mod_arm(straight_length=100.0, radius=10.0, n=2):

        # Define sub components
        bend180 = bend_circular(radius=radius, angle=180)
        pm_wg = straight_pin(length=straight_length)
        wg_short = straight(length=1.0)
        wg_short2 = straight(length=2.0)
        wg_heater = straight_heater(length=10.0)
        taper=_taper()

        # Define a map between symbols and (component, input port, output port)
        symbol_to_component = {
            "I": (taper, "1", "wg_2"),
            "O": (taper, "wg_2", "1"),
            "S": (wg_short, 1, 2),
            "P": (pm_wg, 1, 2),
            "A": (bend180, 1, 2),
            "B": (bend180, 2, 1),
            "H": (wg_heater, 1, 2),
            "-": (wg_short2, 1, 2),
        }

        # Generate a sequence
        # This is simply a chain of characters. Each of them represents a component
        # with a given input and and a given output

        repeated_sequence="SIPOSASIPOSB"
        heater_seq = "-H-H-H-H-"
        sequence = repeated_sequence * n + "SIPO" + heater_seq
        component = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)

        return component

    c = phase_mod_arm()
    c.plot()


```
