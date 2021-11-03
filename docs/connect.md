# Connect

## References

You can connect:

- As a component sequence
- Defining a netlist
- Connecting the references butt to butt

![component levels](images/lib_example.png)


```{eval-rst}

.. plot::
    :include-source:

    import gdsfactory as gf


    @gf.cell
    def ring_single(
        gap: float = 0.2,
        radius: float = 10.0,
        length_x: float = 4.0,
        length_y: float = 0.6,
        coupler_ring: gf.types.ComponentFactory = gf.c.coupler_ring,
        straight: gf.types.ComponentFactory = gf.c.straight,
        bend: gf.types.ComponentFactory = gf.c.bend_euler,
        cross_section: gf.types.CrossSectionFactory = gf.cross_section.strip,
        **kwargs
    ) -> gf.Component:
        """Single bus ring made of a ring coupler (cb: bottom)
        connected with two vertical straights (sl: left, sr: right)
        two bends (bl, br) and horizontal straight (wg: top)

        Args:
            gap: gap between for coupler
            radius: for the bend and coupler
            length_x: ring coupler length
            length_y: vertical straight length
            coupler_ring: ring coupler function
            straight: straight function
            bend: 90 degrees bend function
            cross_section:
            **kwargs: cross_section settings


        .. code::

              bl-st-br
              |      |
              sl     sr length_y
              |      |
             --==cb==-- gap

              length_x

        """
        gf.snap.assert_on_2nm_grid(gap)

        coupler_ring = gf.partial(
            coupler_ring,
                bend=bend,
                gap=gap,
                radius=radius,
                length_x=length_x,
                cross_section=cross_section,
                **kwargs
        )

        straight_side = gf.partial(
            straight, length=length_y, cross_section=cross_section, **kwargs
        )
        straight_top = gf.partial(
            straight, length=length_x, cross_section=cross_section, **kwargs
        )

        bend = gf.partial(bend, radius=radius, cross_section=cross_section, **kwargs)

        c = gf.Component()
        cb = c << coupler_ring()
        sl = c << straight_side()
        sr = c << straight_side()
        bl = c << bend()
        br = c << bend()
        st = c << straight_top()
        # st.mirror(p1=(0, 0), p2=(1, 0))

        sl.connect(port="o1", destination=cb.ports["o2"])
        bl.connect(port="o2", destination=sl.ports["o2"])

        st.connect(port="o2", destination=bl.ports["o1"])
        br.connect(port="o2", destination=st.ports["o1"])
        sr.connect(port="o1", destination=br.ports["o1"])
        sr.connect(port="o2", destination=cb.ports["o3"])

        c.add_port("o2", port=cb.ports["o4"])
        c.add_port("o1", port=cb.ports["o1"])
        return c


    ring = ring_single()
    ring.plot()
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

```{eval-rst}

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
                mmi_short,o3: mmi_long,o4

    ports:
        o4: mmi_short,o1
        o1: mmi_long,o1
    """

    c = gf.read.from_yaml(yaml)
    c.show()
    c.plot()
```

Exporting connectivity map from a GDS is the first step towards verification.

- Adding ports to *every* cells in the GDS
- Generating the netlist


```{eval-rst}
.. plot::
    :include-source:

    import gdsfactory as gf
    mzi = gf.components.mzi()
    mzi.plot()
```

```{eval-rst}
.. plot::
    :include-source:

    import gdsfactory as gf
    mzi_netlist = gf.components.mzi()
    mzi_netlist.plot_netlist()
```



## component_sequence

This is a convenience function for cascading components such as cutbacks

The idea is to associate one symbol per type of section.
A section is uniquely defined by the component, its selected input and its selected output.

The mapping between symbols and components is supplied by a dictionary.
The actual chain of components is supplied by a ASCII string where each character represents one component.




- **Cutback phase**

```{eval-rst}

.. plot::
    :include-source:

    import gdsfactory as gf


    def cutback_phase(straight_length=100.0, radius=10.0, n=2):

        # Define sub components
        bend180 = gf.components.bend_circular(radius=radius, angle=180)
        pm_wg = gf.components.straight_pin(length=straight_length)
        wg_short = gf.components.straight(length=1.0)
        wg_short2 = gf.components.straight(length=2.0)
        wg_heater = gf.c.straight_heater_metal(length=10.0)

        # Define a map between symbols and (component, input port, output port)
        symbol_to_component = {
            "S": (wg_short, "o1", "o2"),
            "P": (pm_wg, "o1", "o2"),
            "A": (bend180, "o1", "o2"),
            "B": (bend180, "o2", "o1"),
            "H": (wg_heater, "o1", "o2"),
            "-": (wg_short2, "o1", "o2"),
        }

        # Generate a sequence
        # This is simply a chain of characters. Each of them represents a component
        # with a given input and and a given output

        repeated_sequence = "SPSASPSB"
        heater_seq = "-H-H-H-H-"
        sequence = repeated_sequence * n + "SP" + heater_seq
        component = gf.components.component_sequence(
            sequence=sequence, symbol_to_component=symbol_to_component
        )

        return component


    cutback = cutback_phase()
    cutback.plot()

```
