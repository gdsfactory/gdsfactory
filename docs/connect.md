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

    import pp

    @pp.cell
    def ring(
        coupler90=pp.components.coupler90,
        coupler_straight=pp.components.coupler_straight,
        waveguide=pp.components.waveguide,
        bend=pp.components.bend_euler,
        length_y=2.0,
        length_x=4.0,
        gap=0.2,
    ):
        """ single bus ring
        """
        c = pp.Component()

        # define subcells
        coupler90 = pp.call_if_func(coupler90, gap=gap)
        waveguide_x = pp.call_if_func(waveguide, length=length_x)
        waveguide_y = pp.call_if_func(waveguide, length=length_y)
        bend = pp.call_if_func(bend)
        coupler_straight = pp.call_if_func(coupler_straight, gap=gap, length=length_x)

        # add references to subcells
        cbl = c << coupler90
        cbr = c << coupler90
        cs = c << coupler_straight
        wyl = c << waveguide_y
        wyr = c << waveguide_y
        wx = c << waveguide_x
        btl = c << bend
        btr = c << bend


        # connect references
        wyr.connect(port="E0", destination=cbr.ports["N0"])
        cs.connect(port="E0", destination=cbr.ports["W0"])

        cbl.reflect(p1=(0, coupler90.y), p2=(1, coupler90.y))
        cbl.connect(port="W0", destination=cs.ports["W0"])
        wyl.connect(port="E0", destination=cbl.ports["N0"])

        btl.connect(port="N0", destination=wyl.ports["W0"])
        btr.connect(port="W0", destination=wyr.ports["W0"])
        wx.connect(port="W0", destination=btl.ports["W0"])
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

    import pp

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
            factory: optical
            links:
                mmi_short,E1: mmi_long,E0

    ports:
        E0: mmi_short,W0
        W0: mmi_long,W0
    """

    c = pp.component_from_yaml(yaml)
    c.show()
    c.plot()
```

Exporting connectivity map from a GDS is the first step towards verification.

- Adding ports to *every* cells in the GDS
- Generating the netlist


```eval_rst
.. plot::
    :include-source:

    import pp
    c = pp.components.mzi()
    c.plot()
```

```eval_rst
.. plot::
    :include-source:

    import pp
    c = pp.components.mzi()
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

    import pp
    from pp.components import bend_circular
    from pp.components.waveguide import waveguide
    from pp.components.waveguide_heater import waveguide_heater
    from pp.components.component_sequence import component_sequence

    @pp.cell
    def test():
        # Define sub components
        radius=10.0
        bend180 = bend_circular(radius=radius, angle=180)
        wg = waveguide(length=5.0)
        wg_heater = waveguide_heater(length=20.0)

        # Define a map between symbols and (component, input port, output port)
        string_to_device_in_out_ports = {
            "A": (bend180, "W0", "W1"),
            "B": (bend180, "W1", "W0"),
            "H": (wg_heater, "W0", "E0"),
            "-": (wg, "W0", "E0"),
        }

        # Generate a sequence
        # This is simply a chain of characters. Each of them represents a component
        # with a given input and and a given output

        sequence = "AB-H-H-H-H-BA"
        component = component_sequence(sequence, string_to_device_in_out_ports)

        return component

    c = test()
    c.plot()

```

- **Cutback phase**

```eval_rst

.. plot::
    :include-source:

    import pp
    from pp.components import bend_circular
    from pp.components.waveguide import waveguide
    from pp.components.waveguide_heater import waveguide_heater
    from pp.components.taper import taper_strip_to_ridge as _taper
    from pp.components.waveguide_pin import waveguide_pin
    from pp.layers import LAYER
    from pp.components.component_sequence import component_sequence



    def phase_mod_arm(straight_length=100.0, radius=10.0, n=2):

        # Define sub components
        bend180 = bend_circular(radius=radius, angle=180)
        pm_wg = waveguide_pin(length=straight_length)
        wg_short = waveguide(length=1.0)
        wg_short2 = waveguide(length=2.0)
        wg_heater = waveguide_heater(length=10.0)
        taper=_taper()

        # Define a map between symbols and (component, input port, output port)
        string_to_device_in_out_ports = {
            "I": (taper, "1", "wg_2"),
            "O": (taper, "wg_2", "1"),
            "S": (wg_short, "W0", "E0"),
            "P": (pm_wg, "W0", "E0"),
            "A": (bend180, "W0", "W1"),
            "B": (bend180, "W1", "W0"),
            "H": (wg_heater, "W0", "E0"),
            "-": (wg_short2, "W0", "E0"),
        }

        # Generate a sequence
        # This is simply a chain of characters. Each of them represents a component
        # with a given input and and a given output

        repeated_sequence="SIPOSASIPOSB"
        heater_seq = "-H-H-H-H-"
        sequence = repeated_sequence * n + "SIPO" + heater_seq
        component = component_sequence(sequence, string_to_device_in_out_ports)

        return component

    c = phase_mod_arm()
    c.plot()


```
