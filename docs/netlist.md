# 2. Connect Netlist

- Level 0 components have geomtry and port information
- Level 1 components are defined by netlist connectivity of level 0 components

![component levels](images/lib_example.png)

Sometimes, when a component is mostly composed of sub-components adjacent to each
other, it can be easier to define the component by only saying what the
sub-component are, how they are connected, and which ports are part of the
new components.

This can be done using a netlist based approach where these 3 parts are defined:

- components: a dictionnary of `{component reference name: (component, transform)}`
- connections: a list of `(component ref name 1, port name A, component ref name 2, port name B)`
- ports_map: a dictionnary of which ports are being exposed together with their new name `{port_name: (component ref name, port name)}`

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
        mmi_short,E1: mmi_long,E0

    ports:
        E0: mmi_short,W0
        W0: mmi_long,W0
    """

    c = pp.component_from_yaml(yaml)
    pp.show(c)
    pp.plotgds(c)
```

Exporting connectivity map from a GDS is the first step towards verification.

- Adding ports to *every* cells in the GDS
- Generating the netlist


```eval_rst
.. plot::
    :include-source:

    import pp
    c = pp.c.mzi()
    pp.qp(c)
```

```eval_rst
.. plot::
    :include-source:

    import pp
    c = pp.c.mzi()
    c.plot_netlist()
```
