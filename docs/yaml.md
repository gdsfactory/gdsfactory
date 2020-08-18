# YAML

YAML is an easy to use human-friendly format to define components, circuits and masks


```eval_rst
.. automodule:: pp.component_from_yaml
   :members:
```


```eval_rst
.. plot::
    :include-source:

    import pp

    netlist = """
    instances:
        CP1:
          component: mmi1x2
          settings:
              width_mmi: 4.5
              length_mmi: 10
        CP2:
            component: mmi1x2
            settings:
                width_mmi: 4.5
                length_mmi: 5
        arm_top:
            component: mzi_arm
        arm_bot:
            component: mzi_arm

    placements:
        arm_bot:
            mirror: [0,0,0,10]
    ports:
        W0: CP1,W0
        E0: CP2,W0

    connections:
        arm_bot,W0: CP1,E0
        arm_top,W0: CP1,E1
        CP2,E0: arm_bot,E0
        CP2,E0: arm_top,E0
    """

    c = pp.component_from_yaml(netlist)
    pp.show(c)
    pp.plotgds(c)
```
