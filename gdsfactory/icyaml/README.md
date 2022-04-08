# ICYAML

Webapp to visualize in real time gdsfactory netlists.

To read more about gdsfactory netlist format you can take a look a the netlist driven flows [docs](https://gdsfactory.github.io/gdsfactory/notebooks/07_yaml_component.html)

The idea is that you have klayout open in one panel and the webapp in the other window, and you can see the circuit live changes.

![windows](https://i.imgur.com/xKCxSpp.png)


Then save the circuit with `.ic.yaml` extension.


You can launch it from the terminal


```

gf yaml webapp

```


For the klayout live update make sure you install the klayout gdsfactory SALT package. See instructions [here](https://gdsfactory.github.io/gdsfactory/notebooks/00_klayout.html)
