# Plugins

## Lumerical FDTD Sparameters

gdsfactory provides you with a Lumerical FDTD interface to calculate Sparameters automatically (without you having to click around the Lumerical GUI)

The function `gdsfactory.simulation.write_sparameters_lumerical` brings up a GUI, runs simulation and then writes the Sparameters both in .CSV and .DAT (Lumerical interconnect / simphony) file formats, as well as the simulation settings in YAML format.

In the CSV format each Sparameter will have 2 columns, `S12m` where `m` stands for magnitude and `S12a` where `a` stands for angle in radians. (S11a, S11m, ...)

For the simulation to work well, your components need to have ports, that will be extended automatically to go over the PML.

![FDTD](https://i.imgur.com/dHAzZRw.png)

The script calls internally the lumerical python API `lumapi` so you will need to make sure that you can run this from python.

```python
import lumapi

session = luampi.FDTD()
```

In linux that may require you to export the PYTHONPATH variable in your shell environment.

You can add one line into your `.bashrc` in your Linux machine. This line will depend also on your Lumerical version. For example for Lumerical 2019b

```bash
[ -d "/opt/lumerical/2019b" ] && export PATH=$PATH:/opt/lumerical/2019b/bin && export PYTHONPATH=/opt/lumerical/2019b/api/python
```

And for 2021v212

```bash
[ -d "/opt/lumerical/v212" ] && export PATH=$PATH:/opt/lumerical/v212/api/python/bin && export PYTHONPATH=/opt/lumerical/v212/api/python
```

```{eval-rst}
.. autofunction:: gdsfactory.simulation.write_sparameters_lumerical.write_sparameters_lumerical
```

```{eval-rst}

.. autofunction:: gdsfactory.simulation.read.read_sparameters_lumerical
```

```{eval-rst}
.. autofunction:: gdsfactory.simulation.read.read_sparameters_pandas
```

I reccommend that you adapt this functions with your :

- simulation settings (wavelength range, mesh)
- LayerStack: GDS layer and thickness/material/zmin of each layer
- dirpath: directory path to write and read the Sparameters

```function
import gdsfactory as gf
import gdsfactory.simulation as sim
import gdsfactory.samples.pdk.fab_c as pdk

write_sparameters_lumerical = gf.partial(
    sim.write_sparameters_lumerical,
    layer_stack=pdk.LAYER_STACK,
    dirpath=pdk.SPARAMETERS_PATH,
)

plot_sparameters = gf.partial(
    sim.plot.plot_sparameters,
    dirpath=pdk.SPARAMETERS_PATH,
    write_sparameters_function=write_sparameters_lumerical,
)

read_sparameters_pandas = gf.partial(
    sim.read_sparameters_pandas,
    layer_stack=pdk.LAYER_STACK,
    dirpath=pdk.SPARAMETERS_PATH,
)

```

## Meep Transmission Sparameters or waveguide modes (MPB)

See work in progess at `plugins/gmeep`. You can also take a look at

- [picwriter](https://github.com/DerekK88/PICwriter)
- [gdshelpers](https://github.com/HelgeGehring/gdshelpers)

## Tidy3d

See work in progress at `plugins/tidy3d`

## Sparameters Circuit solvers

You can chain the Sparameters to solve the Sparameters response of larger circuits using a circuit solver such as:

- Lumerical interconnect
- [simphony (open source)](https://simphonyphotonics.readthedocs.io/en/latest/)
- [sax (open source)](https://sax.readthedocs.io/en/latest/index.html)
