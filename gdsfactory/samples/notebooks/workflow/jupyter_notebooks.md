# Jupyter notebooks

Working with Jupyter notebooks is great for learning gdsfactory as well as running heavy simulations on Cloud servers.

Thanks to [kweb](https://github.com/gdsfactory/kweb) you can use the webapp version on klayout in your browser or inside jupyter notebooks.

```python
import kweb.server_jupyter as kj  # requires `pip install gdsfactory[full]` or `pip install kweb`
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.get_generic_pdk()
PDK.activate()

gf.config.set_log_level("DEBUG")
kj.start()
```

```python
c = gf.components.mzi()
```

```python
c.plot_jupyter()
```

```python
c = gf.components.bend_circular()
c.plot_jupyter()
```

```python
c = gf.components.straight_heater_meander()
c.plot_jupyter()
```

```python
c
```

```python
s = c.to_3d()
s.show()
```
