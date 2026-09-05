# Cell decorator

The `@cell` decorator is for functions that return a Component. Always add it to avoid duplicate component names.

Why?

- In GDS each component must have a unique name, consistent across runs so you can merge GDS files created at different times.
- Two components in a GDS file cannot have the same name — they must be references (instances) of the same component.

What does `@cell` do?

1. Gives the component a unique name based on the function name and its parameters.
2. Caches components by name, so calling the same function with the same parameters returns the cached component instead of rebuilding it.
3. Optionally tags the cell with categories for organization (e.g. `@gf.cell(tags=["mmi", "coupler"])`).

Let's see how it works.

```python
import gdsfactory as gf



gf.gpdk.PDK.activate()
def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component()
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend()
print(f"this cell {c.name!r} does NOT get automatic name")
c.plot()

```

```python
mzi_with_bend_decorated = gf.cell(mzi_with_bend)
c = mzi_with_bend_decorated(radius=10)
print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()
```

```python
@gf.cell
def mzi_with_bend(radius: float = 10.0) -> gf.Component:
    c = gf.Component()
    mzi = c << gf.components.mzi()
    bend = c << gf.components.bend_euler(radius=radius)
    bend.connect("o1", mzi.ports["o2"])
    return c


c = mzi_with_bend()
print(f"this cell {c.name!r} gets automatic name thanks to the `cell` decorator")
c.plot()
```

```python
# See how the cells get the name from the parameters that you pass them.

c = mzi_with_bend()
print(c)

print("second time you get this cell from the cache")
c = mzi_with_bend()
print(c)

print("If you call the cell with different parameters, the cell gets a different name")
c = mzi_with_bend(radius=20)
print(c)
```


## Tags

You can tag cells with categories using the `tags` parameter. This helps organize your component library and enables programmatic filtering by category.

```python
import kfactory as kf


@gf.cell(tags=["mmi", "coupler"])
def mmi1x2(width: float = 0.5, length_mmi: float = 5.0) -> gf.Component:
    return gf.components.mmi1x2(width=width, length_mmi=length_mmi)


@gf.cell(tags=["coupler"])
def coupler_ring(gap: float = 0.2, radius: float = 5.0) -> gf.Component:
    return gf.components.coupler_ring(gap=gap, radius=radius)


# You can retrieve all registered cell factories that share a tag
coupler_factories = kf.kcl.factories.get_by_tag("coupler")
print("Cells tagged 'coupler':", [f.name for f in coupler_factories])

mmi_factories = kf.kcl.factories.get_by_tag("mmi")
print("Cells tagged 'mmi':", [f.name for f in mmi_factories])
```

Sometimes when you are changing the inside code of the function, you need to remove the component from the cache to make sure the code runs again.

This is useful when using jupyter notebooks or the file watcher.


```python
@gf.cell
def wg(length=10, width=1, layer=(1, 0)):
    print("BUILDING waveguide")
    c = gf.Component()
    c.info["area"] = width * length
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(
        name="o1", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    c.add_port(
        name="o2", center=(length, width / 2), width=width, orientation=0, layer=layer
    )
    return c
```

```python
c = wg()
gf.clear_cache()
c = wg()
gf.clear_cache()
c = wg()
gf.clear_cache()
```


Another option is to just delete the cell after creating it.

```python
c = wg(length=11)
c.delete()
c = wg(length=11)
```

## Settings and Info

Together with the GDS file that you send to the foundry you can also store the settings for each component.

- `Component.settings` are the input settings for each cell to generate a component. For example, `wg.settings.length` will return the input `length` setting to you, which you used to create that waveguide.
- `Component.info` are the derived properties that will be computed inside the cell function. For example `wg.info.area` will return the computed area of that waveguide.

```python
c.info
```

```python
c.info['area']
```

```python
c.settings
```

```python
c.settings['length']
```

Components also have pretty print for ports `c.pprint_ports()`

```python
c.pprint_ports()
```


```python
# You always add any relevant information `info` to the cell.
c.info["polarization"] = "te"
c.info["wavelength"] = 1.55
c.info
```


## Cache

To avoid that 2 exact cells are not references of the same cell the `cell` decorator has a cache where if a component has already been built it will return the component from the cache


```python
@gf.cell
def straight(length=10, width=1):
    c = gf.Component()
    c.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=(1, 0))
    print(f"BUILDING {length}um long straight waveguide")
    return c
```

If you run the cell below multiple times it will print a message because we are deleting the CACHE every single time and every time the cell will have a different Unique Identifier (UID).

```python
wg1 = straight()
id(wg1)
```

If you run the cell below multiple times it will NOT print a message because we are hitting CACHE every single time and every time the cell will have the SAME Unique Identifier (UID) because it's the same cell.

```python
wg2 = straight(length=10)
print(id(wg2))  # Same id as wg1 — returned from cache
```


```python
wg3 = straight(length=12)
wg4 = straight(length=13)
```

<!-- #region -->
## Create cells without `cell` decorator

The cell decorator names cells deterministically and uniquely based on the name of the functions and its parameters.

It also uses a caching mechanism that improves performance and guards against duplicated names.

The most common mistake new gdsfactory users make is to create cells without the `cell` decorator.

### Avoid naming cells manually: Use cell decorator

Naming cells manually is susceptible to name collisions

in GDS you can't have two cells with the same name.

For example: this code will raise a `duplicated cell name ValueError`

```python
import gdsfactory as gf

c1 = gf.Component("wg")
c1 << gf.components.straight(length=5)


c2 = gf.Component("wg")
c2 << gf.components.straight(length=50)


c3 = gf.Component("waveguides")
wg1 = c3 << c1
wg2 = c3 << c2
wg2.movey(10)
c3
```

**Solution**: Use the `gf.cell` decorator for automatic naming your components.
<!-- #endregion -->

```python
import gdsfactory as gf


@gf.cell    
def wg(length: float = 3):
    return gf.components.straight(length=length)

gf.clear_cache()

print(wg(length=5).name)
print(wg(length=10).name)
```

### Avoid Unnamed cells. Use `cell` decorator

In the case of not wrapping the function with `cell` you will get unique names thanks to the unique identifier `uuid`.

This name will be different and non-deterministic for different invocations of the script.

However it will be hard for you to know where that cell came from.

```python
c1 = gf.Component()
c2 = gf.Component()

print(c1.name)
print(c2.name)
```

Notice how gdsfactory raises a Warning when you save this `Unnamed` Component.

```python
c1.write_gds()
```


<!-- #region -->
### Avoid Intermediate Unnamed cells. Use `cell` decorator

While creating a cell, you should not create intermediate cells, because they will not be Cached and you can end up with duplicated cell names or name conflicts, where one of the cells that has the same name as the other will be replaced.


```python

@gf.cell
def die_bad():
    """c1 is an intermediate Unnamed cell"""
    c1 = gf.Component() # This is an intermediate cell, it is not named as it is not returned by the function.
    _ = c1 << gf.components.straight(length=10)
    return gf.components.die(size=(2e3, 2e3), street_width=10)


c = die_bad()
c.show()
c.plot()

```
<!-- #endregion -->


**Solution** Do not use intermediate cells


```python
import gdsfactory as gf


@gf.cell
def die_good():
    c = gf.Component()
    _ = c << gf.components.straight(length=10)
    return gf.components.die(size=(2000,2000), street_width=10)


c = die_good()
c.plot()
```
