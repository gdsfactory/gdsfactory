## 8 to 9


In GDSFactory8, ports and instance units were not consistent with each other. Port units were in DBU while instance units were in um.

GDSFactory9 makes ports and instance units consistent with each other. Port units are in um while instance units are in um.

We have had to make some small changes to the API of the following classes:

- Component
- Port
- ComponentReference
- ComponentReferences


#### Main changes

Note the migration changes:
- `port.dangle` has been renamed to `port.orientation`
    - `port.angle` represents the angle 90 degrees multiples.
    - `port.orientation` represents the angle in degrees.
- `CHECK_INSTANCES` has been renamed to `CheckInstances`
    - Most often used in the `@cell` decorator.
- `Port(..., layer: tuple[int, int])` has been replaced with `Port(..., layer: int)`
    - `layer` now represents the layer index in the KLayout layer stack.
    - To update your code you can do `Port(..., layer=gf.kcl.layout.layer(*layer))`
    - We also recommend just using Component.add_port, which lets you specify the layer as tuple.
- Most of the `d` prefix attributes and functions can now be used without the `d` prefix.
- `ComponentReference.parent` has been removed, you should use `ComponentReference.cell` instead.
- `ComponentReference.info` has been removed, you should use `ComponentReference.cell.info` instead.

```python
import gdsfactory as gf



gf.gpdk.PDK.activate()
c = gf.components.straight(length=10)
c.pprint_ports()

```

```python
print('port orientation in degrees', c.ports['o2'].orientation)
print('port x in um', c.ports['o2'].x)
print('port dx in um (decimal)', c.ports['o2'].dx)
print('port ix in integer (database units)', c.ports['o2'].ix)
```

```python
c = gf.Component()
ref = c << gf.c.straight()
```

```python
ref.xmin = 10
```

```python
print('reference x in um', ref.x)
print('reference dx in um (decimal)', ref.dx)
print('reference ix in integer (database units)', ref.ix)
```

```python
c
```
