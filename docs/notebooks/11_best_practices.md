# Best practices


## Use cell decorator

- Do not leave cells unnamed.

Unnamed cells are going to get different names every time you run the code.
It is going to be hard to know where they come from.

```python
import gdsfactory as gf


gf.gpdk.PDK.activate()
c = gf.Component()
print(c.name)

```

- Do not name cells manually. Manually defined names can create duplicated cells.

```python
c1 = gf.Component("my_cell1")
c2 = gf.Component("my_cell2")
c1.add_ref(gf.components.straight(length=10))
c2.add_ref(gf.components.straight(length=11))

c3 = gf.Component("im_going_to_have_duplicated_cell_names")
_ = c3.add_ref(c1)
_ = c3.add_ref(c2)
c3.plot()
```


Solution: Use the cell decorator to name cells.


```python
@gf.cell
def my_pcell(length=10):
    c = gf.Component()
    ref = c.add_ref(gf.components.straight(length=length))
    c.add_ports(ref.ports)
    return c


print(my_pcell(length=11).name)
print(my_pcell(length=12).name)
```


## Keep cell functions simple

As you make functions made of other functions one can start passing a lot of arguments to the function. This makes the code hard to write, read and maintain.

- Avoid complicated functions with many parameters.


```python
@gf.cell
def mzi_with_bend_overly_complicated(
    mzi_delta_length: float = 10.0,
    mzi_length_y: float = 2.0,
    mzi_length_x: float | None = 0.1,
    bend_radius: float = 10,
    bend_cross_section="strip",
):
    """Returns MZI interferometer with bend."""
    c = gf.Component()
    mzi1 = c.add_ref(
        gf.components.mzi(
            delta_length=mzi_delta_length,
            length_y=mzi_length_y,
            length_x=mzi_length_x,
        )
    )
    bend1 = c.add_ref(
        gf.components.bend_euler(radius=bend_radius, cross_section=bend_cross_section)
    )
    bend1.connect("o1", mzi1.ports["o2"])
    c.add_port("o1", port=mzi1.ports["o1"])
    c.add_port("o2", port=bend1.ports["o2"])
    return c


c = mzi_with_bend_overly_complicated(bend_radius=100)
c.plot()
```

Solution:

- leverage `functools.partial` to customize the default parameters of a function.

```python
from functools import partial


@gf.cell
def mzi_with_bend(mzi=gf.components.mzi, bend=gf.components.bend_euler):
    """Returns MZI interferometer with bend."""
    c = gf.Component()
    mzi1 = c.add_ref(mzi())
    bend1 = c.add_ref(bend())
    bend1.connect("o1", mzi1.ports["o2"])
    c.add_port("o1", port=mzi1.ports["o1"])
    c.add_port("o2", port=bend1.ports["o2"])
    return c

# The partial function from Python's functools library is used to create a new function called bend_big.
# This new function is a version of the standard bend_euler component, but with the radius parameter permanently set to 100.
bend_big = partial(gf.components.bend_euler, radius=100)

# This argument overrides the default bend component used in the MZI and replaces it with the custom, larger-radius bend factory defined in the previous step.
c = mzi_with_bend(bend=bend_big)
c.plot()
```


## Use array of references

- Array of references are more memory efficient and faster to create.


```python
@gf.cell
def pad_array_slow(
    cols: int = 30,
    rows: int = 30,
    spacing: tuple[float, float] = (200, 200),
    pad=gf.components.pad,
):
    """Returns a grid of pads. BAD SLOW CODE. Please don't use this. See pad_array_fast instead."""
    xspacing, yspacing = spacing
    c = gf.Component()
    for col in range(cols):
        for row in range(rows):
            c.add_ref(pad()).movex(col * xspacing).movey(row * yspacing)
    return c


c = pad_array_slow()
c.plot()
```


```python
@gf.cell
def pad_array_fast(
    cols: int = 30,
    rows: int = 30,
    column_pitch: float = 200,
    row_pitch: float = 200,
    pad=gf.components.pad,
) -> gf.Component:
    """Returns a grid of pads. GOOD CODE. USE THIS.

    Args:
        cols: number of columns.
        rows: number of rows.
        column_pitch: distance between columns.
        row_pitch: distance between rows.
        pad: pad cell.
    """
    c = gf.Component()
    c.add_ref(
        pad(), columns=cols, rows=rows, column_pitch=column_pitch, row_pitch=row_pitch
    )
    return c


c = pad_array_fast()
c.plot()
```
