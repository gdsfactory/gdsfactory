# Python intro

gdsfactory is written in python and requires some basic knowledge of python.

If you are new to python you can find many resources online:

- [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
- [youTube videos](https://www.youtube.com/c/anthonywritescode)
- [courses](https://github.com/joamatab/practical-python)

This notebook is for you to experiment with some common python patterns in `gdsfactory`.

## Classes

Gdsfactory already has some pre-defined classes for you.

All the other classes (Component, ComponentReference, Port ...) are already available in `gf.typings`

Classes are good for keeping state(s), which means that they store some information inside them (polygons, ports, references ...).

In gdsfactory you will write functions instead of classes. Functions are easier to write and combine, and have clearly defined inputs and outputs.

```python
import gdsfactory as gf

gf.gpdk.PDK.activate()

```

```python
c = gf.Component()
c.add_polygon([(-8, -6), (6, 8), (7, 17), (9, 5)], layer=(1, 0))
c.plot()
```


## Functions

Functions have clear inputs and outputs, they usually accept some parameters (strings, floats, ints ...) and return other parameters.


```python
def double(x):
    return 2 * x


x = 1.5
y = double(x)
print(y)
```

It is also nice to add `type annotations` to your functions to clearly define what are the input/output types (string, int, float ...).


```python
def double(x: float) -> float:
    return 2 * x
```


## Factories

A factory is a function that returns an object. In gdsfactory many functions return a `Component` object.


```python
def bend(radius: float = 5) -> gf.Component:
    return gf.components.bend_euler(radius=radius)


component = bend(radius=10)

print(component)
component.plot()
```


## Decorators

A decorator is a special function in Python that lets you add extra functionality to other functions, without modifying their code.

In gdsfactory, decorators are used for managing functions that return a `Component`. For example, the `@cell` decorator:

- Gives each `Component` a unique name based on its input parameters.
- Caches the returned `Component` for speed and to avoid duplicating cells.

For that you will see a `@cell` decorator on many component functions.

An example of a decorator is `validate_call` in the pydantic library.


```python
from pydantic import validate_call

@validate_call
def double(x: float) -> float:
    return 2 * x


x = 1.5
y = double(x)
print(y)
```

The validator decorator is equivalent to running.


```python
def double(x: float) -> float:
    return 2 * x


double_with_validator = validate_call(double)
x = 1.5
y = double_with_validator(x)
print(y)
```

Let us try to create an error `x` and you will get a clear message the the function `double` does not work with strings.

<!-- #region -->
```python
y = double("not_valid_number")
```

will raise a `ValidationError`

```
ValidationError: 1 validation error for double
0
  Input should be a valid number, unable to parse string as a number ...

```

`validate_call` will also `cast` the input type based on the type annotation. So if you pass another type it will convert it to `float`.
<!-- #endregion -->

```python
x = "12"
print(f"{double(x)=}")
print(f"{double_with_validator(x)=}")
```

## List comprehensions

You will also see some list comprehensions, which are common in python.

For example, you can write many loops in one line:

```python
y = []
for x in range(3):
    y.append(double(x))

print(y)
```

```python
y = [double(x) for x in range(3)]  # Much shorter and easier to read.
print(y)
```


## Functional programming

Functional programming follows the linux philosophy:

- Write functions that do one thing and do it well.
- Write functions to work together.
- Write functions with clear **inputs** and **outputs**.

### partial

Partial is an easy way to modify the default arguments of a function. This is useful in gdsfactory because we define Parametric Cells (PCells) using functions. `from functools import partial`

You can use a partial to create a new function with some default arguments.

The following two functions are equivalent in functionality. Notice how the second one is shorter, more readable and easier to maintain thanks to `partial`.



#### Example without partial

```python
def ring_sc1(gap=0.3, **kwargs) -> gf.Component:
    return gf.components.ring_single(gap=gap, **kwargs)

# Usage
c1 = ring_sc1(radius=10)
```

#### Example with partial

```python
from functools import partial

ring_sc2 = partial(gf.components.ring_single, gap=0.3)

# Usage
c2 = ring_sc2(radius=10)
```

`partial` becomes more useful as you customize more parameters and more widely reuse functions.



### compose

`gf.compose` combines two functions into one. This is useful in gdsfactory because we
define PCells using functions and functions are easier to combine than classes.

(This is the compose function in the toolz package `from toolz import compose`).

```python
ring_sc5 = partial(gf.components.ring_single, radius=10)
add_gratings = gf.routing.add_fiber_array

ring_sc_gc = gf.compose(add_gratings, ring_sc5)
ring_sc_gc5 = ring_sc_gc(radius=5)
ring_sc_gc5
```

This is equivalent to calling the functions one after the other.

```python
ring_sc_gc5 = add_gratings(ring_sc1(radius=5))
ring_sc_gc5
```

## Ipython

This Jupyter Notebook uses an Interactive Python Terminal (Ipython). So you can interact with the code.

For more details on Jupyter Notebooks, you can visit the [Jupyter website](https://jupyter.org/).

The most common trick that you will see is that we use `?` to see the documentation of a function or `help(function)`.

```python
gf.components.coupler?
```

```python
help(gf.components.coupler)
```

To see the source code of a function you can use `??`

```python
gf.components.coupler??
```

To time the execution time of a cell, you can add a `%time` on top of the cell.

```python
%time


def hi() -> None:
    print("hi")


hi()
```
