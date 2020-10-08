# Klayout autoplacer

You can load GDS files and place them using a mix and match of specific or automatic placing.

This placer is a more complicated alternative to the YAML placer


## Library
```eval_rst
.. autoclass:: pp.autoplacer.library.Library
```


## Autoplacer

```eval_rst
.. autoclass:: pp.autoplacer.AutoPlacer
```


## ChipArray

```eval_rst
.. autoclass:: pp.autoplacer.ChipArray
```


- Pack grid


```
    mask.chips[i].pack_grid(
        lib.pop_doe(doe),
        normalization_origin=ap.SOUTH_WEST,
        rows=1,
        name=doe,
        origin=ap.SOUTH_WEST,
        align=None,
    )

```

- Pack groups

```
    mask.chips[i].pack_groups(
        lib.pop_doe(doe),
        normalization_origin=ap.SOUTH_WEST,
        rows=1,
        name=doe,
        origin=ap.SOUTH_WEST,
    )
```
