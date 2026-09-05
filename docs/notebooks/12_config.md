<!-- #region -->
# Configuration

You have 2 ways of configuring gdsfactory:

1. Use `.env` files with CONFIG variables

gdsfactory can store settings in a `.env` file in the current directory or any parent directories. Environment variables do not require any prefix.

You can create a `.env` file in this directory, restart the notebook and see what CONFIG values you are getting.

```python
difftest_ignore_cell_name_differences=True
difftest_ignore_sliver_differences=False
difftest_ignore_label_differences=False
layer_error_path=[1000, 0]
max_cellname_length=64
connect_use_mirror=False
cell_layout_cache=True
```


2. You can import CONF at the beginning of your script and then overwrite any settings.

Here is a list of all the available config fields with their defaults:

| Field | Default |
|---|---|
| `pdk` | `None` |
| `layer_label` | `(100, 0)` |
| `difftest_ignore_label_differences` | `False` |
| `difftest_ignore_sliver_differences` | `False` |
| `difftest_ignore_cell_name_differences` | `True` |
| `bend_radius_error_type` | `ErrorType.ERROR` |
| `layer_error_path` | `(1000, 0)` |
| `layer_marker` | `(205, 0)` |
| `connect_use_mirror` | `False` |
| `max_cellname_length` | `64` |
| `cell_layout_cache` | `True` |
| `port_types` | `["optical", "electrical", "placement", ...]` |
| `port_types_grating_couplers` | `["vertical_te", "vertical_tm", "vertical_dual"]` |
| `exclude_layers` | `None` |
<!-- #endregion -->

```python
import gdsfactory as gf



gf.gpdk.PDK.activate()
gf.CONF

```

```python
gf.CONF.max_cellname_length = 9  # Making the cell name ridiculously short.
c1 = gf.Component("c123456789")
print(c1.name) # The name will only be truncated when saving to GDS
```

As you can see the cell names are truncated when writing them to GDS.

```python
gdspath = c1.write_gds()
c2 = gf.import_gds(gdspath)
print(c2.name)
```
