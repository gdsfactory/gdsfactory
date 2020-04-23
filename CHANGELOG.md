# TODO

- mzi2x2 with heaters have some issue, where waveguide connects to the heater


# dev

- flake8 is passing now
- added flake8 to pre-commit hook
- simpler JSON file for mask metadata mask.tp.json
- added container decorator, can inherit ports, settings, test and data analysis protocols and still have a different name to avoid name collisions
- samples run as part of the test suite, moved samples into pp
- autoname sorts kwarg keys by alphabetical order and has tests
- autoname accepts max_name_length and ignore_from_name kwargs


# 1.1.6 2020-04-11

- mask JSON works with cached GDS files for the klayout placer
- added layers to CONFIG['layers']
- write_labels gets layer from `CONFIG['layers']['LABEL']`
- add_padding works over the same component --> this was not a good idea, reverted in 1.1.7 to avoid name collisions
- import_gds can snap points to a design grid


# 1.1.5 2020-03-17

- added pre-commit hook for code consistency
- allows a list of cladding layers
- all layers are defined as tuples using pp.LAYER.WG, pp.LAYER.WGCLAD


# 1.1.4 2020-02-27

- bug fixes
- new coupler with less snaping errors
- adding Klayout generic DRC rule deck

# 1.1.1 2020-01-27

- first public release

# 1.0.2 2019-12-20

- test components using gdshash
- new CLI commands for `pf`
    - pf library lock
    - pf library pull

# 1.0.1 2019-12-01

- autoplacer and yaml placer
- mask_merge functions (merge metadata, test protocols)
- added mask samples
- all the mask can be build now from a config.yml in the current directory using `pf mask write`

# 1.0.0 2019-11-24

- first release
