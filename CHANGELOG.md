# CHANGELOG

## TODO

- had to ignore settings for netlist defined components to pass regression tests (ring_double_bus, coupler). Maybe we can define those components not as netlist or maybe standarize the netlist definition.
- create a Klayout library so we can also place components from the klayout menu GUI
- extract and represent netlists

## Maybe

- units are in um by default, so instead of using m to symbolize (1e-3) maybe just use nm

## Most recent changes with no release yet

- manhattan text can have cladding

## 1.2.1 2020-07-05

- replaced hiyapyco with omegaconf (better YAML parser that can handle number with exponents 1e9)
- separated conf (important to be saved) from CONFIG that contains useful paths

## 1.2.0 2020-07-04

- added link for [ubc PDK](https://github.com/gdsfactory/ubc) to README
- added a jupyter notebook tutorial for references and array of references
- added dbr and cavity components
- rotate is now a container
- addapted pp.pack from phidl as an easier way to pack masks
- Autoname also has now a build in cache to avoid having two different cells with the same name
- added type annotations

## 1.1.9 2020-05-13

- write and read Sparameters
- pp.extend_ports is now a container
- any component decorated with @pp.autoname can accept `pins=True` flag, and a function `pins_function`.
- Pins arguments will be ignored from the Component `name` and `settings`
- better json serializer for settings
- added units to names (m,K,G ...)

## 1.1.8 2020-05-11

- leaf components (waveguide, bend, mmi ...) have now pins, for circuit simulation

## 1.1.7 2020-05-07

- flake8 is passing now
- added flake8 to pre-commit hook
- simpler JSON file for mask metadata mask.tp.json
- added container decorator, can inherit ports, settings, test and data analysis protocols and still have a different name to avoid name collisions
- samples run as part of the test suite, moved samples into pp
- autoname sorts kwarg keys by alphabetical order
- added autoname tests
- autoname accepts max_name_length and ignore_from_name kwargs
- pp.generate_does raises error if component does not exist in factory
- replaces name_W20_L30 by name_hash for cell names  > 32
- zz_conn cleaner name using `from pp.name import clean_name` no slashes in the name
- add_io is a container
- write labels settings in the middle of the component by default, you can always turn it off by adding `config.yml` in your project
- added pytest-regression for component setting and ports

```
with_settings_label: False

```

## 1.1.6 2020-04-11

- mask JSON works with cached GDS files for the klayout placer
- added layers to CONFIG['layers']
- write_labels gets layer from `CONFIG['layers']['LABEL']`
- add_padding works over the same component --> this was not a good idea, reverted in 1.1.7 to avoid name collisions
- import_gds can snap points to a design grid


## 1.1.5 2020-03-17

- added pre-commit hook for code consistency
- waveguide and bend allow a list of cladding layers
- all layers are defined as tuples using pp.LAYER.WG, pp.LAYER.WGCLAD


## 1.1.4 2020-02-27

- bug fixes
- new coupler with less snaping errors
- adding Klayout generic DRC rule deck

## 1.1.1 2020-01-27

- first public release

## 1.0.2 2019-12-20

- test components using gdshash
- new CLI commands for `pf`
    - pf library lock
    - pf library pull

## 1.0.1 2019-12-01

- autoplacer and yaml placer
- mask_merge functions (merge metadata, test protocols)
- added mask samples
- all the mask can be build now from a config.yml in the current directory using `pf mask write`

## 1.0.0 2019-11-24

- first release
