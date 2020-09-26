# CHANGELOG

## TODO

- extract netlist from layout using PINS and check if connectivity matches intent
- create a Klayout library so we can also place components from the klayout menu GUI

## Maybe

- units are in um by default, so instead of using m to symbolize (1e-3) maybe just use nm

## master

- Works now for python3.6 and python3.7, before only working for python3.7 due to [type annotations](https://www.python.org/dev/peps/pep-0563/)
- nicer netlist representations (adding location to each node in the graph)
- YAML loader accepts strings (no more io.StringIO)
- better docs
- works for python>=3.6

## 1.3.2 2020-08-15

- improve sparameters tutorial
- fixed some issues when using `x = x or x_default` not valid for `x=0`
- added tests for splitter_tree and splitter_chain


## 1.3.1 2020-07-27

- get_netlist by default return a simpler netlist that captures only settings different from default. Full netlist component properties available with `full_settings=True`.
- limited pytest scope to netlist build tests to avoid weird side effects that move ports locations from test_component_ports
- sphinx==1.3.2 in requirements.txt

## 1.3.0 2020-07-26

- `Component.get_netlist()` returns its netlist
- `Component.plot_netlist()` renders netlist graph
- `component_from_yaml` accepts netlist
- routing jupyter notebooks
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
