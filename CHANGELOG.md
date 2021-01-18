# CHANGELOG

## TODO

- document klayout placers
- bundle routing with specific waypoints
- connect_with_waypoints can accept ports

Maybe:

- extract netlist from layout markers
- add grating coupler vertical ports
- create a Klayout library so we can also place components from the klayout menu GUI (available for UBC sample pdk)
- add contracts, or enforcers, either in cell decorator or using

```
from contracts import contract
from pydantic import validate_arguments

@contract(length='Real,>=0', width='float,>=0')

```

## master branch (latest changes not released yet)

## 2.2.5 2021-01-13

- added common types in pp.types
- added simulation settings in name of sparameters
- store Sparameters in .csv as well as in Lumerical interconnect format .dat
- reduce some type errors when running mypy
- fix error in u_bundle_direct_routes for a single route (thanks to tvt173)
- When a component has both a placement and a connection are defined and transform over a component, we raise an error.
- add Component().plot() in matplotlib and Component.show() in Klayout
- clear_cache when running plot() or show(). Useful for Jupyter notebooks
- add logo


## 2.2.4 2020-12-25

- get_netlist() returns a dict. Removed recursive option as it is not consistent with the new netlist extractor in pp/get_netlist.py. Added name to netlist.
    - fixed get_netlist() placements (using origin of the reference instead of x, y which refer to the center). Now we can go back and forth from component -> netlist -> component
    - If there is a label at the same XY as the reference it gets the name from that label, the issue was that we need to add the labels after defining connections in component_from_yaml
- ListConfig iterates as a list in _clean_value
- test component.get_netlist() -> YAML-> pp.component_from_yaml(YAML) = component (both for settings_changed and full_settings)
- add pp.testing with difftest(component) function for boolean GDS testing.
- improved placer documentation and comments in pp/samples/mask/does.yml

## 2.2.3 2020-12-19

- store config.yml in mask build directory (reproduce environment when building masks)
- add tests for add_fiber_single and add_fiber_array labels
- snap name to 1nm grid, try to name it without decimals when possible (L1p00 -> L1)
- more sensitive defaults parameter names for MZI (coupler -> splitter)
- sim settings outputs in YAML file
- fix sparameters sorting of ports when in pp.sp.read_sparameters
- pp.get_netlist() returns top level ports for a component
- output parameters that change in component (c.polarization='te') in get_settings()['info']
- fixed bug in get_settings to clean tuple settings export

## 2.2.2 2020-12-06

- rename coupler ports inside mzi function


## 2.2.1 2020-12-06

- pp.qp hides DEVREC layer
- test netlist of `_circuits`
- sort the keys when loading YAML file for test_netlists
- better docstrings
- add function_name to container
- remove duplicated keys for container
- pp.clear_cache() in pytest fixture in `pp/conftest.py`
- fixed pp.clear_cache() by using a global variable.
- added lytest tests, which test GDS boolean diffs using klayout
- fixed `pf diff` to show (diffs, common, only_old, only_new, old, new) using same layers in different cells. Thanks to Troy Tamas.
- removed `pins` argument from cell decorator as it changes the geometry of a cell with the same name (it was problematic).
- new recurse_instances function. No need to track connections in a global netlist dict. We can extract netlist connections from devices sharing ports.
- component_from_yaml adds label with. instance name. Thanks to Troy Tamas.
- write a pp.add_pins_to_references that adds pins and labels to references.
- make sure @cell decorator checks that it returns a Component
- remove other types of units conversions from
- better type hints
- export hierarchical and flat netlists
- rename 0.5 as 500n (it makes more sense as default units are in um) and submicron features are named in nm
- remove other renames
```
if 1e12 > value > 1e9:
    value = f"{int(value/1e9)}G"
elif 1e9 > value > 1e6:
    value = f"{int(value/1e6)}M"
elif 1e6 > value > 1e3:
    value = f"{int(value/1e3)}K"
if 1 > value > 1e-3:
    value = f"{int(value*1e3)}n"
elif 1e-6 < value < 1e-3:
    value = f"{int(value*1e6)}p"
elif 1e-9 < value < 1e-6:
    value = f"{int(value*1e9)}f"
elif 1e-12 < value < 1e-9:
    value = f"{int(value*1e12)}a"
else:
    value = f"{value:.2f}"
```


## 2.2.0 2020-11-29

- component_from_yaml updates:
    - placements:
        - port: you can define an anchor port
        - dx: delta x
        - dy: delta x
        - mirror: boolean or float (x axis for the mirror)
        - x: number or can also be a port from another instance
    - routes:
        - you can define a route range (left,E:0:3 : rigth,W:0:3)
- connect bundle is now the default way to connect groups of ports in component_from_yaml
- coupler: can change the vertical distance (dy) between outputs
- replace @pp.autoname with @pp.cell as a decorator with cells options (autoname, pins ...)

## 2.1.4 2020-11-14

- fixed installer for windows using copy instead of symlink

## 2.1.3 2020-11-09

- `pf install` installs klive, generate_tech and gitdiff
- `pf diff` makes boolean difference between 2 GDS files

## 2.1.2 2020-11-09

- write conda environment.yml so you can `make conda` to install the conda environment
- setup.py installs klive, generate_tech and gitdiff

## 2.1.0 2020-11-09

- test lengths for routes
- pytest are passing now for windows
    - Fixed the spiral circular error by snapping length to 1nm (windows only)
    - Testing now for windows and linux in the CICD
    - Made the multiprocessing calls pickeable by removing the logger function (that wasn't doing much anyway)
- extend_ports: maintains un-extended ports

## 2.0.2 2020-11-03

- fixing sorting of ports in bundle routing: Thanks to Troy Tamas
- added `factory: optical` and `settings:` in component_from_yaml routes
- write more container metadata for component inside the container (function_name, module ....)
- more checks for the grating coupler decorator (W0 port with 180 degrees orientation)
- CI/CD tests run also on pull requests
- added pp.clear_cache() and call it when we run `pp.show(c)`
- use pp.clear_cache() when testing component port positions


## 2.0.0 2020-10-30

- addded grating coupler decorator to assert polarization and wavelength
- component_from_yaml function allows route filter input
- routes_factory: in pp.routing (optical, electrical)
- routes: in component_from_yaml allows route_factory
- no more routes and route_bundles: now it's all called routes, and you need to specify the routing factory function name [optical, electrical ...]
- renamed component_type2factory to component_factory
- explained factory operation in notebooks/02_components.ipynb
- mzi.py DL is now the actual delta_length

## 1.4.4 2020-10-14

- improved notebooks (thanks to phidl tutorial)
- added C and L components from phidl
- print(component) returns more info (similar to phidl)
- support new way of defining waveguides with pp.Path, pp.CrossSection and pp.path (thanks to phidl)

## 1.4.3 2020-10-07

- clean metadata dict recursively

## 1.4.2 2020-10-07

- renamed add_io_optical to add_fiber_array
- added taper factory and length to add_fiber_single
- fixed JSON metadata for Components with function kwargs
- fixed reference positions in component_from_yaml
- added bundle_routes option in component_from_yaml

## 1.4.0 2020-10-04

- Works now for python>=3.6, before only worked for python3.7 due to [type annotations](https://www.python.org/dev/peps/pep-0563/)
- nicer netlist representations (adding location to each node in the graph)
- YAML loader accepts strings (no more io.StringIO)
- better docs
- add_tapers only tapers optical ports in the new containered component
- add_ports from polygon markers
- add_io_optical maintains other ports
- added single fiber routing capabilities (pp.routing.add_fiber_single)
- added Component.copy()
- added basic electrical routing capabilities
    - pp.routing.add_electrical_pads
    - pp.routing.add_electrical_pads_top
    - pp.routing.add_electrical_pads_shortest

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
- any component decorated with @pp.cell can accept `pins=True` flag, and a function `pins_function`.
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
- cell sorts kwarg keys by alphabetical order
- added cell tests
- cell accepts max_name_length and ignore_from_name kwargs
- pp.generate_does raises error if component does not exist in factory
- replaces name_W20_L30 by name_hash for cell names  > 32
- zz_conn cleaner name using `from pp.cell import clean_name` no slashes in the name
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
