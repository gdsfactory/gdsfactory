# CHANGELOG

## TODO

- enable routing with 180euler and Sbends
- fix FIXMEs
- example on how to customize visualization of a component
- remove kwargs from most components as a way to customize cross_sections to get more intuitive error messages

Maybe:

- gf.component_from_yaml has `cache` decorator
- pads have a port with `pad` name on the center of the pad
- replace LIBRARY with `get_factory_dict`
- rename any thickness_nm to thickness or zmin_nm to zmin
- define Layer as a dataclass instead of Tuple[int, int]
- move add_fiber_array settings into a dataclass
- enable difftest for test_containers
- xdoctest
- mypy passing
- enable add pins option in TECH that can add custom pins to components
- how can we have labels with gdslayer, gdspurpose? Phidl issue?
- write function that generates GDS Klayout library
- pass force-regen flag from pytest
- add contracts cell decorator

```
from contracts import contract
@contract(length='Real,>=0', width='float,>=0')


```
## 3.4.4

- decorators that return new component also work in cell

## 3.4.3

- enable `Component.move()` which returns a new Component that contains a moved reference of the original component
- add `Port._copy()` that is the same as `Port.copy` to keep backwards compatibility with phidl components
- adapt some phidl.geometry boolean operations into `gdsfactory.geometry`
- move some functions (boolean, compute_area, offset, check_width ... ) into `gdsfactory.geometry`
- add `gdsfactory.geometry.boolean` for klayout based boolean operations
- add pydantic validator for `ComponentReference`
- max_name_length is a cell decorator argument used when importing gds cells
- add `geometry.boolean_klayout`


## 3.4.2

- `import_gds` also shares the cell cache
- remove `name_long` from `cell` decorator
- remove `autoname` from `cell` decorator args
- `Component.show()` shows a component copy instead of a container
- remove `Component.get_parent_name()` and replace it with `Component.child_info.name`
- gf.path.extrude adds cross_section.info and path.info to component info

## 3.4.0

- gf.component_from_yaml accepts info settings
- make sure that zero length paths can be extruded without producing degenerated boundaries. They just have ports instead of trying to extrude zero length paths.
- snap.assert_on_2nm_grid for gap in mmi1x2, mmi2x2, coupler, coupler_ring
- gf.Component.rotate() calls gf.rotate so that it uses the Component CACHE
- add `tests/test_rotate.py` to ensure cache is working
- add cache to component_from_yaml
- add `tests/test_component_from_yaml_uid.py`
- ensure consitent name in YAML by hashing the dict in case no name is provided
- `component.settings` contains input settings (full, changed, default)
- `component.info` contains derived settings (including module_name, parent settings, ...)
- `component.to_dict` returns a dict with all information (info, settings, ports)
- rename `via_stack` to `contact`

## 3.3.9

- move `gf.component_from_yaml` to `gf.read.from_yaml`
- unpin triangle version in requirements.txt
- `cell` components accept info settings dict, for the components

## 3.3.8

- add `auto_widen` example in tutorials/routing
- add `plugins` examples in tutorials/plugins
- Component.rotate() returns a new Component with a rotated reference of itself
- increase simulation_time in lumerical `simulation_settings` from 1ps to 10ps, so max simulation region increased 10x
- write_sparameters_lumerical returns session if run=False. Nice to debug sims.
- make consitent names in gf.read: `gf.read.from_phidl` `gf.read.from_picwriter` `gf.read.from_gds`

## 3.3.5

- `route_manhattan` ensures correct route connectivity
- replace `bend_factory` by `bend` to be more consistent with components
- replace `bend90_factory` by `bend90` to be more consistent with components
- replace `straight_factory` by `straight` to be more consistent with components
- replace `get_route_electrical_shortest_path` by `route_quad`
- gf.components.array raises error if columns > 1 and xspacing = 0
- gf.components.array raises error if rows > 1 and yspacing = 0
- simplify `gf.components.rectangle` defintion, by default it gets 4 ports
- containers use Component.copy_settings_from(old_Component), and they keep their parent settings in `parent`, as well as `parent_name`
- `Component.get_parent_name()` returns the original parent name for hierarchical components and for non-hierarchical it just returns the component name

## 3.3.4

- containers use `gf.functions.copy_settings` instead of trying to detect `component=` from kwargs
- `Port._copy()` is now `Port.copy()`
- bend_euler `p=0.5` as default based on this [paper](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-8-9150&id=362937)
- rectangle has 4 ports by default (similar to compass), it just includes the `centered` parameter
- gf.grid accept component factories as well as components and is a cell

## 3.3.3

- fix cutback_component bend
- add `gf.routing.route_quad`

## 3.3.2

- add `gdsfactory.to_3d.to_stl`

## 3.3.1

- adjust z position for lumerical simulation region as well as port locations
- `Component.show()` and `Component.plot()` do not clear_cache by default (`clear_cache=False`)

## 3.3.0

- write_sparameters in lumerical writes simulation_settings in YAML
- replace port_width with port_margin in simulation_settings
- rename `Component.get_porst_east_west_spacing` as `Component.get_ports_ysize()`
- add `Component.get_ports_ysize()`
- fix `mzi` `with_splitter`
- enable `vars` variables in component_from_yaml
- gdsdiff accepts test_name, and uses the path of the test_file for storing GDS files
- add functools cache decorator for gdsfactory.import_gds and gdsfactory.read.gds
- rename cache with lru_cache(maxsize=None) to keep compatibility with python3.7 and 3.8
- update to phidl==1.6.0 and gdspy==1.6.9 in requirements.txt
- new gf.path.extrude adapted from phidl

## 3.2.9

- rename `component_from` to `read`
- remove `gf.bias`
- remove `gf.filecache`
- add `get_layer_to_sidewall_angle` in layer_stack
- rename `gf.lys` to `gf.layers.LAYER_SET` to be consistent

## 3.2.8

- array with via has consistent names

## 3.2.7

- write_sparameters exports the layer_stack together with the simulation_settings
- write simulation_settings with omegaconf instead of YAML. Layer tuples were not exporting correctly.
- layer_stack inherits from dict
- simulation files use get_name_short to keep the name of the suffix within 32 characters + 32 characters for the name. This keeps filepath with less than 64 characters

## 3.2.6

- add_ports_from_labels accepts layer parameter for the port

## 3.2.5

- add add_ports_from_labels function

## 3.2.4

- transition raises ValueError if has no common layers between both cross_sections
- grating coupler wavelength in um to be consistent with all units in gdsfactory
- rename thickness_nm to thickness and zmin_nm to zmin in layer_stack to be consistent with gdsfactory units in um
- rename Ppp to PPP, Npp to NPP to be consistent with nomenclature
- simulation_settings in um to be consistent with all units in gdsfactory being in um

## 3.2.3

- fix gf.to_trimesh
- add `Floats`, `Float2` and `Float3` to types
- add kwargs for component example documentation from signature

## 3.2.2

- add `gf.to_trimesh` to render components in 3D
- replace dx, dy by size in bend_s, and spacing by dx, dy in splitter_tree

## 3.2.1

- simplify contact_with_offset
- contact_with_offset use array of references
- add `gf.components.taper_cross_section` to taper two cross_sections

## 3.2.0

- Ensures that an impossible route raises RouteWarning and draws error route with markers and labels on each waypoint

## 3.1.10

- fix add fiber single for some cases
- create `strip_auto_widen` cross_section with automatic widening of the waveguide
- add `add_grating_couplers_with_loopback_fiber_single`

## 3.1.9

- pad_array and array use array of references, accept columns and rows as args

## 3.1.8

- contact uses array of references

## 3.1.7

- transition ports have different cross_sections
- get_bundle separation is now defined from center to center waveguide
- contact has 4 ports, consistent with pads
- pad takes size argument instead of (width, height), which is consistent with other rectangular structures
- add filecache to store in files

## 3.1.6

- add `Component.write_netlist_dot` to write netlist graph in dot format
- add handling of separation keyword argument to get_bundle_from_waypoints (thanks to Troy @tvt173)

## 3.1.5

- raise ValueError when moving or rotating component. This avoids modifying the state (position, rotation) of any Component after created and stored in the cell cache.
- add cross_section property to ports
- `gdsfactory/routing/fanout.py` passes cross_section settings from port into bend_s
- fix manhattan text, avoid creating duplicated cells
- fix pcm_optical

## 3.1.4

- remove limitation from get_bundle_from_waypoints is that it requires to have all the ports lined up.

## 3.1.3

- because in 3.1.1 cells can accept `*args` containers now are detected when they have `Component.component`
- rename `component.settings['component']` to `component.settings['contains']`
- grating couplers have port with `vertical_te` or `vertical_tm` prefix
- container keep the same syntax
- `add_fiber_array` allows passing `gc_port_labels`
- `add_fiber_array` and `add_fiber_single` propagate any non-optical ports to the container
- fix ports transitions and raise error when saving gdsfile with duplicated cell names

## 3.1.2

- add `make doc` to update components documentation
- add `routing.get_route_electrical` with sensitive defaults for routing electrical routes `bend=wire_corner`
- `components.pad_array_2d` names `e{row}_{col}`
- `components.pad_array` names `e{col}`

## 3.1.1

- cells accept `*args`
- `@cell` autonaming includes the complete keyword arguments keys (not only the first letter of each argument)
- fix straight_pin and straight_heater_doped length when they have tapers
- waveguide template defaults to euler=True for picwriter components (spiral)
- add `Component.get_ports_xsize()`
- add `toolz` library to requirements

## 3.1.0

- move components python files to the same folder
- add components.write_factory function to generate dict
- added filecmp for testing components widht difftest, only does XOR if files are different. This speeds the check for larger files.

## 3.0.3

- change port naming convention from WNES to o1, o2, o3 for optical, and e1, e2, e3, e4 for electrical
- add Component.auto_rename_ports()
- add `ports_layer` property to Component and ComponentReference to get a map
- `Component.show()` `show_ports` and `show_subports` show a container (do not modify original component)
- add port_types to cross_section
- rename straight_horizontal_top to straight_x_top

## 3.0.2

- add straight_rib, straight_heater_metal and straight_heater_doped
- `xs2 = gf.partial(cross_section)` does not require defining `xs2.__name__`
- replace gf.extend[.] with gf.components.extension.
- Component.show() uses `add_pins_triangle` as default to show port orientation
- add gf.comtainers.bend_port
- get_netlist considers x,y,width to extract port connectivity

## 3.0.1

- pass cross_section functions to component factories instead of registering waveguides in TECH.waveguide
- snap_to_grid is now a cross_section property
- replace Waveguide with gdsfactory.cross_section functions
- add pydantic.validate_arguments to cross_section
- functools.partial have unique names
- partial functions include settings for JSON and name
- include xor flag when doing a gdsdiff
- delete StrOrDict, you can use functools.partial instead to customize functions
- include --xor flag to `gf gds diff --xor` CLI to run a detailed XOR between 2 GDS files

## 3.0.0

- rename `pp` to `gdsfactory`
- recommend `import gdsfactory as gf`
- rename `pf` CLI to `gf`

## 2.7.8

- rename post_init to decorator
- add pp.layer.load_lyp_generic
- load_lyp, alpha=1 if visible = 'false'
- LayerStack is now List[LayerLevel] and has no color information

## 2.7.7

- remove taper_factory from pp.routing.add_fiber_array and pp.routing.add_fiber_single
- pp.Component.add_ports(port_list, prefix) to avoid adding duplicated port names
- add pp.components.litho_ruler
- @cell has `post_init` function. Perfect for adding pins
- update `samples/pdk/fabc.py` with partial
- Library can register partial functions
- `contact_with_offset` is now define with via functions instead of StrOrDict, skip it from tests
- add `pp.components.die_box`

## 2.7.6

- add Component.to_dict()
- add pp.config.set_plot_options for configuring matplotlib
- add pp.Component.add_ports(port_list)
- enable in pp.name the option of passing a partial function
- create partial notebook (for functional programming) demonstrating hierarchical components with customized subcomponent functions
- revert mzi and mzi_lattice to 2.5.3 (functional programming version)
- delete mzi_arm, mzi2x2 and mzi1x2
- add mzi_phase_shifter
- add wire_sbend
- pp.add_tapers back to functional programming

## 2.7.5

- fix preview_layerset
- extension_factory default extension layer depends on the port
- add extend_ports_list to pp.extend
- add simulation_settings to pp.write

## 2.7.4

- get_bundle_corner passing waveguide (consistent with other routes)
- fix pp.components.wire_corner
- delete pp.components.electrical.wire.py

## 2.7.3

- pp.grid allows accessing references from Component.aliases
- pp.routing.add_fiber_single and pp.routing.add_fiber_array accept get_input_label_text_loopback, get_input_label_text params

## 2.7.2

- fix print_config asdict(TECH)
- cell decorator validates arguments by default using pydantic, cell_without_validator does not
- add pydantic.vaidate method to Port

## 2.7.1

- add pp.components.die
- fix spiral_external_io
- add_fiber_array also labels loopbacks
- rename with_align_ports as loopback

## 2.7.0

- round_corners raises RouteWarning if not enough space to fit a bend

## 2.6.10

- contact has port with port_type=dc

## 2.6.9

- rename tlm to contact and tlm_with_offset to contact_with_offset

## 2.6.8

- add pp.c.tlm_with_offset
- mzi adds any non-optical ports from straight_x_bot and straight_x_top
- ignore layer_to_inclusion: Optional[Dict[Layer, float]] from get_settings

## 2.6.7

- pp.sp.write has a logger
- via has optional pitch_x and pitch_y

## 2.6.6

- add pp.extend to pp
- fix pp.extend.extend_port, propagates all settings
- pp.gds.read_ports_from_markers accepts a center (xc and yc) for guessing port orientation
- import_gds only accessible from pp.gds.import_gds
- merge assert_grating_coupler_properties and version in pp.asserts.
- created pp.component_from module
- rename pp.import_phidl_component and pp.picwriter_to_component pp.component_from.phidl and pp.component_from.picwriter
- rename pp.load_component to pp.component_from.gds
- rename pp.netlist_to_component to pp.component_from.netlist. added a DeprecationWarning
- move set_plot_options to pp.klive.set_plot_options, stop overriding phidl's set_plot_options in `pp.__init__`
- move pp.merge_cells into pp.component_from.gdspaths and pp.component_from.gdsdir
- waveguide accepts dict(component='fabc_nitride_cband')
- add pp.remove and pp.read
- remove pp.gds

## 2.6.5

- add pp.routing.get_route_from_steps as a more convenient version of pp.routing.get_route_from_waypoints

## 2.6.4

- pp.components.mzi accepts straight_x_bot and straight_x_bot parameters
- pad array has axis='x' argument
- expose utils, sort_ports and fanout in pp.routing

## 2.6.3

- add min_length Waveguide setting, for manhattan routes and get_bundle
- remove grating_coupler.xmin = 0 inside the route_fiber function

## 2.6.2

- add pp.c.delay_snake2 and pp.c.delay_snake3
- rename FACTORY as LIBRARY. Now we have LIBRARY.factory as the Dict[str, Callable] with all the library functions.
- LIBRARY.get_component adds component.\_initialized=True attribute only once

## 2.6.0

- cell decorator propagates settings for Component, only if isinstance(kwargs['component'], Component)
- get_route_from_waypoints adds port1 and port2 waypoints automatically
- get_route_from_waypoints accepts waveguide and waveguide_settings
- add_port warns you when trying to add ports with off-grid port points
- fix add_fiber: not passing factory to get_bundle
- Factory(name=) has required argument name
- Factory has `__str__` and `__repr__`
- add_port(width=) width automatically snaps width to 1nm grid
- add DeprecationWarning to get_routes
- update pipfile
- remove conda environment.yml as it was out of date
- add automatic release of any tag that starts with v

## 2.5.7

- Component.show(show_ports=True) adds port names and pins by default (before show_ports=False)
- splitter_tree, also propagates extra coupler ports
- add_ports_from_markers has an optional `port_layer` for the new created port.
- component_settings = OmegaConf.to_container(component_settings, resolve=True)
- pp.c.pad_array consistent parameters with pp.c.array (pitch_x)

## 2.5.6

- better error messages for off-grid ports, add suggestions for fixes
- Component.validator `assert len(name) <= MAX_NAME_LENGTH`, before `assert len(name) < MAX_NAME_LENGTH`

## 2.5.5

- update to omegaconf=2.1.0
- add loguru logger
- added pydantic validator to Component
- pp.add_tapers.add_tapers can accept taper port names
- add_tapers, add_fiber_array, add_fiber_single accepts taper with StrOrDict
- components accept waveguide StrOrDict
- some names were having 33 characters, fixed max characters name

## 2.5.4

- add `pf gds` CLI commands for `merge_gds_from_directory`, `layermap_to_dataclass`, `write_cells`
- component_from_yaml has a get_bundle_from_waypoints factory
- add heater with single metal
- fix routing with cross-sections with defined Sections
- add TECH.rename_ports
- add pp.containers.
- mzi accepts a factory and can accept StrOrDict for for leaf components
- Factory(post_init=function). Useful for adding pins when using Factory.get_component()

## 2.5.3

- enable fixed timestamp in saved cells, which allows having the same hash for files that do not change

## 2.5.2

- fixed pp.import_phidl_component and added test

## 2.5.1

- compatible with latest version of phidl (1.5.2)
- renamed routing functions
- reduced routing functions functions in pp.routing
- better error messages for waveguide settings (print available keyword args)
- fixed cell decorator to raise Error if any non keyword args defined
- pin more requirements in requirements.txt
- added pur (pip update requirements) in a separate workflow to test gdsfactory with bleeding edge dependencies

## 2.5.0

- add pp.routing.sort_ports
- add pp.routing.get_route_sbend for a single route
- add pp.routing.get_route_sbend_bundle for a bundle of Sbend routes
- rename start_ports, end_ports with ports1 and ports2
- straight_with_heater fixed connector
- straight_with_heater accepts port_orientation_input and port_orientation_output
- TECH defined in config.yml
- refactor pp.path.component to pp.path.extrude
- write to GDS again even if component already has a component.path
- define all TECH in tech.py dataclasses and delete Tech, and Pdk
- add pp.routing.fanout
- add Factory dataclass
- fix pp.routing.routing \_gradual_bend
- add TestClass for Component
- fix get_bundle for indirect routing
- get_netlist returns cleaned names for components (-3.2 -> m3p2)
- add pp.assert_version
- fix naming for components with long funcnames (already over 24 chars + 8 chars for name hash) to keep name shorter than 32 chars
- add pydantic validate_arguments decorator. @pp.cell_with_validator

```
from pydantic import validate_arguments
@validate_arguments
```

## 2.4.9

- rename pp.plot file to pp.set_plot_options to avoid issues with pp.plot function

## 2.4.8

- remove `pins` kwargs from ring and mzi
- simpler coupler straight
- renamed `_get` to `get` in LayerStack
- import_gds raise FileNotFoundError
- import_gds sends gdspy str(gdspath)
- remove pp.plotgds, as you can just do component.plot()
- add pp.set_plot_options() to be consistent with latest 1.5.0 phidl release

## 2.4.7

- better README
- get_settings exports int if possible (3.0 -> 3)
- add cross_section pin for doped waveguides
- Raise error for making transition with unnamed sections
- store component settings in YAML as part of tech.component_settings
- add add_padding_to_size function
- simplify add_pins function. Replace port_type_to_layer with simple layer and port_type kwargs
- add Pdk.add_pins()
- replace pp.write_gds(component, gdspath) with component.write_gds(gdspath)
- replace pp.write_component(component, gdspath) with component.write_gds_with_metadata(gdspath)
- rename pp.components.waveguide to pp.components.straight
- rename auto_taper_to_wide_waveguides auto_widen
- rename wg_heater_connected to straight_with_heater

## 2.4.6

- more consistent names on component factories
- add simulation_settings to Tech
  - sparameters_path: pathlib.Path = CONFIG["sp"]
  - simulation_settings: SimulationSettings = simulation_settings
  - layer_stack: LayerStack = LAYER_STACK
- add Pdk.write_sparameters()

## 2.4.5

- better docstrings
- simplify code for pp.path.smooth
- replace `pp.c.waveguide()` by `pp.components.waveguide()`. `pp.c.waveguide()` still works.
- replace `pp.qp()` by `pp.plot()` to be consistent with `c = Component()` and `c.plot()`
- added `get_component_from_yaml` Pdk class

## 2.4.4

- add vertical_te and vertical_tm ports to grating couplers
- remove klive warning when not klayout is not open (if klayout is not installed or running it will just fail silently)
- replace cladding for bend_circular and bend_euler with square cladding
- added `component.show(show_ports=True)`
- added `component.show(show_subports=True)`
- added `pf merge-cells`
- added `auto_taper_to_wide_waveguides` option to add_fiber_array
- `add_padding` returns the same component, `add_padding_container` returns a container with component inside
- remove container decorator, containers are just regular cells now with @cell decorator
- add `add_pin_square_double` and make it the default

## 2.4.3

- consistent port naming path.component extrusion

## 2.4.2

- better docs

## 2.4.0

- euler bends as default (with_arc_floorplan=True)
- define bends and straighs by path and cross_section
- tech file dataclass in pp.config
- added pp.pdk with tests
- include notebooks in docs with nbsphinx
- regression test for labels
- fixed CACHE key value by using the actual cellname

## 2.3.4

- gdsdiff does not do booleans by default
- pin pre-commit versions

## 2.3.3

- added pp.components.path to easily extrude CrossSections
- added more pp.types (ComponentFactory, RouteFactory) as Callable[..., Component]
- Load a LayerSet object from a Klayout lyp file
- clean lyp from generic tech
- bend_euler accepts similar parameters as bend_circular (layers_cladding, cladding_offset)
- renamed bend_euler90 as bend_euler
- components adapted from picwriter take more similar values (layer_cladding, layer)
- pp.difftest can step over each GDS file with a geometric difference and decide what to do interactively
- adapted pp.path and pp.cross_section from phidl

## 2.3.2

- fixed some mypy errors
- added dx to coupler
- bezier has now number of points as a parameter
- improved docs
- allow to set min and max area of port markers to read

## 2.3.1

- refactor
  - connect_strip to get_route
  - connect_bundle to get_bundle
  - connect_strip_way_points to get_route_from_waypoints
- make diff shows all difference from the difftest run
- snap length to 1nm in route waveguide references
- remove any waveguide reference on the routes which have a 1nm-snapped length equal to zero

## 2.3.0

- move tests to tests/ folder
- rename from `from pp.testing import difftest` to `from pp.difftest import difftest`
- remove pp.container containerize
- better type annontations
- replace some `c.show()` by a simpler `c.show()`

## 2.2.9

- better settings export
- fixed docs

## 2.2.8

- flat routes with no more zz_conn cells
- added from pp.import_gds import add_ports_from_markers_square

## 2.2.7

- using mirror (port) in pp.component_from_yaml
- remove old, untested code to deal with libraries. Libraries should use factory
- add pp.get_name_to_function_dict to build factories as dict(func_name=func)
- component_from_yaml can also use (north, east, west, ne, nw ...) for placement
- added regression tests for component_from_yaml

## 2.2.6

- added badges from github in README (codecoverage, docs ... )
- pp.import_gds can import and move cells with labels (thanks to Adam McCaughan)
- add margin and min_pin_area_um2 to read_ports_from_markers
- replace grating_coupler decorator with a simpler pp.assert_grating_coupler_properties() function
- rename \_containers to container_names and \_components to component_names
- simplify tests for components, containers and circuits

## 2.2.5

- added common types in pp.types
- added simulation settings in name of sparameters
- store Sparameters in .csv as well as in Lumerical interconnect format .dat
- reduce some type errors when running mypy
- fix error in u_bundle_direct_routes for a single route (thanks to tvt173)
- When a component has both a placement and a connection are defined and transform over a component, we raise an error.
- add Component().plot() in matplotlib and Component.show() in Klayout
- clear_cache when running plot() or show(). Useful for Jupyter notebooks
- add logo

## 2.2.4

- get_netlist() returns a dict. Removed recursive option as it is not consistent with the new netlist extractor in pp/get_netlist.py. Added name to netlist.
  - fixed get_netlist() placements (using origin of the reference instead of x, y which refer to the center). Now we can go back and forth from component -> netlist -> component
  - If there is a label at the same XY as the reference it gets the name from that label, the issue was that we need to add the labels after defining connections in component_from_yaml
- ListConfig iterates as a list in \_clean_value
- test component.get_netlist() -> YAML-> pp.component_from_yaml(YAML) = component (both for settings_changed and full_settings)
- add pp.testing with difftest(component) function for boolean GDS testing.
- improved placer documentation and comments in pp/samples/mask/does.yml

## 2.2.3

- store config.yml in mask build directory (reproduce environment when building masks)
- add tests for add_fiber_single and add_fiber_array labels
- snap name to 1nm grid, try to name it without decimals when possible (L1p00 -> L1)
- more sensitive defaults parameter names for MZI (coupler -> splitter)
- sim settings outputs in YAML file
- fix sparameters sorting of ports when in pp.sp.read_sparameters
- pp.get_netlist() returns top level ports for a component
- output parameters that change in component (c.polarization='te') in settings['info']
- fixed bug in get_settings to clean tuple settings export

## 2.2.2

- rename coupler ports inside mzi function

## 2.2.1

- pp.plot hides DEVREC layer
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

## 2.2.0

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

## 2.1.4

- fixed installer for windows using copy instead of symlink

## 2.1.3

- `pf install` installs klive, generate_tech and gitdiff
- `pf diff` makes boolean difference between 2 GDS files

## 2.1.2

- write conda environment.yml so you can `make conda` to install the conda environment
- setup.py installs klive, generate_tech and gitdiff

## 2.1.0

- test lengths for routes
- pytest are passing now for windows
  - Fixed the spiral circular error by snapping length to 1nm (windows only)
  - Testing now for windows and linux in the CICD
  - Made the multiprocessing calls pickeable by removing the logger function (that wasn't doing much anyway)
- extend_ports: maintains un-extended ports

## 2.0.2

- fixing sorting of ports in bundle routing: Thanks to Troy Tamas
- added `factory: optical` and `settings:` in component_from_yaml routes
- write more container metadata for component inside the container (function_name, module ....)
- more checks for the grating coupler decorator (W0 port with 180 degrees orientation)
- CI/CD tests run also on pull requests
- added pp.clear_cache() and call it when we run `c.show()`
- use pp.clear_cache() when testing component port positions

## 2.0.0

- addded grating coupler decorator to assert polarization and wavelength
- component_from_yaml function allows route filter input
- routes_factory: in pp.routing (optical, electrical)
- routes: in component_from_yaml allows route_factory
- no more routes and route_bundles: now it's all called routes, and you need to specify the routing factory function name [optical, electrical ...]
- renamed component_type2factory to component_factory
- explained factory operation in notebooks/02_components.ipynb
- mzi.py DL is now the actual delta_length

## 1.4.4

- improved notebooks (thanks to phidl tutorial)
- added C and L components from phidl
- print(component) returns more info (similar to phidl)
- support new way of defining waveguides with pp.Path, pp.CrossSection and pp.path (thanks to phidl)

## 1.4.3

- clean metadata dict recursively

## 1.4.2

- renamed add_io_optical to add_fiber_array
- added taper factory and length to add_fiber_single
- fixed JSON metadata for Components with function kwargs
- fixed reference positions in component_from_yaml
- added bundle_routes option in component_from_yaml

## 1.4.0

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

## 1.3.2

- improve sparameters tutorial
- fixed some issues when using `x = x or x_default` not valid for `x=0`
- added tests for splitter_tree and splitter_chain

## 1.3.1

- get_netlist by default return a simpler netlist that captures only settings different from default. Full netlist component properties available with `full_settings=True`.
- limited pytest scope to netlist build tests to avoid weird side effects that move ports locations from test_component_ports
- sphinx==1.3.2 in requirements.txt

## 1.3.0

- `Component.get_netlist()` returns its netlist
- `Component.plot_netlist()` renders netlist graph
- `component_from_yaml` accepts netlist
- routing jupyter notebooks
- manhattan text can have cladding

## 1.2.1

- replaced hiyapyco with omegaconf (better YAML parser that can handle number with exponents 1e9)
- separated conf (important to be saved) from CONFIG that contains useful paths

## 1.2.0

- added link for [ubc PDK](https://github.com/gdsfactory/ubc) to README
- added a jupyter notebook tutorial for references and array of references
- added dbr and cavity components
- rotate is now a container
- addapted pp.pack from phidl as an easier way to pack masks
- Autoname also has now a build in cache to avoid having two different cells with the same name
- added type annotations

## 1.1.9

- write and read Sparameters
- pp.extend_ports is now a container
- any component decorated with @pp.cell can accept `pins=True` flag, and a function `pins_function`.
- Pins arguments will be ignored from the Component `name` and `settings`
- better json serializer for settings
- added units to names (m,K,G ...)

## 1.1.8

- leaf components (waveguide, bend, mmi ...) have now pins, for circuit simulation

## 1.1.7

- flake8 is passing now
- added flake8 to pre-commit hook
- simpler JSON file for mask metadata mask.tp.json
- added container decorator, can inherit ports, settings, test and data analysis protocols and still have a different name to avoid name collisions
- samples run as part of the test suite, moved samples into pp
- cell sorts kwarg keys by alphabetical order
- added cell tests
- cell accepts max_name_length and ignore_from_name kwargs
- pp.generate_does raises error if component does not exist in factory
- replaces name_W20_L30 by name_hash for cell names > 32
- zz_conn cleaner name using `from pp.cell import clean_name` no slashes in the name
- add_io is a container
- write labels settings in the middle of the component by default, you can always turn it off by adding `config.yml` in your project
- added pytest-regression for component setting and ports

```
with_settings_label: False

```

## 1.1.6

- mask JSON works with cached GDS files for the klayout placer
- added layers to CONFIG['layers']
- write_labels gets layer from `CONFIG['layers']['LABEL']`
- add_padding works over the same component --> this was not a good idea, reverted in 1.1.7 to avoid name collisions
- import_gds can snap points to a design grid

## 1.1.5

- added pre-commit hook for code consistency
- waveguide and bend allow a list of cladding layers
- all layers are defined as tuples using pp.LAYER.WG, pp.LAYER.WGCLAD

## 1.1.4

- bug fixes
- new coupler with less snaping errors
- adding Klayout generic DRC rule deck

## 1.1.1

- first public release

## 1.0.2

- test components using gdshash
- new CLI commands for `pf`
  - pf library lock
  - pf library pull

## 1.0.1

- autoplacer and yaml placer
- mask_merge functions (merge metadata, test protocols)
- added mask samples
- all the mask can be build now from a config.yml in the current directory using `pf mask write`

## 1.0.0

- first release
