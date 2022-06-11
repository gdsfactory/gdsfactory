# [CHANGELOG](https://keepachangelog.com/en/1.0.0/)

## [5.9.0](https://github.com/gdsfactory/gdsfactory/pull/446)

- add doe_settings and doe_names to pack_doe and pack_doe_grid
- add with_hash setting to `gf.cell` that hashes parameters. By default `with_hash=False`, which gives meaningful name to component.
- update to tidy3d 1.4.0, add erosion, dilation and sidewall_angle_deg [PR](https://github.com/gdsfactory/gdsfactory/pull/447)


## [5.8.11](https://github.com/gdsfactory/gdsfactory/pull/445)

- validate pdk layers after activate the pdk
- pdk layers, cells and cross_sections are an empty dict by default
- fix [spiral](https://github.com/gdsfactory/gdsfactory/pull/444)

## [5.8.10](https://github.com/gdsfactory/gdsfactory/pull/443)

- add `SHOW_PORTS = (1, 12)` layer.
- document needed layers for the pdk.

| Layer          | Purpose                                                      |
| -------------- | ------------------------------------------------------------ |
| PORT           | optical port pins. For connectivity checks.                  |
| PORTE          | electrical port pins. For connectivity checks.               |
| DEVREC         | device recognition layer. For connectivity checks.           |
| SHOW_PORTS     | add port pin markers when `Component.show(show_ports=True)`  |
| LABEL_INSTANCE | for adding instance labels on `gf.read.from_yaml`            |
| LABEL          | for adding labels to grating couplers for automatic testing. |
| TE             | for TE polarization fiber marker.                            |
| TM             | for TM polarization fiber marker.                            |

## 5.8.9

- [PR](https://github.com/gdsfactory/gdsfactory/pull/440)
  - add default layers to pdk. fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/437)
  - apply default_decorator before returning component if pdk.default_decorator is defined.
- [PR](https://github.com/gdsfactory/gdsfactory/pull/441) Component.show(show_ports=False) `show_ports=False` and use `LAYER.PORT`, fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/438)

## [5.8.8](https://github.com/gdsfactory/gdsfactory/pull/436)

- assert ports on grid works with None orientation ports.

## [5.8.7](https://github.com/gdsfactory/gdsfactory/pull/435)

- bring back python3.8 compatibility

## [5.8.6](https://github.com/gdsfactory/gdsfactory/pull/434)

- remove gf.set_active_pdk(), as we should only be using pdk.activate(), so there is only one way to activate a PDK.
- change default ComponentFactory from 'mmi2x2' string to straight componentFactory.

## [5.8.5](https://github.com/gdsfactory/gdsfactory/pull/433)

- bring back layer validator to ensure DEVREC, PORTE and PORT are defined in the pdk

## [5.8.4](https://github.com/gdsfactory/gdsfactory/pull/430)

- remove default layers dict for pdk.
- validate layers to ensure you define layers for connectivity checks (DEVREC, PORT, PORTE). Fix [comment](https://github.com/gdsfactory/gdsfactory/discussions/409#discussioncomment-2862105). Add default layers if they don't exist [PR](https://github.com/gdsfactory/gdsfactory/pull/432)
- extend ports do not absorb extension references.
- fix filewatcher. Make sure it shows only components that exist.
- Prevent mutation of double-cached cells [PR](https://github.com/gdsfactory/gdsfactory/pull/429)

## [5.8.3](https://github.com/gdsfactory/gdsfactory/pull/422)

- Allow user to specify steps or waypoints in the call to get_bundle
- Add path length matching keyword arguments to functions called by get_bundle

## 5.8.2

- Fix factory default for Pdk.layers [PR](https://github.com/gdsfactory/gdsfactory/pull/418)
- Use shapely's implementation of simplify when extruding paths [PR](https://github.com/gdsfactory/gdsfactory/pull/419)
- fix [issue](https://github.com/gdsfactory/gdsfactory/issues/415) with fill
- fix [issue](https://github.com/gdsfactory/gdsfactory/issues/417) where copying a cross_section, does not include `add_bbox`, `add_pins` and `decorator`

## [5.8.1](https://github.com/gdsfactory/gdsfactory/pull/414)

- add layers as a default empty dict for Pdk
- improve documentation
- mzi uses straight function instead of 'straight' string

## 5.8.0

- works with siepic verification [PR](https://github.com/gdsfactory/gdsfactory/pull/410)
  - cross_section has optional add_pins and add_bbox, which can be used for verification.
    - add `cladding_layers` and `cladding_offset`.
    - cladding_layers follow path shape, while bbox_layers are rectangular.
  - add 2nm siepic pins and siepic DeviceRecognition layer in cladding_layers, to allow SiEPIC verification scripts.
  - add `with_two_ports` to taper. False for edge couplers and terminators.
  - fix ring_double_heater open in the heater top waveguide.
- Make pdk from existing pdk [PR](https://github.com/gdsfactory/gdsfactory/pull/406)
- add events module and events relating to Pdk modifications [PR](https://github.com/gdsfactory/gdsfactory/pull/412)
  - add default_decorator attribute to Pdk. adding pdk argument to pdk-related events
- add LayerSpec as Union[int, Tuple[int,int], str, None][pr](https://github.com/gdsfactory/gdsfactory/pull/413/)
  - add layers dict to Pdk(layers=LAYER.dict()), and `pdk.get_layer`

## [5.7.1](https://github.com/gdsfactory/gdsfactory/pull/403)

- add cross_section_bot and cross_section_top to mzi, fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/402)
- add electrical ports to heater cross_sections, fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/394)

## [5.7.0](https://github.com/gdsfactory/gdsfactory/pull/400)

- tidy3d mode solver accepts ncore and nclad floats.
- add file cache to tidy3d to `gt.modes.find_modes`
- fix get_bundle [issue](https://github.com/gdsfactory/gdsfactory/issues/396)
- clean cross-sections [PR](https://github.com/gdsfactory/gdsfactory/pull/398/files)
- fix N/S routing in route_ports_to_side [PR](https://github.com/gdsfactory/gdsfactory/pull/395)
- Add basic multilayer electrical routing to most routing functions [PR](https://github.com/gdsfactory/gdsfactory/pull/392)
  - Use via_corner instead of wire_corner for bend function
  - Use MultiCrossSectionAngleSpec instead of CrossSectionSpec to define multiple cross sections
  - Avoids refactoring as much as possible so it doesn't interfere with current single-layer routing

## [5.6.12](https://github.com/gdsfactory/gdsfactory/pull/397)

- improve types and docs

## [5.6.11](https://github.com/gdsfactory/gdsfactory/pull/391)

- add python3.6 deprecation notice in the docs [issue](https://github.com/gdsfactory/gdsfactory/issues/384)
- add edge_coupler, edge_coupler_array and edge_coupler_array_with_loopback
- add python3.10 tests

## [5.6.10](https://github.com/gdsfactory/gdsfactory/pull/390)

- add_fiber_single and add_fiber_array tries to add port with `vertical` prefix to the new component. It not adds the regular first port. This Keeps backwards compatibility with grating couplers that have no defined verical ports.
- rename spiral_inner_io functions

## [5.6.9](https://github.com/gdsfactory/gdsfactory/pull/389)

- add_port_from_marker function only allows for ports to be created parallel to the long side of the pin marker. [PR](https://github.com/gdsfactory/gdsfactory/pull/386)

## [5.6.7](https://github.com/gdsfactory/gdsfactory/pull/385)

- fix some pydocstyle errors
- write_gds creates a new file per save
- improve filewatcher for YAML files
- add python_requires = >= 3.7 in setup.cfg

## [5.6.6](https://github.com/gdsfactory/gdsfactory/pull/382)

- `gf yaml watch` uses the same logging.logger
- `gf.functions.rotate` can recenter component [PR](https://github.com/gdsfactory/gdsfactory/pull/381)

## [5.6.5](https://github.com/gdsfactory/gdsfactory/pull/380)

- copy paths when copying components [PR](https://github.com/gdsfactory/gdsfactory/pull/377)
- shear face fixes [PR](https://github.com/gdsfactory/gdsfactory/pull/379)
- fix some pydocstyle
- add port_orientations to gf.components.compass, if None it adds a port with None orientation

## [5.6.4](https://github.com/gdsfactory/gdsfactory/pull/376)

- add_fiber_array adds vertical ports to grating couplers
- add_fiber_single adds vertical ports to grating couplers. Before it was adding only loopback ports.
- import gds fixes [PR](https://github.com/gdsfactory/gdsfactory/pull/374)

## [5.6.3](https://github.com/gdsfactory/gdsfactory/pull/373)

- fix get_labels rotation

## [5.6.2](https://github.com/gdsfactory/gdsfactory/pull/372)

- add `gdsfactory.simulation.tidy3d.modes.sweep_bend_radius`

## [5.6.1](https://github.com/gdsfactory/gdsfactory/pull/371)

- import `load_lyp`

## [5.6.0](https://github.com/gdsfactory/gdsfactory/pull/369)

- add `gf.dft` design for testing, test protocols example in the mask section documentation.
- fix sparameters_meep_mpi [PR](https://github.com/gdsfactory/gdsfactory/pull/366)

## [5.5.9](https://github.com/gdsfactory/gdsfactory/pull/365)

- MaterialSpec for lumerical simulation to address [feature request](https://github.com/gdsfactory/gdsfactory/issues/363)

## [5.5.8](https://github.com/gdsfactory/gdsfactory/pull/364)

- support ports with None orientation

## [5.5.7](https://github.com/gdsfactory/gdsfactory/pull/362)

- fix json schema

## [5.5.6](https://github.com/gdsfactory/gdsfactory/pull/361)

- expose `gf.add_pins` module instead of `add_pins` function. So you can use any of the functions inside the module.
- improve tutorial

## [5.5.5](https://github.com/gdsfactory/gdsfactory/pull/360)

- add `gdsdir` to write_cells CLI command
- rewrite write_cells, before it was writing some empty cells.
- add `debug=False` to add_ports_from_markers_center and remove logger output

## [5.5.4](https://github.com/gdsfactory/gdsfactory/compare/554?expand=1)

- update tidy3d from `1.1.1` to `1.3.2`

## [5.5.3](https://github.com/gdsfactory/gdsfactory/pull/358)

- add `read_metadata` flag to `gf.read.import_gds`
- move dashboard to experimental `requirements_exp` file, that can be install with `pip install gdsfactory[exp]`

## [5.5.2](https://github.com/gdsfactory/gdsfactory/pull/350)

- add `gtidy3d` mode solver

## [5.5.1](https://github.com/gdsfactory/gdsfactory/pull/349)

- waveguide separation in get_bundle_from_waypoints [fix](https://github.com/gdsfactory/gdsfactory/issues/346)
- cell get_metadata [fix](https://github.com/gdsfactory/gdsfactory/issues/348)

## [5.5.0](https://github.com/gdsfactory/gdsfactory/pull/345)

- `gf.read.import_gds()` is now a cell (no more lru cache). LRU cache was not working properly with partial functions.
- add `flatten=False` to cell and decorator
- remove flatten argument `import_gds`
- Component.to_dict() also exports component name
- revert [show changes](https://github.com/gdsfactory/gdsfactory/pull/326/files) as it was causing some files not to reload in klayout.

## [5.4.3](https://github.com/gdsfactory/gdsfactory/pull/344)

- bring back python3.7 compatibility

## [5.4.2](https://github.com/gdsfactory/gdsfactory/compare/542?expand=1)

- add `Pdk.containers` and `Pdk.register_containers`

## 5.4.1

- bring back python3.7 compatibility [PR](https://github.com/gdsfactory/gdsfactory/pull/338)
- rename `vars` to `settings` in `read.from_yaml` [PR](https://github.com/gdsfactory/gdsfactory/pull/339)
  - use settings combined with kwargs for getting component name
- fix mirror isse in `gf.read.from_yaml` [PR](https://github.com/gdsfactory/gdsfactory/pull/341)

## [5.4.0](https://github.com/gdsfactory/gdsfactory/pull/337)

- add `gf yaml watch` folder watcher using watchdog, looking for `pic.yml` files
- add `PDK.register_cells_yaml`

## 5.3.8

- update netlist driven flow tutorial with ipywidgets, so you can live update the YAML and see it in matplotlib and Klayout [PR](https://github.com/gdsfactory/gdsfactory/pull/329)
- [PR fixes problem with showing new layers, not in the previous layer props](https://github.com/gdsfactory/gdsfactory/pull/328)
- [fix show](https://github.com/gdsfactory/gdsfactory/pull/326)
- Fixes gf.show() when gdsdir is passed as a kwarg (for cases when the user wants to retain the output gds file at a specific directory)
- Changes the default behavior to use a context manager to clean up the temp directory after it is created
- Adds tests for the two different invocation types

## [5.3.7](https://github.com/gdsfactory/gdsfactory/pull/325)

- add ipywidgets for `read_from_yaml` netlist driven flow tutorial.

## [5.3.6](https://github.com/gdsfactory/gdsfactory/pull/324)

- update gf.read.from_dphox to the latest version

## [5.3.5](https://github.com/gdsfactory/gdsfactory/pull/323)

- [clean code](https://github.com/gdsfactory/gdsfactory/pull/321)
- if no optical ports found with add_fiber_array or add_fiber_array it will raise ValueError [inspired by issue](https://github.com/gdsfactory/gdsfactory/issues/322)

## [5.3.4](https://github.com/gdsfactory/gdsfactory/pull/320)

- fix tests

## [5.3.3](https://github.com/gdsfactory/gdsfactory/pull/319)

- [copy component info and settings if they exist](https://github.com/gdsfactory/gdsfactory/pull/316)
- clean code
- add https://sonarcloud.io code checker
- add https://sourcery.ai code checker
- drop support for python3.7 so we can use [named expressions only supported in python >= 3.8](https://docs.sourcery.ai/refactorings/use-named-expression/)

## [5.3.0](https://github.com/gdsfactory/gdsfactory/pull/312)

- fix some fstrings [issues](https://github.com/gdsfactory/gdsfactory/issues/311)
- fix lumerical notebook [typo](https://github.com/gdsfactory/gdsfactory/issues/309)
- enable Component.plot() with ports with orientation = None
- add gf.routing.get_route_from_steps_electrical
- rename ComponentFactory to ComponentSpec and ComponentOrFactory to ComponentSpec [PR](https://github.com/gdsfactory/gdsfactory/pull/313)
  - replace callable(component) with gf.get_component(component)
  - replace some call_if_func(component) with gf.get_component(component)

## [5.2.9](https://github.com/gdsfactory/gdsfactory/pull/308)

- route ports with orientation = None

## [5.2.8](https://github.com/gdsfactory/gdsfactory/pull/307)

- add more type annotations. To reduce the number of mypy errors.
- [PR](https://github.com/gdsfactory/gdsfactory/pull/306)

## [5.2.7](https://github.com/gdsfactory/gdsfactory/pull/305)

- fix [issue](https://github.com/gdsfactory/gdsfactory/issues/301)
- show how to customize text_with_grid [issue](https://github.com/gdsfactory/gdsfactory/issues/302)

## [5.2.6](https://github.com/gdsfactory/gdsfactory/pull/304)

- remove tempfile and tmpdir after Component.show() sends GDS to klayout. To avoid filling /tmp/ with GDS files

## [5.2.5](https://github.com/gdsfactory/gdsfactory/pull/303)

- add fail_on_duplicates=False to add_ports_from_labels

## [5.2.4](https://github.com/gdsfactory/gdsfactory/pull/299)

- allow ports to have None orientation. The idea is that DC ports don't care about orientation. This still requires some work.
- adapt route_sharp from phidl to gf.routing.route_sharp for electrical routes
- cross_section function width and offset parameters are consistent with CrossSection class

## 5.2.3

- add electrical routes to routing_strategy

## [5.2.2](https://github.com/gdsfactory/gdsfactory/pull/296)

- add `get_name_from_label` to `add_ports_from_labels`
- add optional `layer_label` to `add_ports_from_labels`
- remove `.` in clean_name, before it was renaming `.` to `p`

## [5.2.1](https://github.com/gdsfactory/gdsfactory/pull/289)

- [PR](https://github.com/gdsfactory/gdsfactory/pull/289)

  - rename cladding_offsets as bbox_offsets
  - copy_child_info propagates polarization and wavelength info

- make sure 0 or None is 0 in `xmin` or `xmax` keys from component_from_yaml

## [5.2.0](https://github.com/gdsfactory/gdsfactory/pull/287)

- rename `contact` to `via_stack`

## [5.1.2](https://github.com/gdsfactory/gdsfactory/pull/286)

- `Component.remove_layers` also removes layers from paths
- add `bbox_layers` and `bbox_offsets` to `taper`

## [5.1.1](https://github.com/gdsfactory/gdsfactory/pull/285)

- add `gf yaml webapp -d` or `gf yaml webapp --debug` for debug mode
- fix [get_netlist for component arrays issue](https://github.com/gdsfactory/gdsfactory/issues/263)

## [5.1.0](https://github.com/gdsfactory/gdsfactory/pull/284)

- improve shear angle algorithm to work with waveguides at any angle [PR](https://github.com/gdsfactory/gdsfactory/pull/283)

  - add examples in notebooks
  - add tests
  - add shear_angle attribute to Port
  - Update test_shear_face_path.py

- remove default port width, layer and midpoint

## [5.0.7](https://github.com/gdsfactory/gdsfactory/pull/281)

- define layermap as pydantic BaseModel
- Sometimes it is desirable to have a waveguide with a shear face (i.e. the port face is not orthogonal to the propagation direction, but slightly slanted). [PR](https://github.com/gdsfactory/gdsfactory/pull/280) adds the capability to extrude basic waveguides with shear faces.

## [5.0.6](https://github.com/gdsfactory/gdsfactory/pull/279)

- fix set active PDK on component_from_yaml

## [5.0.5](https://github.com/gdsfactory/gdsfactory/pull/278)

- implements `get_active_pdk()` and `set_active_pdk()` functions to avoid side-effects of using ACTIVE_PDK global variable in different scopes. Renames `ACTIVE_PDK` to `_ACTIVE_PDK` to make it private, and instead promotes `get_active_pdk()`
- fixes instances where cross_section was specified and/or used as a factory rather than CrossSectionSpec
- fixes cases where cross_section was directly called as a function rather than invoking get_cross_section(cross_section) pattern
- Section.layer type needs to be the Union of Layer and Tuple[Layer,Layer] as long as we use the current implementation of Transition
- when getting instances in read_yaml(), uses the dictionary ComponentSpec format to get each component rather than using component name and `**settings` the old method causes an error for container-style components which have an argument named component
- for CrossSection class, makes info non-optional and by default instantiates empty dictionary. also replaces default values for mutable types with factories creating empty mutable types
- for cross_section() function, removes unused args

## [5.0.4](https://github.com/gdsfactory/gdsfactory/pull/277)

- fix cross_section from get_route_from_steps
- replace CrossSectionFactory to CrossSectionSpec
- replace ComponentFactory to ComponentSpec

## [5.0.3](https://github.com/gdsfactory/gdsfactory/pull/276)

- fix mmi1x2 and 2x2 definition to use waveguide cross_sections

## [5.0.2](https://github.com/gdsfactory/gdsfactory/pull/275)

- get_cells and get_component_factories work with module and list of modules
- add `gf.get_cells` and `gf.get_cross_section_factories`
- get_component and get_cross_section accepts also omegaconf.DictConfig
- add pack_doe and pack_doe_grid to containers
- add gf.get_cell, and enable partials

## [5.0.1](https://github.com/gdsfactory/gdsfactory/pull/274)

- fix bends bbox

## [5.0.0](https://github.com/gdsfactory/gdsfactory/pull/273)

- refactor cross_section. I recommend reviewing the Layout Tutorial -> Paths and CrossSections
  - include routing parameters (width, layer)
  - rename ports to port_names
  - make it immutable and remove add method
  - raise Error when creating a foreign key
  - rename `ports` to `port_names`
- refactor Section
  - make it immutable
  - raise Error when creating a foreign key
- add gf.Pdk
  - add gf.get_component(component_spec) returns a Component from the active PDK using the registered Cells
  - add gf.get_cross_section(cross_section_spec) returns a CrossSection from the active PDK using the registered CrossSectionFactory
  - add Pdk.register_cells()
  - add Pdk.register_cross_sections()
- add gf.ACTIVE_PDK
- delete klayout autoplacer code. Use gf.read.from_yaml instead.
- delete YAML placer code. Use gf.read.from_yaml instead.

## [4.7.3](https://github.com/gdsfactory/gdsfactory/pull/272)

- add `has_routing_info` to [CrossSection](CrossSection) to ensure it has routing information
- rename cross_section_factory to cross_sections
- rename component_factory to cells
- add ComponentSpec, CrossSectionSpec, gf.get_component, gf.get_cross_section, gf.Pdk

## [4.7.2](https://github.com/gdsfactory/gdsfactory/pull/270)

- add vscode plugin to docs
- get_bundle accepts also cross_section as well as cross_section_factory
- rename gethash to text_lines
- simplify component_factory definition
- simplify cross_section_factory definition

## [4.7.1](https://github.com/gdsfactory/gdsfactory/pull/265)

- `gf yaml build` can read from stdin

## [4.7.0](https://github.com/gdsfactory/gdsfactory/pull/264)

- convert LayerStack from dict to BaseModel, which accepts a single layers: Dict[str, LayerLevel]
- add gf.get_factories to get_component_factories and get_module_factories
- add `gf yaml build filepath` where filepath is a YAML path that you want to show in klayout
- update to phidl 1.6.1

## [4.6.3](https://github.com/gdsfactory/gdsfactory/pull/262)

- pack_doe and pack_doe_grid have a function argument
- fix netlist.json schema for instances to have pack kwarg
- add `gf yaml watch` CLI command to watch a YAML file

## 4.6.2

- add Component.get_netlist_dict
- document gdsfactory to sax

## [4.6.1](https://github.com/gdsfactory/gdsfactory/pull/261)

- add xmin, xmax, ymin, ymax to JSON schema
- remove placer schema, as it's being deprecated in favor of JSON YAML schema

## [4.6.0](https://github.com/gdsfactory/gdsfactory/pull/260)

- add `pack_doe` and `pack_doe_grid` as part of YAML component definition.
- add deprecation warning on gf.placer and gf.autoplacer.
- add `get_module_factories` to get all Component factories from a module.
- add gf.read.from_yaml placer support for xmin, xmax, ymin, ymax
- simpler documentation (remove API, gf, YAML mask)
  - remove mask klayout YAML placer documentation, as it's being deprecated

## [4.5.4](https://github.com/gdsfactory/gdsfactory/pull/258)

- enable schema validation in `ic yaml ide`
- validate schema and fail with unknown keys

## [4.5.3](https://github.com/gdsfactory/gdsfactory/pull/257)

- icyaml does not validate schema
- routes = None by default in NetlistModel

## [4.5.2](https://github.com/gdsfactory/gdsfactory/pull/256)

- better cross_section parsing in YAML component [PR](https://github.com/gdsfactory/gdsfactory/pull/254)
- recursive netlist extraction [PR](https://github.com/gdsfactory/gdsfactory/pull/255)
- add Component.get_netlist_recursive()

## [4.5.1](https://github.com/gdsfactory/gdsfactory/pull/253)

- replace asserts by raise ValueError in read.from_yaml

## [4.5.0](https://github.com/gdsfactory/gdsfactory/pull/252)

- `gf yaml ide` brings up dashboard to build YAML based circuits.
- gf.read.from_yaml has `cache=False` by default.
- revert get_netlist to version 4.0.17, add option `full_settings=False` back.
- fix notebook examples for extruding cross_sections with variable width or offset. Increased default `npoints = 2` to `npoints = 40`

## 4.4.15

- fix add_pins_siepic order [PR](https://github.com/gdsfactory/gdsfactory/pull/248)

## 4.4.14

- add cross_section settings to cutback_bend [PR](https://github.com/gdsfactory/gdsfactory/pull/246)

## 4.4.13

- add [klayout SALT package](https://github.com/gdsfactory/gdsfactory/issues/240)

## 4.4.7

- add dx_start and dy_start to route ports to side [PR](https://github.com/gdsfactory/gdsfactory/pull/242/files) when using route_ports_to_side to route up and to the left/right, the minimum distance of the bottom route could not be less than the separation between routes. This adds options to override this behavior and use the larger of dx_start/dy_start and the radius instead.
- add suffix option to select ports [PR](https://github.com/gdsfactory/gdsfactory/pull/243)
- Interconnect improvements [PR](https://github.com/gdsfactory/gdsfactory/pull/241)
- fix gdsfactory meep interface, it works now with different layer stacks [PR](https://github.com/gdsfactory/gdsfactory/pull/244)

## [4.4.6](https://github.com/gdsfactory/gdsfactory/pull/239)

- fix klive macro to maintain position and do not reload layers. Make sure you run `gf tool install` to update your macro after you update to the latest gdsfactory version.

## [4.4.5](https://github.com/gdsfactory/gdsfactory/pull/238)

- remove absorb from coupler ring and coupler90
- [update interconnect plugin](https://github.com/gdsfactory/gdsfactory/pull/237)
- [add siepic labels to components](https://github.com/gdsfactory/gdsfactory/pull/234)

## [4.4.4](https://github.com/gdsfactory/gdsfactory/pull/236)

- snap_to_grid straight waveguide length to reduce 1nm DRC snapping errors

## [4.4.3](https://github.com/gdsfactory/gdsfactory/pull/235)

- document mask metadata merging

## [4.4.2](https://github.com/gdsfactory/gdsfactory/pull/231)

- Component.absorb keeps paths from absorbed reference
- add port_name to ring_single_dut

## [4.4.0](https://github.com/gdsfactory/gdsfactory/pull/227)

- change siepic pin_length from 100nm to 10nm
- absorb maintains labels
- rename add_pins to decorator in cross_section function and class
- add add_pins_siepic_optical and add_pins_siepic_electrical
- add PORTE: Layer = (1, 11)
- remove add_pins_to_references and add_pins_container

## [4.3.10](https://github.com/gdsfactory/gdsfactory/pull/225)

- add package data in setup.py
- remove bend_radius from mzit

## 4.3.8

- move load_lyp_generic to try Except

## [4.3.7](https://github.com/gdsfactory/gdsfactory/pull/222)

- add_pin_path now works with siepic
- add add_pins_siepic in gf.add_pins
- gf.path.extrude can also add pins
- unpin `requirements.txt` [issue](https://github.com/gdsfactory/gdsfactory/issues/221)

## [4.3.6](https://github.com/gdsfactory/gdsfactory/pull/217)

- add_pin_path fixes

## [4.3.5](https://github.com/gdsfactory/gdsfactory/pull/216)

- rename add_pin_square to add_pin_rectangle
- add_pin_path to gf.add_pins

## [4.3.4](https://github.com/gdsfactory/gdsfactory/pull/215)

- tidy3d improvements:
  - get_simulation and write_sparameters accepts componentOrFactory
  - grating_coupler simulations can also be dispersive

## [4.3.3](https://github.com/gdsfactory/gdsfactory/pull/214)

- tidy3d improvements:
  - add dispersive flag in tidy3d get_simulation
  - write_sparameters_batch can accept list of kwargs
  - write_sparameters accepts with_all_monitors: if True, includes field monitors which increase results file size.
  - add test_write_sparameters
  - run tidy3d tests on every push as part of test_plugins CI/CD

## [4.3.1](https://github.com/gdsfactory/gdsfactory/pull/213)

- gf.components.grating_coupler_circular improvements:
  - rename teeth_list by a simpler widths and gaps separate arguments
  - delete grating_coupler_circular_arbitrary as it's now unnecessary
  - add bias_gap
- gf.components.grating_coupler_elliptical improvements:
  - add bias_gap
- fix [serialization of ports](https://github.com/gdsfactory/gdsfactory/pull/212)
- extend_ports works with cross_sections that do not have layer
- `pip install gdsfactory` also installs most of the plugins
  - `pip install gdsfactory[full]` only adds SIPANN (which depends on ternsorflow, which is a heavy dependency)

## 4.3.0

- tidy3d improvements:
  - update to version 1.1.1
- change port angle type annotation from int to float

## [4.2.17](https://github.com/gdsfactory/gdsfactory/pull/210)

- tidy3d improvements:
  - change tidy3d grating_coupler angle positive to be positive for the most normal case (grating coupler waveguide facing west)
  - tidy3d plot simulations in 2D only shows one plot
- add cross_section to grating_coupler waveguide ports

## [4.2.16](https://github.com/gdsfactory/gdsfactory/pull/209)

- grating_coupler_circular does not auto_rename_ports
- simulation.tidy3d.write_sparameters_batch accepts kwargs for general simulations settings
- add simulation.tidy3d.utils print_tasks
- increase grating_coupler simulation wavelengths from 1.2 to 1.8um

## 4.2.15

- add sklearn notebook on fitting dispersive coupler model
- add sklearn to requirements_full

## 4.2.14

- add with_all_monitors=False by default to avoid storing all fields when running gtidy3d.write_sparameters_grating_coupler

## 4.2.13

- fix `is_3d=False` case 2D sims for tidy3d write_sparameters_grating

## 4.2.13

- gmeep simulation improvements:

  - ymargin=3 by default
  - add write_sparameters_meep_1x1 for reciprocal devices (port_symmetries1x1)
  - add write_sparameters_meep_1x1_bend90 for 90degree bend simulations

- fix `is_3d=False` case to run simulations in 2D with [tidy3d](https://github.com/flexcompute/tidy3d/issues/229)

## 4.2.12

- update tidy3d client to latest version 1.0.2
- add `is_3d` to run simulations in 2D

## 4.2.11

- tidy3d simulation plugin improvements
  - add tidy3d.get_simulation_grating_coupler

## 4.2.10

- tidy3d simulation plugin improvements
  - add run_time_ps to tidy3d plugin, increase by 10x previous default run_time_ps
  - if a task was deleted it raises WebError exception, catch that in get results

## [4.2.9](https://github.com/gdsfactory/gdsfactory/pull/199)

- thread each tidy3d.write_sparameters simulation, so they run in paralell
- add tidy3d.write_sparameters_batch to run multiple sparameters simulations in paralell

## [4.2.8](https://github.com/gdsfactory/gdsfactory/pull/198)

- fix tidy3d materials. Si3N4 uses Luke2015 by default

## [4.2.7](https://github.com/gdsfactory/gdsfactory/pull/197)

- fix meep grating_coupler (draw teeth instead of etch)
- add triangle2 and triangle4 to components
- tidy3d.plot_simulation_xy accepts wavelength to plot permitivity
- tidy3d.get_simulation accepts wavelength_min, wavelength_max, wavelength_steps
- tidy3d.get_simulation accepts wavelength_min, wavelength_max, wavelength_steps
- tidy3d.get_sparameters returns Sparameters dataframe wavelength_min, wavelength_max, wavelength_steps
- rename meep.write_sparameters wl_min to wavelength_start, wl_max to wavelength_stop and wl_steps to wavelength_points
- add port_source_offset to tidy3d.get_simulation as a workaround for [tidy3d issue](https://github.com/gdsfactory/gdsfactory/issues/191)

## [4.2.6](https://github.com/gdsfactory/gdsfactory/pull/196)

- rename gen_loopback() function to add_loopback in gdsfactory.add_loopback

## 4.2.5

- add gf.simulation.gmeep.write_sparameters_grating

## 4.2.4

- tidy3d plugin improvements

## [4.2.3](https://github.com/gdsfactory/gdsfactory/pull/190)

- better notebook doc
- update tidy3d plugin to latest version 1.0.1

## 4.2.2

- add gf.components.delay_snake_sbend
- rename gf.simulation.sax.from_csv to read
- rename gf.simulation.sax.models.coupler to coupler_single_wavelength
- add more models to sax: grating_coupler, coupler (dispersive)

## 4.2.1

- center gdsfactory.simulation.modes at z=0
- rename dirpath to cache for gdsfactory.simulation.modes
- change sidewall_angle from radians to degrees

## 4.2.0

- add gdsfactory.simulation.simphony circuit simulation plugin
- fix gdsfactory.modes.overlap test

## 4.1.5

- add gdsfactory.simulation.sax circuit simulation plugin

## 4.1.4

- improve gdsfactory/samples tutorial
- make klive python2 compatible

## 4.1.3

- fix netlist tests

## 4.1.2

- fix netlist export

## 4.1.0

- difftest copy run_file to ref_file if prompt = Y (before it was just deleting it)
- Component.info is just now a regular dict (no more DictConfig)
- move Component.info.{changed, full, default} to Component.settings
- Component.metadata is a DictConfig property
- serialize with numpy arrays with orjson
- add Component.metadata and Component.metadata_child
- reduce total test time from 50 to 25 seconds thanks to faster serialization

## 4.0.18

- improve gdsfactory.simulation.modes
  - replace dataclass with pydantic.BaseModel
  - add pickle based file cache to speed up mode calculation
  - find_modes_waveguide and find_modes_coupler do not need to pass mode_solver
  - add single_waveguide kwarg to find_modes_waveguide and find_modes_coupler

## 4.0.17

- pass layer_stack to read_sparameters_lumerical, so that it reads the same file as write_sparameters_lumerical

## 4.0.14

- add delete_fsp_files kwarg to write_sparameters_lumerical

## 4.0.13

- rename write_sparameters_meep_mpi_pool to write_sparameters_meep_batch
- redirect write_sparameters_meep_mpi stderr and stdout to logger
- if stderr write_sparameters_meep_mpi does not wait for the results
- add gf.simulation.modes.find_modes_coupler

## 4.0.12

- improve tidy3d plugin
  - add xmargin_left, xmargin_right, ymargin_bot, ymargin_top
  - plot_simulation_xy and plot_simulation_yz
  - fix materials
  - add tests

## 4.0.8

- Explicit port serialization [PR](https://github.com/gdsfactory/gdsfactory/pull/178)
- difftest should fail when there is no regression reference [PR](https://github.com/gdsfactory/gdsfactory/pull/177)
- add Sparameters calculation in tidy3d plugin

## 4.0.7

- add progress bar to write_sparameters_lumerical_components

## 4.0.4

- modify the write_gds() function to fix the checking of duplicate cell names (recursively), and it also gives an option to choose how to handle duplicate cell names on write. It changes the default behavior to warn and overwrite duplicates, rather than throw an error. [PR](https://github.com/gdsfactory/gdsfactory/pull/174)
- remove clear_cache in `show()`. Intermediate clearing of cache can cause errors in final gds export, by leaving two versions of the same cell lingering within subcells created before/after cache clearing.
- remove clear_cache in some of the tests

## 4.0.3

- add `safe_cell_names` flag to gf.read.import_gds, append hash to imported cell names to avoid duplicated cell names.

## 4.0.2

- move triangle into requirements_dev.txt. Now that there is wheels for python3.9 and 3.10 you can manage the dependency with pip.

## 4.0.1

- [Mode field profile interpolation + overlap integrals](https://github.com/gdsfactory/gdsfactory/pull/170)
- [Properly serialize transitions](https://github.com/gdsfactory/gdsfactory/pull/171)

## 4.0.0

- Consider only changed component args and kwargs when calculating hash for component name
- meep plugin write_sparameters_meep_mpi deletes old file when overwrite=True
- ensure write_sparameters_meep `**kwargs` have valid simulation settings
- fix component lattice mutability
- Component.auto_rename_ports() raises MutabilityError if component is locked
- add `Component.is_unlocked()` that raises MutabilityError
- rename component_lattice `components` to `symbol_to_component`
- raise error when trying to add two ports with the same name in `gf.add_ports.add_ports_from_markers_center`. Before it was just ignoring ports if it already had a port with the same name, so it was hard to debug.
- difftest adds failed test to logger.error, to clearly see test_errors and to log test error traces
- clean_value calls clean_value_json, so we only need to maintain one function to serialize both settings and name

## 3.12.9

- fix tests

## 3.12.8

- rename `padding_north`, `padding_west`, `padding_east`, `padding_south` -> `ymargin_top`, `xmargin_left`, `xmargin_right`, `ymargin_bot` for consistency of the meep plugin with the Lumerical plugin.
- add `write_sparameters_meep_lr` with left and right ports and `write_sparameters_meep_mpi_lt` with left and top ports
- add xmargin and ymargin to write_sparameters_meep

## 3.12.7

- add Optional nslab to gm.modes.get_mode_solver_rib
- add `padding_north`, `padding_west`, `padding_east`, `padding_south`
- add tqdm progress bar to meep sims

## 3.12.6

- make trimesh an optional dependency by moving imports inside function

## 3.12.3

- fix docker container gdsfactory:latest
- leverage meep plot flag to avoid initializing the structure
- recommend to install triangle with mamba, and the rest of the dependencies with pip

## 3.12.1

- rename gdsfactory.components.array to gdsfactory.components.array_component
- create `.gitpod.yml`

## 3.12.0

- Consider only passed component args and kwargs when calculating hash for component name
- replace `_clean_value` by `clean_value_json`
- delete `tech.Library` as it's not being used. You can just use a dict of functions instead

## 3.11.5

- move rectpack import inside pack function
- create `pip install[dev]` just for developers, and reduce the dependencies for `pip install[full]`
- recommend installing gdspy and meep with mamba (faster than conda)
- rename w1 as width1 and w2 as width2 in find_neff_vs_width

## 3.11.4

- Remove numpy.typing from snap.py to be compatible with minimum version of numpy

## 3.11.3

- rename `res` to `resolution` in simulation.modes to be consistent with simulation.gmeep

## 3.11.2

- add plugins to notebooks and coverage

## 3.11.0

- get_sparameters_path filepath based on component_name + simulation_settings hash
- move gdsfactory.simulation.write_sparameters_lumerical to gdsfactory.simulation.lumerical.write_sparameters_lumerical
- Sparameters are all lowercase (both for meep and lumerical plugins)

## 3.10.12

- write_sparameters_lumerical allows passing material refractive index or any material in Lumerical's material database

## 3.10.11

- improve docs

## 3.10.10

- cell name with no parameters passed only includes prefix [PR](https://github.com/gdsfactory/gdsfactory/pull/158)
- write_sparameters_meep can exploit symmetries [PR](https://github.com/gdsfactory/gdsfactory/pull/157)

## 3.10.9

- add tests for `write_sparameters_meep_mpi` and `write_sparameters_meep_mpi_pool` in `gdsfactory.simulation.gmeep` module
- `write_sparameters_meep_mpi` has `wait_to_finish` flag

## 3.10.8

- improve meep simulation interface documentation and functions
- expose new `write_sparameters_meep_mpi` and `write_sparameters_meep_mpi_pool` in `gdsfactory.simulation.gmeep` module
- `get_sparameters_path` can also accept a layer_stack

## 3.10.7

- fix crossing hard coded layers. Add cross_section setting to ports so that they can be extended.
- extend_ports creates cross_section with port_width and layer, if port has no cross_section and extend_ports does not have a specific cross_section

## 3.10.6

- add mzi_pads_center to components

## 3.10.5

- fix add_ports_from_markers_center port location for square ports, depending on inside parameter

## 3.10.4

- use matplotlib for default plotter instead of holoviews
- add_ports default prefix is 'o' for optical and 'e' for electrical ports

## 3.10.3

- [plot Sparameters uses lowercase s11, s21 ...](https://github.com/gdsfactory/gdsfactory/pull/146)

## 3.10.2

- write_cells in gf.write_cells uses gdspy interface directly
- gf.import_gds has an optional gdsdir argument
- remove unused max_name_length parameter in gf.import_gds
- bring back matplotlib as the default plotter backend. Holoviews does not work well with some `sphinx.autodoc` docs
- add_fiber_array prints warning if grating coupler port is not facing west

## 3.10.1

- You can set up the default plotter from the gdsfactory config `gf.CONF.plotter = 'matplotlib'`
- [PR 142](https://github.com/gdsfactory/gdsfactory/pull/142)
  - dispersive flag to meep simulations
  - fixed bug where adding a layer would throw an error if "visible" or "transparent" were undefined in the .lyp file
- remove p_start (starting period) from grating_coupler_elliptical

## 3.10.0

- add Component.ploth() to plot with holoviews (inspired by dphox)
- Component.plot(plotter='holoviews') accepts plotter argument for plotting backend (matplotlib, qt or holoviews)
- use holoviews as the default plotting backend
- remove clear_cache from Component.plot() and Component.show(), it's easier to just do `gf.clear_cache()`
- remove `Component.plotqt` as the qt plotter is now available with `Component.plot(plotter='qt')`
- gf.geometry.boolean works with tuples of components or references as well as single component or Reference. Overcome phidl bug, where tuples are not trated as lists.
- Before plotting make sure we recompute the bounding box
- YAML mask definition allows using `settings` for global variables
- grating_coupler_rectangular first teeth starts next to the taper

## 3.9.28

- seal_ring accepts bbox instead of component
- die_bbox_frame accepts bbox
- die_bbox: rename text_position to text_anchor
- die_bbox: text_anchor accepts Literal instead of string

## 3.9.27

- Add [sidewall angles in MPB](https://github.com/gdsfactory/gdsfactory/pull/136)

## 3.9.26

- add some extra kwargs (with_taper1, with_taper2) to straight_heater_doped_rib
- add slab offset kwargs to cross_section.rib_heater_doped_contact

## 3.9.25

- `gf.components.contact_slot` accepts optional layer_offsetsx and layer_offsetsy
- extend_ports cross_section is optional, and overrides port cross_section

## 3.9.23

- validate cross_section
- update requirements
- add acks in README

## 3.9.22

- add `gf.read.from_dphox`
- update requirements.txt

## 3.9.21

- thanks to @thomasdorch [PR](https://github.com/gdsfactory/gdsfactory/pull/128) you can now use Meep's material database in your mode and FDTD simulations

## 3.9.20

- add `loopback_xspacing` to `gf.routing.add_fiber_single`

## 3.9.19

- add `Component.get_setting()` which looks inside info, settings.full and child_info
- add `gf.function.add_settings_label` decorator

## 3.9.18

- rename get_sparametersNxN to write_sparameters_meep, to be consistent with write_sparameters_lumerical function name

## 3.9.17

- meep interface stores simulation metadata

## 3.9.16

- meep interface improvements
  - add run=True flag, if run=False, plots simulation
- docker includes mpi version of meep

## 3.9.15

- meep interface improvements
  - add test Sparameters file dataframe
- lumerical interface improvements (consitent with meep)
  - wavelengths in um
  - Sparameters starts with lowercase

## 3.9.14

- fix seal_ring snap_to_grid to 2nm
- add Component.version and Component.changelog
- Component.to_dict() exports version
- add ring_single_heater and ring_double_heater to components
- add port_inclusion to pad and compass

## 3.9.13

- fix seal_ring snap_to_grid to 2nm

## 3.9.12

- fix text_rectangular_multi_layer layers

## 3.9.11

- add label_prefix to test and measurement labels

## 3.9.10

- add `gf.mask.merge_yaml` to merge yaml metadata
- rename `pcm_optical` to `cdsem_all`
- add `cdsem_coupler`
- Component.copy hash cache=True flag that adds new copies to CACHE (similarly to import_gds) to avoid duplicated cells

## 3.9.9

- pack_row in klayout_yaml_placer also accepts rotation
- placer uses Literal ('N', 'S', 'E', 'W') from gf.types
- rename label_layer as layer_label for consistency

## 3.9.8

- better DRC messages
- write_drc allows you to define the shortcut
- fix resistance_sheet offset
- add comments to build does flow

## 3.9.7

- build docker container
- recommend building triangle with conda forge instead of pip for conda based distributions
- add `pip install gdsfactory[pip]` as a pip-based alternative of `pip install gdsfactory[full]`

## 3.9.6

- Component.show() writes component in a different tempfile everytime. This avoids the `reload` question prompt from klayout.
- update klive to 0.0.7 to keep the same layers active between sessions

## 3.9.5

- imported cell names get incremented (starting on index = 1) with a `$` (based on Klayout naming convention)
- add test for flatten = True
- raise ValueError if the passed name is already on any CAHE (CACHE_IMPORTED or CACHE)
- avoid duplicate cells decorating import_gds with functools.lru_cache
- show accepts `**kwargs` for write_gds
- simplify decorator in @cell (does not change name)

## 3.9.4

- imported cell names get incremented (starting on index = 0) as we find them also in the CACHE. This avoids duplicated cell names.

## 3.9.3

- better error messages using f"{component!r}" to get `'component_name'`
- import*gds avoids duplicated cells by checking CACHE_IMPORTED and adding and underscore `*` suffix in case there are some name conflicts.
- add `Component.lock()` and `Component.unlock()` allows you to modify component after adding it into CACHE
- add `gf.geometry.check_duplicated_cells` to check duplicated cells. Thanks to Klayout
- fix `mzi_with_arms`, before it had `delta_length` in both arms

## 3.9.2

- increase `gf.routing.get_route_electrical` default min_straight_length from 10nm to 2um
- rename text_rectangular to text_rectangular_multi_layer
- rename manhattan_text to text_rectangular

## 3.9.1

- gf.import_gds updates info based on `kwargs`. In case you want to specify (wavelength, test_protocol...)
- store gf.import_gds name

## 3.9.0

- move add_ports_from_markers functions from `gf.import_gds` to `gf.add_ports`
- move write_cells functions from `gf.import_gds` to `gf.write_cells`
- move `gf.import_gds` to `gf.read.import_gds`. keep `gf.import_gds` as a link to `gf.read.import_gds`
- combine gf.read.from_gds with gf.import_gds
- add logger.info for write_gds, write_gds_with_metadata, gf.read.import_gds, klive.show()

## 3.8.15

- gf.read.from_gds passes kwargs to gf.import_gds
- rename grating_coupler_loss to grating_coupler_loss_fiber_array4 gf.components
- add grating_coupler_loss_fiber_single to components

## 3.8.14

- klayout is an optional dependency

## 3.8.13

- copy adds `_copy` suffix to minimize chances of having duplicated cell names

## 3.8.12

- add gf.functions.add_texts to add labels to a list of components or componentFactories

## 3.8.11

- gf.assert.version supports [semantic versioning](https://python-semanticversion.readthedocs.io/en/latest/)

## 3.8.10

- get_netlist works even with cells that have have no settings.full or info.changed (not properly decorated with cell decorator)

## 3.8.9

- pack and grid accepts tuples of text labels (text_offsets, text_anchors), in case we want multiple text labels per component
- add `gf.functions.add_text` to create a new component with a text label
- add rotate90, rotate90n and rotate180 to functions

## 3.8.8

- rename pack parameters (offset->text_offset, anchor->text_anchor, prefix->text_prefix)
- pack and grid can mirror references

## 3.8.7

- rotate accepts component or factory
- add plot_imbalance1x2 and plot_loss1x2 for component.simulation.plot
- rename bend_circular c.info.radius_min = float(radius) to c.info.radius = float(radius)

## 3.8.6

- add gf.grid_with_text

## 3.8.5

- fix rectangle_with_slits
- rename mzi2x2 as mzi2x2_2x2, so it's clearly different from mzi1x2_2x2

## 3.8.4

- straight_heater_doped has with_top_contact and with_bot_contact settings to remove some contacts
- rib_heater_doped and rib_heater_doped_contact has with_bot_heater and with_top_heater settings

## 3.8.3

- replace in contact_yspacing by heater_gap in straight_heater_doped

## 3.8.2

- add kwarg `auto_rename_ports=True` to `add_ports_from_markers_center`
- mzi length_x is optional and defaults to straight_x_bot/top defaults
- change mzi_phase_shifter straight_x = None, to match phase shifter footprint
- replace gf.components.mzi_phase_shifter_90_90 with gf.components.mzi_phase_shifter_top_heater_metal

## 3.8.1

- add `gf.components.mzi` as a more robust implementation for the MZI
- rename `gf.components.mzi` to `gf.components.mzi_arms`
- expose `toolz.compose` as `gf.compose`
- add `gf.components.mzi1x2`, `mzi1x2_2x2`, `mzi_coupler`

## 3.8.0

- add `gf.components.copy_layers` to duplicate a component in multiple layers.
- better error message for `gf.pack` when it fails to pack some Component.
- rename gf.simulation.gmpb as gf.simulation.modes
- rename gf.simulation.gtidy3d as gf.simulation.tidy3d
- gf.simulation.modes.find_neff_vs_width can store neffs in CSV file when passing `filepath`
- `gf.components.rectangle_with_slits` has now `layer_slit` parameter

## 3.7.8

- cell accepts `autoname` (True by default)
- import_gds defaults calls cell with `autoname=False`

## 3.7.7

- `write_gds` prints warning when writing GDS files with Unnamed cells. Unnamed cells don't get deterministic names. warning includes the number of unnamed cells
- cells with `decorator=function` that return a new cell do not leave Unnamed cells now
- pack includes a name_prefix to avoid unnamed cells
- add `taper_cross_section` into a container so we can use a decorator over it without triggering InmutabilityError

## 3.7.6

- to dict accepts component and function prefixes of the structures that we want to ignore when saving the settings dict
- `write_gds` prints warning when writing GDS files with Unnamed cells. Unnamed cells don't get deterministic names.

## 3.7.5

- add `add_tapers_cross_section` to taper component cross_sections
- letter `v` in text_rectangular_multi_layer is now DRC free

## 3.7.4

- add pad_gsg_short and pad_gsg_open to components
- export function parameters in settings exports as dict {'function': straight, 'width': 3}
  - works also for partial and composed functions
- add `get_child_name` for Component, so that when you run `copy_child_info` the name prefix also propagates
- only add layers_cladding for waveguide lengths > 0. Otherwise it creates non-orientable boundaries

## 3.7.3

- add `**kwargs` to `cutback_bend`
- pack type annotation is more general with `List[ComponentOrFactory]` instead of `List[Component]`, it also builds any Components if you pass the factory instead of the component.
- add `straight_length` parameter and more sensitive default values (2\*radius) to `cutback_component`
- add `gf.components.taper_parabolic`
- `mzi_lattice` adds all electrical ports from any of the mzi stages
- rename `mzi_factory` to `mzi` in mzi_lattice to be consistent with other component kwargs
- replace taper_factory with taper to be consistent with other component kwargs
- coupler snaps length to grid, instead of asserting length is on_grid
- add layers_cladding to rib so bezier_slabs render correctly for rib couplers

## 3.7.2

- add_fiber_array and add_fiber_single can also get a component that has no child_info

## 3.7.1

- keep python3.7 compatibility for `gf.functions.cache` decorator by using `cache = lru_cache(maxsize=None)` instead of `cache = lru_cache`
- `add_fiber_array` accepts ComponentOrFactory, convenient for testing the function without building a component

## 3.7.0

- fix clean_name
  - generators and iterables are properly hashed now
  - toolz.compose functions hash both the functions and first function
  - casting foats to ints when possible, so straight(length=5) and straight(length=5.0) return the same component
- set Component.\_cached = True when adding Component into cache, and raises MutabilityError when adding any element to it.
- Component.flatten() returns a copy of the component, that includes the flattened component. New name adds `_flat` suffix to original name
- add bias to grating_coupler_lumerical
- try to cast float to int when exporting info
- remove `ComponentSweep` as it was trivial to define as a list comprehension
- remove `add_text` as it is prone to creating mutability errors
- pack can now add text labels if passed a text ComponentFactory

## 3.6.8

- `add_fiber_single` allows to have multiple gratings
- converted add_fiber_single, component_sequence and add_fiber_array from `cell_without_validator` to `cell`
- Component pydantic validator accepts cell names below 100 characters (before it was forcing 32)

## 3.6.7

- rename doe, write_does and load_does to `sweep` module `read_sweep`, `write_sweep` ...
- Route and Routes are pydantic.BaseModel instead of dataclasses
- composed functions get a unique name. You can compose functions with `toolz.compose`
- add `gf.add_text` for adding text labels to a list of Components
- add `gf.types.ComponentSweep`
- increase MAX_NAME_LENGTH to 100 characters when validating a component
- add typing_extensions to requirements to keep 3.7 compatibility. Changed `from typing import Literal` (requires python>=3.8) to `from typing_extensions import Literal`
- add type checking error messages for Component and ComponentReference
- add type checking pydantic validator for Label
- replace `phidl.device_layout.Label` with `gf.Label`
- Route has an Optional list of Label, in case route fails, or in case you want to add connectivity labels

## 3.6.6

- add slab arguments (slab_layer, slab_xmin) to grating couplers
- remove align to bottom left in gdsdiff
- gdsdiff after asking question, re-rises GeometryDifferencesError

## 3.6.5

- fix gdsfactory/samples
- better docstrings documents `keyword Args` as well as `Args`
- refactor:
  - pads_shorted accepts pad as parameter
  - rename `n_devices` to columns in splitter_chain
  - rename `dbr2` to `dbr_tapered`
  - simpler pn cross_section definition

## 3.6.3

- args in partial functions was being ignore when creating the name. Only kwargs and func.**name** were being considered

## 3.6.2

- update rectpack dependency from 0.2.1 to 0.2.2

## 3.6.1

- spiral_external_io_fiber_single has a cross_section_ports setting
- seal_ring snaps to grid
- Component.bbox and ComponentReference.bbox properties snap to 1nm grid
- add `gf.components.bend_straight_bend`

## 3.6.0

- snap_to_grid_nm waypoints in round_corners to avoid 1nm gaps in some routes
- add `gf.components.text_rectangular_multi_layer`
- add `gf.components.rectangle_with_slits`

## 3.5.12

- add tolerance to netlist extraction. Snap to any nm grid for detecting connectivity (defaults to 1nm).

## 3.5.10

- enable having more than 2 ports per cross_section. Include test for that.

## 3.5.9

- better docstrings
- component_sequence also accepts component factories

## 3.5.9

- gf.simulation.get_sparameters_path takes kwargs with simulation_settings
- cross have port_type argument
- splitter_tree exposes bend_s info
- change simulation_settings default values
  - port_margin = 0.5 -> 1.5
  - port_extension = 2.0 -> 5.0
  - xmargin = 0.5 -> 3.0
  - ymargin = 2.0 -> 3.0
  - remove pml_width as it was redundant with xmargin and ymargin
- route with auto_taper was missing a mirror

## 3.5.8

- gf.components.extend_ports uses port.cross_section to extend the port

## 3.5.6

- add `cell` decorator to gf.components.text

## 3.5.5

- expose spacing parameter in `gf.routing.get_bundle_from_steps`

## 3.5.3

- make trimesh, and tidy3d optional dependencies that you can install with `pip install gdsfactory[full]`

## 3.5.1

- add `gf.routing.get_bundle_from_steps`

## 3.5.0

- rename `end_straight` to `end_straight_length`
- rename `start_straight` to `start_straight_length`

## 3.4.9

- add pad_pitch to `resistance_sheet`
- enable multimode waveguide in straight_heater_meander
- add `grating_coupler_elliptical_arbitrary`
- add `grating_coupler_elliptical_lumerical` using lumerical parametrization
- rename `grating_coupler_elliptical2` to `grating_coupler_circular`. rename `layer_core` to `layer`, `layer_ridge` to `layer_slab` for a more consistent parametrization of other grating couplers.
- add Component.add_padding

## 3.4.8

- pad has vertical_dc port

## 3.4.6

- add `gf.functions.move_port_to_zero`
- `gf.routing.add_fiber_single` has new parameter `zero_port` that can move a port to (0, 0)
- add fixme/routing
- enable `gf.read.from_yaml` to read ports that are defined without referencing any reference

## 3.4.5

- decorate `gf.path.extrude` with cell, to avoid duplicated cell names
- enforce contact_startLayer_endLayer naming convention
- gf.grid accepts rotation for reference
- add pydantic validator class methods to Path and CrossSection
- CrossSection has a `to_dict()`
- rename Component `to_dict` to `to_dict()`: is now a method instead of a property
- rename Component `pprint` to `pprint()`: is now a method instead of a property
- rename Component `pprint_ports` to `pprint_ports()`: is now a method instead of a property
- Component.mirror() returns a container

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

- simplify contact_with_offset_m1_m2
- contact_with_offset_m1_m2 use array of references
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
- fix cdsem_all

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
- `contact_with_offset_m1_m2` is now define with via functions instead of StrOrDict, skip it from tests
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

- rename tlm to contact and tlm_with_offset to contact_with_offset_m1_m2

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
- `add_padding` returns the same component, `add_padding` returns a container with component inside
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
- ListConfig iterates as a list in \clean_value_json
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
