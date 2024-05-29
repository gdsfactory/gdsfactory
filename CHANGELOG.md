# [Changelog](https://keepachangelog.com/en/1.0.0/)
## [Unreleased](https://github.com/gdsfactory/gdsfactory/compare/v7.27.0...main)

<!-- towncrier release notes start -->

## [7.27.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.27.1) - 2024-05-29

No significant changes.


## [7.27.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.27.0) - 2024-05-20

- Improve get route astar [#2752](https://github.com/gdsfactory/gdsfactory/pull/2752)
- add mimcap [#2751](https://github.com/gdsfactory/gdsfactory/pull/2751)
- fix disk with wrap_angle=0 [#2745](https://github.com/gdsfactory/gdsfactory/pull/2745)
- fix spiral_inner_io add_grating_couplers [#2742](https://github.com/gdsfactory/gdsfactory/pull/2742)
- Clean schematic [#2748](https://github.com/gdsfactory/gdsfactory/pull/2748)
- test other pdks [#2746](https://github.com/gdsfactory/gdsfactory/pull/2746)
- update ruff [#2743](https://github.com/gdsfactory/gdsfactory/pull/2743)
- adds cross-section kwarg [#2741](https://github.com/gdsfactory/gdsfactory/pull/2741)
- auto_rename_ports for C [#2739](https://github.com/gdsfactory/gdsfactory/pull/2739)
- get_route_sbend allow mismatch [#2737](https://github.com/gdsfactory/gdsfactory/pull/2737)
- Improve port types [#2732](https://github.com/gdsfactory/gdsfactory/pull/2732)
- Update kfactory[git,ipy] requirement from <0.13,>=0.9.1 to >=0.9.1,<0.14 [#2735](https://github.com/gdsfactory/gdsfactory/pull/2735)
- Fixed issue #2740 where cross-section kwarg was ignored for generic pdk grating coupler array. [#2741](https://github.com/gdsfactory/gdsfactory/issues/2741)


## [7.26.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.26.1) - 2024-05-09

- derived component has ports [#2731](https://github.com/gdsfactory/gdsfactory/pull/2731)

## [7.26.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.26.0) - 2024-05-09

- Improve layer stack [#2726](https://github.com/gdsfactory/gdsfactory/pull/2726)
- Add to svg [#2723](https://github.com/gdsfactory/gdsfactory/pull/2723)
- CrossSection Factory [#2730](https://github.com/gdsfactory/gdsfactory/pull/2730)
- gf.path.extrude() Inset Fix [#2729](https://github.com/gdsfactory/gdsfactory/pull/2729)
- Improve layer stack [#2726](https://github.com/gdsfactory/gdsfactory/pull/2726)
- routing/get_bundle_sbend: Get input port dynamically [#2725](https://github.com/gdsfactory/gdsfactory/pull/2725)
- removed deprecated pdk elements [#2727](https://github.com/gdsfactory/gdsfactory/pull/2727)
- more_robust_updk_import [#2722](https://github.com/gdsfactory/gdsfactory/pull/2722)
- improve_updk cell decorator [#2717](https://github.com/gdsfactory/gdsfactory/pull/2717)

## [7.25.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.25.2) - 2024-05-05

- improve_via_stack45 [#2716](https://github.com/gdsfactory/gdsfactory/pull/2716)
- improve straight_heater_metal [#2715](https://github.com/gdsfactory/gdsfactory/pull/2715)
- improve get_cross_sections [#2714](https://github.com/gdsfactory/gdsfactory/pull/2714)
- improve updk port layer [#2713](https://github.com/gdsfactory/gdsfactory/pull/2713)


## [7.25.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.25.1) - 2024-04-30

- Expose gc fiber port [#2709](https://github.com/gdsfactory/gdsfactory/pull/2709)
- bugfix in manhattan routing if the first reference is a bend turn [#2704](https://github.com/gdsfactory/gdsfactory/pull/2704)
- Add sidewall angle to Etch process step [#2703](https://github.com/gdsfactory/gdsfactory/pull/2703)
- better docstrings [#2707](https://github.com/gdsfactory/gdsfactory/pull/2707)

## [7.25.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.25.0) - 2024-04-25

- Forbid width mismatch [#2701](https://github.com/gdsfactory/gdsfactory/pull/2701)
- Rename fiber ports [#2700](https://github.com/gdsfactory/gdsfactory/pull/2700)
- fix stl export [#2702](https://github.com/gdsfactory/gdsfactory/pull/2702)
- add via_stack to spiral_heater [#2698](https://github.com/gdsfactory/gdsfactory/pull/2698)

## Dependency Updates

- Bump sphinx from 7.2.6 to 7.3.7 [#2693](https://github.com/gdsfactory/gdsfactory/pull/2693)
- Update kweb requirement from <1.3,>=1.1.9 to >=1.1.9,<2.1 [#2675](https://github.com/gdsfactory/gdsfactory/pull/2675)
- Update trimesh requirement from <4.3,>=4 to >=4,<4.4 [#2674](https://github.com/gdsfactory/gdsfactory/pull/2674)
- Update pydantic requirement from <2.7,>=2 to >=2,<2.8 [#2676](https://github.com/gdsfactory/gdsfactory/pull/2676)


## [7.24.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.24.0) - 2024-04-22

- add component_with_function [#2683](https://github.com/gdsfactory/gdsfactory/pull/2683)
- add via_yoffset to via_chain [#2678](https://github.com/gdsfactory/gdsfactory/pull/2678)
- import_gds unique names appends uuid [#2696](https://github.com/gdsfactory/gdsfactory/pull/2696)
- Fix cell annotation [#2692](https://github.com/gdsfactory/gdsfactory/pull/2692)
- fix docs [#2681](https://github.com/gdsfactory/gdsfactory/pull/2681)
- fix cross_section for ring_single_bend_coupler [#2668](https://github.com/gdsfactory/gdsfactory/pull/2668)
- Pdk improvements [#2694](https://github.com/gdsfactory/gdsfactory/pull/2694)
- Improve component with function [#2687](https://github.com/gdsfactory/gdsfactory/pull/2687)
- Update type annotations in Component [#2686](https://github.com/gdsfactory/gdsfactory/pull/2686)
- remove fill_rectangle from docs [#2685](https://github.com/gdsfactory/gdsfactory/pull/2685)
- better import gds and ports [#2680](https://github.com/gdsfactory/gdsfactory/pull/2680)
- improve updk [#2679](https://github.com/gdsfactory/gdsfactory/pull/2679)
- add via_yoffset to via_chain [#2678](https://github.com/gdsfactory/gdsfactory/pull/2678)
- c.info requires native types [#2691](https://github.com/gdsfactory/gdsfactory/pull/2691)
- Fix type annotation for mirror_x [#2670](https://github.com/gdsfactory/gdsfactory/pull/2670)
- Bump sphinx from 7.2.6 to 7.3.7 [#2693](https://github.com/gdsfactory/gdsfactory/pull/2693)
- Update kweb requirement from <1.3,>=1.1.9 to >=1.1.9,<2.1 [#2675](https://github.com/gdsfactory/gdsfactory/pull/2675)
- Update trimesh requirement from <4.3,>=4 to >=4,<4.4 [#2674](https://github.com/gdsfactory/gdsfactory/pull/2674)
- Update pydantic requirement from <2.7,>=2 to >=2,<2.8 [#2676](https://github.com/gdsfactory/gdsfactory/pull/2676)


## [7.23.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.23.0) - 2024-04-10

- add flag to remap layers [#2662](https://github.com/gdsfactory/gdsfactory/pull/2662)
- fix text justify [#2666](https://github.com/gdsfactory/gdsfactory/pull/2666)
- Fix cell decorator [#2665](https://github.com/gdsfactory/gdsfactory/pull/2665)
- Bugfix: make `Component.remap_layers` actually return a copy [#2658](https://github.com/gdsfactory/gdsfactory/pull/2658)
- allow custom import gds [#2661](https://github.com/gdsfactory/gdsfactory/pull/2661)
- release-drafter workflow [#2659](https://github.com/gdsfactory/gdsfactory/pull/2659)
- [minor bugs] Port.info: shallow copy -> deep copy [#2657](https://github.com/gdsfactory/gdsfactory/pull/2657)
- Obey stacklevel in `warnings.warn` for loguru output [#2656](https://github.com/gdsfactory/gdsfactory/pull/2656)

**Full Changelog**: https://github.com/gdsfactory/gdsfactory/compare/v7.22.3...v7.23.0


## [7.22.3](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.22.3) - 2024-04-04

- fix array ports [#2655](https://github.com/gdsfactory/gdsfactory/pull/2655)
- fix dc assymm [#2644](https://github.com/gdsfactory/gdsfactory/pull/2644)
- [pre-commit.ci] pre-commit autoupdate [#2652](https://github.com/gdsfactory/gdsfactory/pull/2652)
- get_bundle_sbend for vertical ports [#2649](https://github.com/gdsfactory/gdsfactory/pull/2649)
- use partial instead of gf.partial [#2646](https://github.com/gdsfactory/gdsfactory/pull/2646)
- snap ports to grid [#2654](https://github.com/gdsfactory/gdsfactory/pull/2654)
- Move auto-widen settings to args for get_route_from_steps [#2653](https://github.com/gdsfactory/gdsfactory/pull/2653)
- bugfix: assert_on_grid failing for large values [#2650](https://github.com/gdsfactory/gdsfactory/pull/2650)
- fix to the output of the from_updk function [#2645](https://github.com/gdsfactory/gdsfactory/pull/2645)


## [7.22.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.22.2) - 2024-03-24

- fix docs for straight_heater [#2642](https://github.com/gdsfactory/gdsfactory/pull/2642)
- Small improvements [#2641](https://github.com/gdsfactory/gdsfactory/pull/2641)

## [7.22.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.22.1) - 2024-03-23

- cross_section can be defined as a dict derived from a cross_section [#2639](https://github.com/gdsfactory/gdsfactory/pull/2639)
- Better schematic docs [#2640](https://github.com/gdsfactory/gdsfactory/pull/2640)
- add gf.components.array(centered=False/True)

## [7.22.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.22.0) - 2024-03-23

- add get_cross_section_name from pdk [#2638](https://github.com/gdsfactory/gdsfactory/pull/2638)
- Add schematic [#2622](https://github.com/gdsfactory/gdsfactory/pull/2622)
- more processes [#2634](https://github.com/gdsfactory/gdsfactory/pull/2634)
- Use bend xsize rather than radius for bend size [#2633](https://github.com/gdsfactory/gdsfactory/pull/2633)
- clean polygon vertices after converting to numpy array [#2629](https://github.com/gdsfactory/gdsfactory/pull/2629)
- fix ring_pn [#2631](https://github.com/gdsfactory/gdsfactory/pull/2631)
- Treat array as a cell_with_child and copy child info [#2625](https://github.com/gdsfactory/gdsfactory/pull/2625)
- Update trimesh requirement from <4.2,>=4 to >=4,<4.3 [#2630](https://github.com/gdsfactory/gdsfactory/pull/2630)

## [7.21.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.21.0) - 2024-03-11

- inject gf.Component return annotation when using gf.cell [#2620](https://github.com/gdsfactory/gdsfactory/pull/2620)


## [7.20.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.20.0) - 2024-03-10

- Add models to pdk [#2614](https://github.com/gdsfactory/gdsfactory/pull/2614)
- Simpler cross section [#2613](https://github.com/gdsfactory/gdsfactory/pull/2613)
- add post_process to cell decorator [#2610](https://github.com/gdsfactory/gdsfactory/pull/2610)
- fix read_from_yaml with x=instance,port [#2617](https://github.com/gdsfactory/gdsfactory/pull/2617)
- Fix electrical corner port location [#2615](https://github.com/gdsfactory/gdsfactory/pull/2615)
- deprecate gf.fill [#2616](https://github.com/gdsfactory/gdsfactory/pull/2616)


## [7.19.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.19.0) - 2024-03-07

- support activating more than 2 pdks [#2608](https://github.com/gdsfactory/gdsfactory/pull/2608)
- preview layerset [#2606](https://github.com/gdsfactory/gdsfactory/pull/2606)
- add default hatch_pattern [#2602](https://github.com/gdsfactory/gdsfactory/pull/2602)
- remove start_straight_length from cross_section [#2603](https://github.com/gdsfactory/gdsfactory/pull/2603)
- make gdsfactory less verbose [#2599](https://github.com/gdsfactory/gdsfactory/pull/2599)
- bugfix for rectangle instantiation [#2605](https://github.com/gdsfactory/gdsfactory/pull/2605)
- Update args for from_image to match from_np [#2595](https://github.com/gdsfactory/gdsfactory/pull/2595)


## [7.18.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.18.0) - 2024-03-06

- Accept callable or list [#2598](https://github.com/gdsfactory/gdsfactory/pull/2598)
- add required layers [#2597](https://github.com/gdsfactory/gdsfactory/pull/2597)
- allow post process list [#2594](https://github.com/gdsfactory/gdsfactory/pull/2594)
- Propagate `warnings.warn` category to loguru output [#2592](https://github.com/gdsfactory/gdsfactory/pull/2592)
- Add more pics [#2596](https://github.com/gdsfactory/gdsfactory/pull/2596)
- add required layers [#2597](https://github.com/gdsfactory/gdsfactory/pull/2597)
- Update args for from_image to match from_np [#2595](https://github.com/gdsfactory/gdsfactory/pull/2595)
- Add arg for constant pad value in read.from_np [#2582](https://github.com/gdsfactory/gdsfactory/pull/2582)

## [7.17.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.17.0) - 2024-03-05

- Add info to pcells [#2585](https://github.com/gdsfactory/gdsfactory/pull/2585)
- prevent validating schematic subdictionary of info [#2583](https://github.com/gdsfactory/gdsfactory/pull/2583)
- improve vias [#2584](https://github.com/gdsfactory/gdsfactory/pull/2584)
- Adding length_x and lengths_y arguments in ring_crow() component [#2586](https://github.com/gdsfactory/gdsfactory/pull/2586)
- Update kfactory[git,ipy] requirement from <0.12,>=0.9.1 to >=0.9.1,<0.13 [#2580](https://github.com/gdsfactory/gdsfactory/pull/2580)

## [7.16.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.16.0) - 2024-03-03

- Better grating coupler port names [#2578](https://github.com/gdsfactory/gdsfactory/pull/2578)
- Fix transition port size and MMIs with slab [#2579](https://github.com/gdsfactory/gdsfactory/pull/2579)

## [7.15.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.15.2) - 2024-03-03

- fix bend_direction ports [#2577](https://github.com/gdsfactory/gdsfactory/pull/2577)
- fix taper [#2575](https://github.com/gdsfactory/gdsfactory/pull/2575)

## [7.15.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.15.1) - 2024-03-02

- Improve docs [#2574](https://github.com/gdsfactory/gdsfactory/pull/2574)

## [7.15.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.15.0) - 2024-03-01

- Improve pdk import [#2573](https://github.com/gdsfactory/gdsfactory/pull/2573)
- Allow for different cross sections in cdsem [#2570](https://github.com/gdsfactory/gdsfactory/pull/2570)
- better pad assignments in add_fiber_array_optical_south_electrical_north [#2566](https://github.com/gdsfactory/gdsfactory/pull/2566)
- Update edge_coupler_array.py [#2568](https://github.com/gdsfactory/gdsfactory/pull/2568)

## [7.14.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.14.0) - 2024-02-27

- add via_chain [#2559](https://github.com/gdsfactory/gdsfactory/pull/2559)
- fix taper extrude [#2565](https://github.com/gdsfactory/gdsfactory/pull/2565)
- fix taper extrude [#2565](https://github.com/gdsfactory/gdsfactory/pull/2565)
- Handling top and bottom GC routing [#2560](https://github.com/gdsfactory/gdsfactory/pull/2560)
- Section ports [#2558](https://github.com/gdsfactory/gdsfactory/pull/2558)

## [7.13.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.13.0) - 2024-02-24

- Do not snap to grid by default.
- Snap lengths to 2dbu [#2550](https://github.com/gdsfactory/gdsfactory/pull/2550)
- Do not remove 1nm straights [#2554](https://github.com/gdsfactory/gdsfactory/pull/2554)
- round widths to 2nm when extruding [#2553](https://github.com/gdsfactory/gdsfactory/pull/2553)
- Make CrossSection hashable [#2552](https://github.com/gdsfactory/gdsfactory/pull/2552)
- remove test_offset [#2551](https://github.com/gdsfactory/gdsfactory/pull/2551)
- Snap lengths to 2dbu [#2550](https://github.com/gdsfactory/gdsfactory/pull/2550)
- Add context manager methods to Pdk [#2555](https://github.com/gdsfactory/gdsfactory/pull/2555)

## [7.12.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.12.0) - 2024-02-20

New

- add more post_process [#2546](https://github.com/gdsfactory/gdsfactory/pull/2546)
- add to_kfactory method [#2536](https://github.com/gdsfactory/gdsfactory/pull/2536)
- Add possibility of adding label when adding fiber array [#2535](https://github.com/gdsfactory/gdsfactory/pull/2535)
- add docs and development packages to docker container [#2533](https://github.com/gdsfactory/gdsfactory/pull/2533)
- add post_process to bezier [#2532](https://github.com/gdsfactory/gdsfactory/pull/2532)
- add post process to more pcells [#2524](https://github.com/gdsfactory/gdsfactory/pull/2524)

Bug Fixes

- fix typos [#2547](https://github.com/gdsfactory/gdsfactory/pull/2547)
- fixes to kfactory interface as well as show_ports with None orientation [#2544](https://github.com/gdsfactory/gdsfactory/pull/2544)
- fix extrude [#2529](https://github.com/gdsfactory/gdsfactory/pull/2529)
- fix seal_ring spacing [#2528](https://github.com/gdsfactory/gdsfactory/pull/2528)
- fix gdsfactory full [#2525](https://github.com/gdsfactory/gdsfactory/pull/2525)

Maintenance

- fix typos [#2547](https://github.com/gdsfactory/gdsfactory/pull/2547)
- [pre-commit.ci] pre-commit autoupdate [#2545](https://github.com/gdsfactory/gdsfactory/pull/2545)
- add more post_proces [#2546](https://github.com/gdsfactory/gdsfactory/pull/2546)
- fixes to kfactory interface as well as show_ports with None orientation [#2544](https://github.com/gdsfactory/gdsfactory/pull/2544)
- better annotations [#2542](https://github.com/gdsfactory/gdsfactory/pull/2542)
- Possibility of specifying offset in add_fiber_array [#2543](https://github.com/gdsfactory/gdsfactory/pull/2543)
- Add possibility of adding label when adding fiber array [#2535](https://github.com/gdsfactory/gdsfactory/pull/2535)
- add docs and development packages to docker container [#2533](https://github.com/gdsfactory/gdsfactory/pull/2533)
- add post_process to bezier [#2532](https://github.com/gdsfactory/gdsfactory/pull/2532)
- post_process [#2531](https://github.com/gdsfactory/gdsfactory/pull/2531)
- more post_process [#2530](https://github.com/gdsfactory/gdsfactory/pull/2530)
- fix extrude [#2529](https://github.com/gdsfactory/gdsfactory/pull/2529)
- fix gdsfactory full [#2525](https://github.com/gdsfactory/gdsfactory/pull/2525)
- [pre-commit.ci] pre-commit autoupdate [#2523](https://github.com/gdsfactory/gdsfactory/pull/2523)

Documentation

- better sample reticle [#2534](https://github.com/gdsfactory/gdsfactory/pull/2534)
- fix extrude [#2529](https://github.com/gdsfactory/gdsfactory/pull/2529)
- Better label handling [#2541](https://github.com/gdsfactory/gdsfactory/pull/2541)
- Update watchdog requirement from <4 to <5 [#2522](https://github.com/gdsfactory/gdsfactory/pull/2522)


## [7.11.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.11.2) - 2024-02-10

- publish docker container in [github registry](https://github.com/orgs/gdsfactory/packages?repo_name=gdsfactory)

## [7.11.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.11.0) - 2024-02-10

- Centering vias in 45degree via stack [#2518](https://github.com/gdsfactory/gdsfactory/pull/2518)
- fix docker build [#2516](https://github.com/gdsfactory/gdsfactory/pull/2516)
- add autolabeler [#2515](https://github.com/gdsfactory/gdsfactory/pull/2515)

## [7.10.8](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.8) - 2024-02-08

- Centering vias in 45degree via stack [#2518](https://github.com/gdsfactory/gdsfactory/pull/2518)
- fix docker build [#2516](https://github.com/gdsfactory/gdsfactory/pull/2516)
- add autolabeler [#2515](https://github.com/gdsfactory/gdsfactory/pull/2515)

## [7.10.7](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.7) - 2024-02-06

- Update kfactory latest [#2512](https://github.com/gdsfactory/gdsfactory/pull/2512)
- add_fiber_array_electrical control pad assignment [#2499](https://github.com/gdsfactory/gdsfactory/pull/2499)
- [pre-commit.ci] pre-commit autoupdate [#2510](https://github.com/gdsfactory/gdsfactory/pull/2510)
- improve Component.simplify to keep layers and name [#2507](https://github.com/gdsfactory/gdsfactory/pull/2507)
- add Component.simplify [#2505](https://github.com/gdsfactory/gdsfactory/pull/2505)

Dependency Updates:

- Update pydantic requirement from <2.6,>=2 to >=2,<2.7 [#2508](https://github.com/gdsfactory/gdsfactory/pull/2508)


## [7.10.6](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.6) - 2024-02-02

- fix add_ports_from_markers_center [#2503](https://github.com/gdsfactory/gdsfactory/pull/2503)
- Add possibility of different edge couplers in ec_array [#2502](https://github.com/gdsfactory/gdsfactory/pull/2502)
- Add option to toggle KLayout's ruler when plotting [#2500](https://github.com/gdsfactory/gdsfactory/pull/2500)
- Add seal ring segmented [#2493](https://github.com/gdsfactory/gdsfactory/pull/2493)
- better warnings traceability [#2486](https://github.com/gdsfactory/gdsfactory/pull/2486)
- export metadata as json [#2487](https://github.com/gdsfactory/gdsfactory/pull/2487)
- Adding possibility of different bends in 2x2 cutback [#2485](https://github.com/gdsfactory/gdsfactory/pull/2485)
- fixes some of the samples [#2484](https://github.com/gdsfactory/gdsfactory/pull/2484)
- [pre-commit.ci] pre-commit autoupdate [#2482](https://github.com/gdsfactory/gdsfactory/pull/2482)
- allow disabling automatic showing of figure in quickplotter [#2483](https://github.com/gdsfactory/gdsfactory/pull/2483)
- Component reduce [#2480](https://github.com/gdsfactory/gdsfactory/pull/2480)
- fix docker tag [#2479](https://github.com/gdsfactory/gdsfactory/pull/2479)
- Better doping layers [#2477](https://github.com/gdsfactory/gdsfactory/pull/2477)

- Bump styfle/cancel-workflow-action from 0.12.0 to 0.12.1 [#2495](https://github.com/gdsfactory/gdsfactory/pull/2495)
- Bump codecov/codecov-action from 3 to 4 [#2496](https://github.com/gdsfactory/gdsfactory/pull/2496)
- Bump actions/cache from 3 to 4 [#2494](https://github.com/gdsfactory/gdsfactory/pull/2494)
- Bump actions/upload-artifact from 3 to 4 [#2426](https://github.com/gdsfactory/gdsfactory/pull/2426)


## [7.10.5](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.5) - 2024-01-18


- add get_netlist merge_info flag [#2476](https://github.com/gdsfactory/gdsfactory/pull/2476)


## [7.10.4](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.4) - 2024-01-16


- Avoid pack deprecation warning [#2475](https://github.com/gdsfactory/gdsfactory/pull/2475)
- [pre-commit.ci] pre-commit autoupdate [#2474](https://github.com/gdsfactory/gdsfactory/pull/2474)
- fixing error message [#2473](https://github.com/gdsfactory/gdsfactory/pull/2473)
- Add additional & optional font to existing text_rectangular(...) API [#2471](https://github.com/gdsfactory/gdsfactory/pull/2471)
- Add additional reasonable default lumerical mappings [#2469](https://github.com/gdsfactory/gdsfactory/pull/2469)
- fix with_sbend option in get_route_from_steps [#2468](https://github.com/gdsfactory/gdsfactory/pull/2468)
- Serialize output for variable width/offsets in Section and CrossSection [#2466](https://github.com/gdsfactory/gdsfactory/pull/2466)
- Add __contains__ method to Info [#2459](https://github.com/gdsfactory/gdsfactory/pull/2459)
- add gf.functions.add_marker_layer_container [#2458](https://github.com/gdsfactory/gdsfactory/pull/2458)
- Fix release drafter to properly use gdsfactory project labels [#2457](https://github.com/gdsfactory/gdsfactory/pull/2457)


## [7.10.3](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.3) - 2024-01-09

- remove test warnings [#2455](https://github.com/gdsfactory/gdsfactory/pull/2455)
- remove cad installation on CICD [#2454](https://github.com/gdsfactory/gdsfactory/pull/2454)
- better naming [#2434](https://github.com/gdsfactory/gdsfactory/pull/2434)
- Add helper function for changing nested partial parameters [#2450](https://github.com/gdsfactory/gdsfactory/pull/2450)
- expose ports_map [#2449](https://github.com/gdsfactory/gdsfactory/pull/2449)
- [pre-commit.ci] pre-commit autoupdate [#2448](https://github.com/gdsfactory/gdsfactory/pull/2448)
- fix pic model settings [#2447](https://github.com/gdsfactory/gdsfactory/pull/2447)
- improve updk [#2439](https://github.com/gdsfactory/gdsfactory/pull/2439)


## [7.10.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.2) - 2024-01-07

- Improve grid snapping docs, functions and names [#2446](https://github.com/gdsfactory/gdsfactory/pull/2446)
- fixed flatten_offgrid_references return [#2445](https://github.com/gdsfactory/gdsfactory/pull/2445)
- rename flatten_invalid_refs to flatten_offgrid_references [#2444](https://github.com/gdsfactory/gdsfactory/pull/2444)
- make sbend router errors obvious [#2443](https://github.com/gdsfactory/gdsfactory/pull/2443)
- Consolidate warnings [#2442](https://github.com/gdsfactory/gdsfactory/pull/2442)
- improve tutorial avoid showing not recommended functions [#2438](https://github.com/gdsfactory/gdsfactory/pull/2438)


## [7.10.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.1) - 2024-01-04

- replace multiple invalid transformations warnings by a single one [#2437](https://github.com/gdsfactory/gdsfactory/pull/2437)
- avoid lambda functions [#2436](https://github.com/gdsfactory/gdsfactory/pull/2436)
- Recommend pip instead of conda for installation [#2433](https://github.com/gdsfactory/gdsfactory/pull/2433)

## [7.10.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.10.0) - 2024-01-02

- add xsize, ysize and size to via_stack_slot and improve grid reference naming [#2431](https://github.com/gdsfactory/gdsfactory/pull/2431)
- more robust serialization [#2430](https://github.com/gdsfactory/gdsfactory/pull/2430)

## [7.9.4](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.9.4) - 2024-01-01

- Bump actions/setup-python from 4 to 5 [#2424](https://github.com/gdsfactory/gdsfactory/pull/2424)
- Bump actions/stale from 8 to 9 [#2425](https://github.com/gdsfactory/gdsfactory/pull/2425)
- Bump actions/upload-pages-artifact from 2 to 3 [#2428](https://github.com/gdsfactory/gdsfactory/pull/2428)
- Bump actions/deploy-pages from 2 to 4 [#2427](https://github.com/gdsfactory/gdsfactory/pull/2427)
- Improve component settings serialization [#2421](https://github.com/gdsfactory/gdsfactory/pull/2421)
- update install instructions [#2419](https://github.com/gdsfactory/gdsfactory/pull/2419)
- make main cross section name parametrizable [#2417](https://github.com/gdsfactory/gdsfactory/pull/2417)
- expose mirror_straight in spiral_external_io [#2416](https://github.com/gdsfactory/gdsfactory/pull/2416)
- 2414 underplotoverplot dataprep [#2415](https://github.com/gdsfactory/gdsfactory/pull/2415)
- [pre-commit.ci] pre-commit autoupdate [#2413](https://github.com/gdsfactory/gdsfactory/pull/2413)
- fix get_route_from_steps cross_section [#2412](https://github.com/gdsfactory/gdsfactory/pull/2412)
- convert notebooks from python to jupyter [#2411](https://github.com/gdsfactory/gdsfactory/pull/2411)
- run stale workflow once a month instead of daily [#2410](https://github.com/gdsfactory/gdsfactory/pull/2410)
- use cross_section factories instead of copy [#2409](https://github.com/gdsfactory/gdsfactory/pull/2409)
- better error messages for modifying CrossSections [#2408](https://github.com/gdsfactory/gdsfactory/pull/2408)
- [deprecate decorator](https://github.com/gdsfactory/gdsfactory/commit/9837dc14a9a16b5067516874ad0c3390abf1127c)


## [7.9.3](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.9.3) - 2023-12-22

- fix cicd docker flow [#2407](https://github.com/gdsfactory/gdsfactory/pull/2407)
- Raise warning on empty add marker layer instead of failing [#2406](https://github.com/gdsfactory/gdsfactory/pull/2406)
- More robust add ports [#2404](https://github.com/gdsfactory/gdsfactory/pull/2404)
- Fix 1nm gaps in manhattan routes due to snapping issues [#2403](https://github.com/gdsfactory/gdsfactory/pull/2403)
- allow pack components with same name [#2401](https://github.com/gdsfactory/gdsfactory/pull/2401)
- fix cutback_loss [#2399](https://github.com/gdsfactory/gdsfactory/pull/2399)
- fix taper from csv [#2398](https://github.com/gdsfactory/gdsfactory/pull/2398)
- Update kweb requirement from <1.2,>=1.1.9 to >=1.1.9,<1.3 [#2396](https://github.com/gdsfactory/gdsfactory/pull/2396)
- [pre-commit.ci] pre-commit autoupdate [#2397](https://github.com/gdsfactory/gdsfactory/pull/2397)
- simpler cicd [#2395](https://github.com/gdsfactory/gdsfactory/pull/2395)
- Validate anchors [#2391](https://github.com/gdsfactory/gdsfactory/pull/2391)
- Fix access to references in component_sequence [#2394](https://github.com/gdsfactory/gdsfactory/pull/2394)
- improve cicd [#2392](https://github.com/gdsfactory/gdsfactory/pull/2392)
- fix direction parameter bend_straight_bend [#2390](https://github.com/gdsfactory/gdsfactory/pull/2390)
- improve via_stack to support slots and deprecate via_stack_slot [#2388](https://github.com/gdsfactory/gdsfactory/pull/2388)
- Custom transition [#2387](https://github.com/gdsfactory/gdsfactory/pull/2387)
- use functions instead of strings for routing [#2385](https://github.com/gdsfactory/gdsfactory/pull/2385)
- fix issues in flat netlist export [#2386](https://github.com/gdsfactory/gdsfactory/pull/2386)
- No straight factory to metal heater meander [#2382](https://github.com/gdsfactory/gdsfactory/pull/2382)


## [7.9.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.9.2) - 2023-12-11

- Remove warnings [#2380](https://github.com/gdsfactory/gdsfactory/pull/2380)
- Support straight, bend & taper factories in heater meander [#2378](https://github.com/gdsfactory/gdsfactory/pull/2378)
- fix cell decorator [#2379](https://github.com/gdsfactory/gdsfactory/pull/2379)
- Select layers to extract in `add_marker_layer` [#2375](https://github.com/gdsfactory/gdsfactory/pull/2375)
- make via_stack layers optional [#2374](https://github.com/gdsfactory/gdsfactory/pull/2374)
- better error message for pack [#2373](https://github.com/gdsfactory/gdsfactory/pull/2373)
- Improve routing [#2372](https://github.com/gdsfactory/gdsfactory/pull/2372)
- [pre-commit.ci] pre-commit autoupdate [#2371](https://github.com/gdsfactory/gdsfactory/pull/2371)
- Update kfactory[git,ipy] requirement from <0.10,>=0.9.1 to >=0.9.1,<0.11 [#2370](https://github.com/gdsfactory/gdsfactory/pull/2370)

## [7.9.1](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.9.1) - 2023-12-09

- Pass cross_section kwargs through coupler_straight [#2369](https://github.com/gdsfactory/gdsfactory/pull/2369)
- Support `**kwargs` in `Path.extrude` [#2367](https://github.com/gdsfactory/gdsfactory/pull/2367)
- Improve type annotations for gf.cell [#2368](https://github.com/gdsfactory/gdsfactory/pull/2368)
- Actually support ComponentSpec in `add_marker_layer` [#2365](https://github.com/gdsfactory/gdsfactory/pull/2365)
- Fixes show_subports not showing top-level Component ports [#2363](https://github.com/gdsfactory/gdsfactory/pull/2363)
- clean up generic pdk cells and docs [#2362](https://github.com/gdsfactory/gdsfactory/pull/2362)
- Exclude layers [#2358](https://github.com/gdsfactory/gdsfactory/pull/2358)
- More flexible support for different connectors in `get_bundle_all_angle` [#2360](https://github.com/gdsfactory/gdsfactory/pull/2360)
- avoid zero length polygons [#2356](https://github.com/gdsfactory/gdsfactory/pull/2356)
- raise error on mutation [#2354](https://github.com/gdsfactory/gdsfactory/pull/2354)
- improve non manhattan router docs [#2353](https://github.com/gdsfactory/gdsfactory/pull/2353)
- better default settings and docs [#2352](https://github.com/gdsfactory/gdsfactory/pull/2352)
- better straight_heater_meander [#2351](https://github.com/gdsfactory/gdsfactory/pull/2351)
- add spiral_meander [#2350](https://github.com/gdsfactory/gdsfactory/pull/2350)
- add round_corners_east_west and round_corners_north_south to rectangle [#2349](https://github.com/gdsfactory/gdsfactory/pull/2349)

## [7.9.0](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.9.0) - 2023-12-02

- improve readme [#2348](https://github.com/gdsfactory/gdsfactory/pull/2348)
- use ruff instead of black [#2347](https://github.com/gdsfactory/gdsfactory/pull/2347)
- add angle to edge_coupler_array [#2346](https://github.com/gdsfactory/gdsfactory/pull/2346)
- preparing for minor release [#2345](https://github.com/gdsfactory/gdsfactory/pull/2345)
- improve docstrings [#2344](https://github.com/gdsfactory/gdsfactory/pull/2344)
- Support adding partialed cross_sections in taper cross section again [#2343](https://github.com/gdsfactory/gdsfactory/pull/2343)
- fix add_tapers_cross_section [#2342](https://github.com/gdsfactory/gdsfactory/pull/2342)
- Add info length taper cross section [#2341](https://github.com/gdsfactory/gdsfactory/pull/2341)
- fix add_pads [#2340](https://github.com/gdsfactory/gdsfactory/pull/2340)
- add radius_min [#2339](https://github.com/gdsfactory/gdsfactory/pull/2339)
- Fix difftest bugs [#2337](https://github.com/gdsfactory/gdsfactory/pull/2337)
- add functions for writing test_manifest [#2336](https://github.com/gdsfactory/gdsfactory/pull/2336)
- Use shutil.copytree to install KLayout tech on Windows [#2332](https://github.com/gdsfactory/gdsfactory/pull/2332)
- add_port can set the port info [#2333](https://github.com/gdsfactory/gdsfactory/pull/2333)
- Allow labels on different layers when `label_layer=None` in `add_labels` [#2331](https://github.com/gdsfactory/gdsfactory/pull/2331)
- Add pins in `taper_cross_section` [#2330](https://github.com/gdsfactory/gdsfactory/pull/2330)
- [pre-commit.ci] pre-commit autoupdate [#2322](https://github.com/gdsfactory/gdsfactory/pull/2322)
- Improve docstrings for `add_pins` [#2325](https://github.com/gdsfactory/gdsfactory/pull/2325)
- Update pydantic requirement from <2.5,>=2 to >=2,<2.6 [#2321](https://github.com/gdsfactory/gdsfactory/pull/2321)


## [7.8.18](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.18) - 2023-11-19

- add wire corner kwargs [#2320](https://github.com/gdsfactory/gdsfactory/pull/2320)
- enable sbend routes in get_route_from_steps [#2319](https://github.com/gdsfactory/gdsfactory/pull/2319)
- Allow connections w/o vias in `KLayoutTechnology` [#2317](https://github.com/gdsfactory/gdsfactory/pull/2317)
- snap detector to 2x grid [#2314](https://github.com/gdsfactory/gdsfactory/pull/2314)

## [7.8.17](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.17) - 2023-11-15

- remove_zero_length_tapers_and_pins [#2312](https://github.com/gdsfactory/gdsfactory/pull/2312)
- fix plot [#2310](https://github.com/gdsfactory/gdsfactory/pull/2310)
- [pre-commit.ci] pre-commit autoupdate [#2308](https://github.com/gdsfactory/gdsfactory/pull/2308)
- access ports by index [#2307](https://github.com/gdsfactory/gdsfactory/pull/2307)
- fix spiral inner io with kwargs in grating couplers [#2306](https://github.com/gdsfactory/gdsfactory/pull/2306)
- add info label to cell_settings [#2305](https://github.com/gdsfactory/gdsfactory/pull/2305)
- simpler_add_grating_couplers [#2304](https://github.com/gdsfactory/gdsfactory/pull/2304)
- remove unused kwargs [#2303](https://github.com/gdsfactory/gdsfactory/pull/2303)
- faster extract [#2302](https://github.com/gdsfactory/gdsfactory/pull/2302)
- simpler [#2300](https://github.com/gdsfactory/gdsfactory/pull/2300)
- Fix `inspect.signature` bug for partials with partialed decorators [#2299](https://github.com/gdsfactory/gdsfactory/pull/2299)
- Support giving straight spec in `get_route_from_steps` [#2298](https://github.com/gdsfactory/gdsfactory/pull/2298)
- Update `gf install` command for KLayout integration [#2295](https://github.com/gdsfactory/gdsfactory/pull/2295)

## [7.8.16](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.16) - 2023-11-08

- Bump kweb from 1.1.9 to 1.1.10 [#2277](https://github.com/gdsfactory/gdsfactory/pull/2277)
- Components can extrude transitions [#2293](https://github.com/gdsfactory/gdsfactory/pull/2293)
- Fix mutability issues [#2291](https://github.com/gdsfactory/gdsfactory/pull/2291)
- fix zero length polygons [#2290](https://github.com/gdsfactory/gdsfactory/pull/2290)
- Add labels along with add_marker_layer and write pin to port layer on None [#2286](https://github.com/gdsfactory/gdsfactory/pull/2286)
- simpler container [#2287](https://github.com/gdsfactory/gdsfactory/pull/2287)
- default_show_to_gds [#2283](https://github.com/gdsfactory/gdsfactory/pull/2283)
- [pre-commit.ci] pre-commit autoupdate [#2278](https://github.com/gdsfactory/gdsfactory/pull/2278)
- remove show ports side effects [#2282](https://github.com/gdsfactory/gdsfactory/pull/2282)
- fix remove_from_cache [#2280](https://github.com/gdsfactory/gdsfactory/pull/2280)
- fix rectangle_with_slits [#2279](https://github.com/gdsfactory/gdsfactory/pull/2279)
- add edX course training link [#2276](https://github.com/gdsfactory/gdsfactory/pull/2276)


## [7.8.15](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.15) - 2023-11-05

- Fix typing for `get_cross_sections` and add tests [#2246](https://github.com/gdsfactory/gdsfactory/pull/2246)
- fix extrude_transition [#2243](https://github.com/gdsfactory/gdsfactory/pull/2243)
- Write cells can take list [#2241](https://github.com/gdsfactory/gdsfactory/pull/2241)
- deep copy child [#2240](https://github.com/gdsfactory/gdsfactory/pull/2240)
- fix add_electrical_pads_top_dc [#2239](https://github.com/gdsfactory/gdsfactory/pull/2239)
- add json formatted labels [#2237](https://github.com/gdsfactory/gdsfactory/pull/2237)
- add best practices [#2236](https://github.com/gdsfactory/gdsfactory/pull/2236)
- Allow different ComponentSpecs in edge coupler array [#2235](https://github.com/gdsfactory/gdsfactory/pull/2235)
- Allow none cross sections [#2234](https://github.com/gdsfactory/gdsfactory/pull/2234)
- Add flake8-debugger checks to ruff [#2233](https://github.com/gdsfactory/gdsfactory/pull/2233)
- fix cross_section for paths [#2231](https://github.com/gdsfactory/gdsfactory/pull/2231)
- add validate [#2221](https://github.com/gdsfactory/gdsfactory/pull/2221)
- filter warnings [#2229](https://github.com/gdsfactory/gdsfactory/pull/2229)
- pick good length_x for mzi length_x=None [#2228](https://github.com/gdsfactory/gdsfactory/pull/2228)
- add get_child_name for cell that contain other cells [#2224](https://github.com/gdsfactory/gdsfactory/pull/2224)
- fix get_layer [#2223](https://github.com/gdsfactory/gdsfactory/pull/2223)
- update pre-commit [#2222](https://github.com/gdsfactory/gdsfactory/pull/2222)
- Cell improved [#2219](https://github.com/gdsfactory/gdsfactory/pull/2219)
- Rename layers [#2217](https://github.com/gdsfactory/gdsfactory/pull/2217)
- allow to access cross_section by names [#2214](https://github.com/gdsfactory/gdsfactory/pull/2214)


## [7.8.14](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.14) - 2023-10-30

- Write test manifest [#2247](https://github.com/gdsfactory/gdsfactory/pull/2247)
- Fix cache [#2248](https://github.com/gdsfactory/gdsfactory/pull/2248)

## [7.8.13](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.13) - 2023-10-30

- fix extrude_transition [#2243](https://github.com/gdsfactory/gdsfactory/pull/2243)
- Write cells can take list [#2241](https://github.com/gdsfactory/gdsfactory/pull/2241)
- fix add_electrical_pads_top_dc [#2239](https://github.com/gdsfactory/gdsfactory/pull/2239)


## [7.8.12](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.12) - 2023-10-28

- add json formatted labels [#2237](https://github.com/gdsfactory/gdsfactory/pull/2237)
- add best practices [#2236](https://github.com/gdsfactory/gdsfactory/pull/2236)
- Allow different ComponentSpecs in edge coupler array [#2235](https://github.com/gdsfactory/gdsfactory/pull/2235)
- Allow none cross sections [#2234](https://github.com/gdsfactory/gdsfactory/pull/2234)
- Add flake8-debugger checks to ruff [#2233](https://github.com/gdsfactory/gdsfactory/pull/2233)
- fix cross_section for paths [#2231](https://github.com/gdsfactory/gdsfactory/pull/2231)
- add validate [#2221](https://github.com/gdsfactory/gdsfactory/pull/2221)
- filter warnings [#2229](https://github.com/gdsfactory/gdsfactory/pull/2229)
- pick good length_x for mzi length_x=None [#2228](https://github.com/gdsfactory/gdsfactory/pull/2228)
- add get_child_name for cell that contain other cells [#2224](https://github.com/gdsfactory/gdsfactory/pull/2224)
- fix get_layer [#2223](https://github.com/gdsfactory/gdsfactory/pull/2223)
- update pre-commit [#2222](https://github.com/gdsfactory/gdsfactory/pull/2222)
- Cell improved [#2219](https://github.com/gdsfactory/gdsfactory/pull/2219)
- Rename layers [#2217](https://github.com/gdsfactory/gdsfactory/pull/2217)
- allow to access cross_section by names [#2214](https://github.com/gdsfactory/gdsfactory/pull/2214)


## [7.8.11](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.11) - 2023-10-24

- remove get_capacitance_path [#2213](https://github.com/gdsfactory/gdsfactory/pull/2213)
- add keep_ports flag to union [#2212](https://github.com/gdsfactory/gdsfactory/pull/2212)
- Serialize width [#2211](https://github.com/gdsfactory/gdsfactory/pull/2211)
- consistent cutback_bend [#2210](https://github.com/gdsfactory/gdsfactory/pull/2210)
- Small fix to routing logic [#2208](https://github.com/gdsfactory/gdsfactory/pull/2208)
- fixing gf watch for watching directories [#2207](https://github.com/gdsfactory/gdsfactory/pull/2207)
- port_orientation can be tuple in ring heaters [#2206](https://github.com/gdsfactory/gdsfactory/pull/2206)
- add cell_settings as a separate column for manifest [#2205](https://github.com/gdsfactory/gdsfactory/pull/2205)

## [7.8.10](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.10) - 2023-10-20

- Add straight function to ring dut [#2203](https://github.com/gdsfactory/gdsfactory/pull/2203)
- add optional function add_bbox to bend_euler and bend_circular [#2202](https://github.com/gdsfactory/gdsfactory/pull/2202)
- Cleaner rings [#2201](https://github.com/gdsfactory/gdsfactory/pull/2201)
- allow align to be multi-layer [#2200](https://github.com/gdsfactory/gdsfactory/pull/2200)
- Add bbox [#2198](https://github.com/gdsfactory/gdsfactory/pull/2198)
- Fix file watcher [#2195](https://github.com/gdsfactory/gdsfactory/pull/2195)
- Less general error catching in LayerViews [#2190](https://github.com/gdsfactory/gdsfactory/pull/2190)
- make optional packages optional [#2187](https://github.com/gdsfactory/gdsfactory/pull/2187)
- make klayout optional [#2185](https://github.com/gdsfactory/gdsfactory/pull/2185)
- fix area [#2183](https://github.com/gdsfactory/gdsfactory/pull/2183)

## [7.8.9](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.9) - 2023-10-14

- fix add_pads with defined electrical ports [#2182](https://github.com/gdsfactory/gdsfactory/pull/2182)
- Add fixme [#2181](https://github.com/gdsfactory/gdsfactory/pull/2181)
- Add connectivity information to  PDK and generate KLayoutTechnology [#2179](https://github.com/gdsfactory/gdsfactory/pull/2179)

## [7.8.8](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.8) - 2023-10-13

- Fix cs and transitions [#2158](https://github.com/gdsfactory/gdsfactory/pull/2158)
- add via_stack_heater_m2 [#2178](https://github.com/gdsfactory/gdsfactory/pull/2178)
- Fix lvs demo [#2175](https://github.com/gdsfactory/gdsfactory/pull/2175)
- warn_connect_with_width_layer_or_type_missmatch [#2176](https://github.com/gdsfactory/gdsfactory/pull/2176)
- more representative implants generic_process [#2177](https://github.com/gdsfactory/gdsfactory/pull/2177)
- remove enforce_port_ordering arg from get_bundle_from_waypoint params [#2174](https://github.com/gdsfactory/gdsfactory/pull/2174)
- fixes difftest to use XOR instead of NOT [#2172](https://github.com/gdsfactory/gdsfactory/pull/2172)
- Fix docs [#2170](https://github.com/gdsfactory/gdsfactory/pull/2170)
- Faster remap layers [#2168](https://github.com/gdsfactory/gdsfactory/pull/2168)
- Import gds names have longer limit [#2167](https://github.com/gdsfactory/gdsfactory/pull/2167)


## [7.8.7](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.7) - 2023-10-08

- fix difftest [#2164](https://github.com/gdsfactory/gdsfactory/pull/2164)
- Extrude can have off grid ports [#2163](https://github.com/gdsfactory/gdsfactory/pull/2163)
- update to latest kfactory [#2161](https://github.com/gdsfactory/gdsfactory/pull/2161)
- fix slab_layer arg to accept None [#2160](https://github.com/gdsfactory/gdsfactory/pull/2160)

## [7.8.6](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.6) - 2023-10-04

- Fix yaml cell naming [#2156](https://github.com/gdsfactory/gdsfactory/pull/2156)
- fix off-grid ports [#2155](https://github.com/gdsfactory/gdsfactory/pull/2155)

## [7.8.5](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.5) - 2023-10-03

- bbox does not extend beyond ports in bends [#2154](https://github.com/gdsfactory/gdsfactory/pull/2154)

## [7.8.4](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.4) - 2023-10-02

- Fix clear cache [#2153](https://github.com/gdsfactory/gdsfactory/pull/2153)
- Update pydantic requirement from <2.4,>=2 to >=2,<2.5 [#2151](https://github.com/gdsfactory/gdsfactory/pull/2151)

## [7.8.3](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.3) - 2023-10-01

• improve change template #2144 https://github.com/gdsfactory/gdsfactory/pull/#2144
• cross_section_fixes #2143 https://github.com/gdsfactory/gdsfactory/pull/#2143

## [7.8.2](https://github.com/gdsfactory/gdsfactory/releases/tag/v7.8.2) - 2023-09-30

- fixed pn cross_section functions [#2141](https://github.com/gdsfactory/gdsfactory/issues/2141)
* Fix pn cross sections (#2142)
* use tbump and towncrier (#2140)
* better rib_with_trenches (#2133)
* better ascii art (#2139)
* extrude transition (#2138)
* Add dbr kwargs (#2136)
* fix docs for non-manhattan router (#2135)
* Improve cdsem (#2132) @joamatab

## [7.8.1](https://github.com/gdsfactory/gdsfactory/compare/v7.8.0...v7.8.1)

- fix `route_info` [PR](https://github.com/gdsfactory/gdsfactory/pull/2130)

## [7.8.0](https://github.com/gdsfactory/gdsfactory/compare/v7.7.1...v7.8.0)

- fixes name issue https://github.com/gdsfactory/gdsfactory/issues/2089
- adds `CONF.enforce_ports_on_grid` flag that allows you to create offgrid ports fixes https://github.com/gdsfactory/gdsfactory/issues/2118
- simplify CrossSection so that it's serializable.
- simplify Transition, it does not inherit from CrossSection
- create separate `gf.path.extrude_transition`
- remove A-star router from docs as it's an experimental feature, not ready for use
- remove **kwargs from many components
- add warnings for off-grid-ports and non-manhattan ports (0, 90, 180, 270)
- add `CrossSection.validate_radius`
- pin pydantic min version <2.4

## [7.7.1](https://github.com/gdsfactory/gdsfactory/compare/v7.7.0...v7.7.1)

- make snap to grid backwards compatible with <= 7.6.1

## [7.7.0](https://github.com/gdsfactory/gdsfactory/compare/v7.6.2...v7.7.0)

- fix `spiral_heater`
- raise error when using a single layer
- assert ports on grid at the cell layer, and bounding box points on grid
- snap to grid acctepts `grid_factor`

## [7.6.1](https://github.com/gdsfactory/gdsfactory/compare/v7.6.0...v7.6.1)

- serialize `dict_keys`

## [7.6.0](https://github.com/gdsfactory/gdsfactory/compare/v7.5.0...v7.6.0)

- improve labels and documentation for test protocol
- fix `layer_stack` mutability by redefining `LayerStack.model_copy()`
- add `spiral_inner_io_fiber_array` and `add_grating_couplers_fiber_array` to components

## [7.5.0](https://github.com/gdsfactory/gdsfactory/compare/v7.4.6...v7.5.0)

- fix layerstack.filtered
- Change default `Component.get_ports()` depth to 0
- handle complex numbers in serialization
- protect against no kweb installed
- Fix pydantic serializer warnings

## [7.4.6](https://github.com/gdsfactory/gdsfactory/compare/v7.4.5...v7.4.6)

- add layermap model [PR](https://github.com/gdsfactory/gdsfactory/pull/2074)

## [7.4.4](https://github.com/gdsfactory/gdsfactory/compare/v7.4.3...v7.4.4)

- update docs
- fix cli, remove uvicorn optional dependency from cli

## [7.4.3](https://github.com/gdsfactory/gdsfactory/compare/v7.4.1...v7.4.3)

- add schema version for YAML files [PR](https://github.com/gdsfactory/gdsfactory/pull/2050)
- add `gf.components.cutback_loss`
- improve `gf.components.die` and `gf.components.wafer`

## [7.4.1](https://github.com/gdsfactory/gdsfactory/compare/v7.4.0...v7.4.1)

- fix route with sbends

## [7.4.0](https://github.com/gdsfactory/gdsfactory/compare/v7.3.4...v7.4.0)

- port to pydantic2
- fix windows paths

## [7.3.4](https://github.com/gdsfactory/gdsfactory/compare/v7.3.3...v7.3.4)

- import `add_tapers`
- add name counter with `$1`

## [7.3.3](https://github.com/gdsfactory/gdsfactory/compare/v7.3.2...v7.3.3)

- clean path characters
- add bundle topology validation to `get_bundle`

## [7.3.2](https://github.com/gdsfactory/gdsfactory/compare/v7.3.1...v7.3.2)

- warn on bad bundles [PR](https://github.com/gdsfactory/gdsfactory/pull/1993)

## [7.3.1](https://github.com/gdsfactory/gdsfactory/compare/v7.3.0...v7.3.1)

- minor fixes

## [7.3.0](https://github.com/gdsfactory/gdsfactory/compare/v7.2.1...v7.3.0)

- make `flatten_offgrid_references=False` in GdsWriteSettings.
- add component flatten invalid refs.

## [7.2.1](https://github.com/gdsfactory/gdsfactory/compare/v7.2.0...v7.2.1)

- add `from_image` to import GDS from image.
- add `gf.components.rectangles`

## [7.2.0](https://github.com/gdsfactory/gdsfactory/compare/v7.1.4...v7.2.0)

- fix justify text rectangular
- switch from click to typer for CLI. Requires reinstall gdsfactory.
- add process info to layerstack

## [7.1.4](https://github.com/gdsfactory/gdsfactory/compare/v7.1.3...v7.1.4)

- Improve cell decorator updk [PR](https://github.com/gdsfactory/gdsfactory/pull/1967)
- add ComponentAlongPath [PR](https://github.com/gdsfactory/gdsfactory/pull/1965)

## [7.1.2](https://github.com/gdsfactory/gdsfactory/compare/v7.1.1...v7.1.2)

- set the run directory for difftests [PR](https://github.com/gdsfactory/gdsfactory/pull/1960)
- fix text(justify) [PR](https://github.com/gdsfactory/gdsfactory/pull/1961)

## [7.1.1](https://github.com/gdsfactory/gdsfactory/compare/v7.1.0...v7.1.1)

- updk improve [PR](https://github.com/gdsfactory/gdsfactory/pull/1954)

## [7.1.0](https://github.com/gdsfactory/gdsfactory/compare/v7.0.1...v7.1.0)

- switch from matplotlib to klayout from default plotter [PR](https://github.com/gdsfactory/gdsfactory/pull/1953)
- use jinja2 as the default YAML parser [PR](https://github.com/gdsfactory/gdsfactory/pull/1952)

## [7.0.1](https://github.com/gdsfactory/gdsfactory/compare/v7.0.0...v7.0.0)

- fix package for conda [PR](https://github.com/gdsfactory/gdsfactory/pull/1947)
- improve `clean_value_json` [PR](https://github.com/gdsfactory/gdsfactory/pull/1945)

## [7.0.0](https://github.com/gdsfactory/gdsfactory/compare/v7.0.0...v6.115.0)

- move plugins and simulation to gplugins repo [PR](https://github.com/gdsfactory/gdsfactory/pull/1935)
- add path length analyzer [PR](https://github.com/gdsfactory/gdsfactory/pull/1935)
- only works for python>=3.10

Migration guidelines:

- Modernize types Optional[float] -> float | None
```
find . -name "*.py" -exec sed -ri 's/Optional\[(.*)\]/\1 | None/g' {} \;
```
- Replace gdsfactory[.]simulation. with gplugins.
```
grep -rl 'gdsfactory.simulation.' /path/to/your/files | xargs sed -i 's/gdsfactory.simulation./gplugins./g'
```

## 6.116.0
- Warning: You need python>=3.10 to get the latest version of gdsfactory.

## [6.115.0](https://github.com/gdsfactory/gdsfactory/compare/v6.114.1...v6.115.0)

- add mmi for nxm [PR](https://github.com/gdsfactory/gdsfactory/pull/1915)

## [6.114.1](https://github.com/gdsfactory/gdsfactory/compare/v6.114.0...v6.114.1)

- fix `get_hash` for windows [PR](https://github.com/gdsfactory/gdsfactory/pull/1898)
- component preprocessing before meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1891)

## [6.114.0](https://github.com/gdsfactory/gdsfactory/compare/v6.114.0...v6.113.0)

- add uPDK support [PR](https://github.com/gdsfactory/gdsfactory/pull/1862)
- fix font [PR](https://github.com/gdsfactory/gdsfactory/pull/1854)
- update meow and tidy3d

## [6.113.0](https://github.com/gdsfactory/gdsfactory/compare/v6.113.0...v6.112.0)

- improve webapp [PR](https://github.com/gdsfactory/gdsfactory/pull/1833)
- fix `via_stack` ports orientation [PR](https://github.com/gdsfactory/gdsfactory/pull/1844)
- update meow to 0.7.0

## [6.112.0](https://github.com/gdsfactory/gdsfactory/compare/v6.112.0...v6.111.0)

- add loopback to grating coupler array [PR](https://github.com/gdsfactory/gdsfactory/pull/1819)
- add offset to terminator [PR](https://github.com/gdsfactory/gdsfactory/pull/1822)
- fix test metadata [PR](https://github.com/gdsfactory/gdsfactory/pull/1824)
- add `coupler_ring_point` [PR](https://github.com/gdsfactory/gdsfactory/pull/1825)

## [6.111.0](https://github.com/gdsfactory/gdsfactory/compare/v6.111.0...v6.109.0)

- add rib-strip routing example notebook [PR](https://github.com/gdsfactory/gdsfactory/pull/1808)
- fix wire components with bbox [PR](https://github.com/gdsfactory/gdsfactory/pull/1808)

## [6.109.0](https://github.com/gdsfactory/gdsfactory/compare/v6.109.0...v6.108.1)

- add module to `function_name` [PR](https://github.com/gdsfactory/gdsfactory/pull/1808)
- minor updates to MEOW [PR](https://github.com/gdsfactory/gdsfactory/pull/1806)
- use meshwell for 3D meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1800)
- add component meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1807)
- fix plugins tests and remove module from function name [PR](https://github.com/gdsfactory/gdsfactory/pull/1813)

## [6.108.1](https://github.com/gdsfactory/gdsfactory/compare/v6.108.1...v6.108.0)

- add module info to cell decorator [PR](https://github.com/gdsfactory/gdsfactory/pull/1805)

## [6.108.0](https://github.com/gdsfactory/gdsfactory/compare/v6.108.0...v6.107.4)

- add handshake for klive [PR](https://github.com/gdsfactory/gdsfactory/pull/1796)
- make `plot_kweb` the default [PR](https://github.com/gdsfactory/gdsfactory/pull/1802)
- update to latest MEOW [PR](https://github.com/gdsfactory/gdsfactory/pull/1799)
- add `gf.components.rectangular_ring` [PR](https://github.com/gdsfactory/gdsfactory/pull/1803)
- fix `get_bundle` regressions [PR](https://github.com/gdsfactory/gdsfactory/pull/1801)

## [6.107.4](https://github.com/gdsfactory/gdsfactory/compare/v6.107.4...v6.107.3)

- fix SDL notebook [PR](https://github.com/gdsfactory/gdsfactory/pull/1788)
- improve difftest [PR](https://github.com/gdsfactory/gdsfactory/pull/1787)

## [6.107.3](https://github.com/gdsfactory/gdsfactory/compare/v6.107.3...v6.107.2)

- pin meow [PR](https://github.com/gdsfactory/gdsfactory/pull/1786)
- upgrade tidy3d to 2.2.3 [PR](https://github.com/gdsfactory/gdsfactory/pull/1784)

## [6.107.2](https://github.com/gdsfactory/gdsfactory/compare/v6.107.2...v6.107.1)

- fix extrude serialization [PR](https://github.com/gdsfactory/gdsfactory/pull/1781)

## [6.107.1](https://github.com/gdsfactory/gdsfactory/compare/v6.107.1...v6.107.0)

- fix transitions [PR](https://github.com/gdsfactory/gdsfactory/pull/1777)
- improve docs and fix tidy3d local cache [PR](https://github.com/gdsfactory/gdsfactory/pull/1776)

## [6.107.0](https://github.com/gdsfactory/gdsfactory/compare/v6.107.0...v6.106.0)

- check for component overlap [PR](https://github.com/gdsfactory/gdsfactory/pull/1730)
- add wire corner with sections [PR](https://github.com/gdsfactory/gdsfactory/pull/1766)
- update femwell to 0.1.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/1775)
- enable kweb on GitHub codespaces [PR](https://github.com/gdsfactory/gdsfactory/pull/1774)
- active PDK does not clear cache [PR](https://github.com/gdsfactory/gdsfactory/pull/1773)

## [6.106.0](https://github.com/gdsfactory/gdsfactory/compare/v6.106.0...v6.105.0)

- add `fraction_te` to tidy3d mode solver.

## [6.105.0](https://github.com/gdsfactory/gdsfactory/compare/v6.105.3...v6.103.3)

- fix installer.
- add die text location [PR](https://github.com/gdsfactory/gdsfactory/pull/1760)
- make validate layers more generic [PR](https://github.com/gdsfactory/gdsfactory/pull/1761)
- fix taper [PR](https://github.com/gdsfactory/gdsfactory/pull/1762)

## [6.103.3](https://github.com/gdsfactory/gdsfactory/compare/v6.103.3...v6.103.1)

- add python3.7 bare bones tests [PR](https://github.com/gdsfactory/gdsfactory/pull/1745)

## [6.103.1](https://github.com/gdsfactory/gdsfactory/compare/v6.103.1...v6.103.0)

- fix jupyter widget [PR](https://github.com/gdsfactory/gdsfactory/pull/1740)

## [6.103.0](https://github.com/gdsfactory/gdsfactory/compare/v6.103.0...v6.102.4)

- add `gf.Polygon` [PR](https://github.com/gdsfactory/gdsfactory/pull/1736)
    - add `gf.Polygon.to_shapely()`
    - add `gf.Polygon.from_shapely()`
    - add `gf.Polygon.snap()`
- add `gf.components.coupler_bent`
    - fix 1nm gaps coming from forcing snapping to grid
- improve widget [PR](https://github.com/gdsfactory/gdsfactory/pull/1738)

## [6.102.4](https://github.com/gdsfactory/gdsfactory/compare/v6.102.4...v6.102.3)

- fix loopback snapping [PR](https://github.com/gdsfactory/gdsfactory/pull/1729)

## [6.102.3](https://github.com/gdsfactory/gdsfactory/compare/v6.102.3...v6.102.1)

- fix heater meander issues [PR](https://github.com/gdsfactory/gdsfactory/pull/1727)

## [6.102.1](https://github.com/gdsfactory/gdsfactory/compare/v6.102.1...v6.102.0)

- improve script to extract ports [PR](https://github.com/gdsfactory/gdsfactory/pull/1725)

## [6.102.0](https://github.com/gdsfactory/gdsfactory/compare/v6.102.0...v6.101.1)

- fix snapping references [PR](https://github.com/gdsfactory/gdsfactory/pull/1719)
- re-enable all angle routing tests [PR](https://github.com/gdsfactory/gdsfactory/pull/1721)

## [6.101.1](https://github.com/gdsfactory/gdsfactory/compare/v6.101.1...v6.101.0)

- fix kfactory dependency [PR](https://github.com/gdsfactory/gdsfactory/pull/1714)

## [6.101.0](https://github.com/gdsfactory/gdsfactory/compare/v6.101.0...v6.100.0)

- fix git diff gds [PR](https://github.com/gdsfactory/gdsfactory/pull/1712)

## [6.100.0](https://github.com/gdsfactory/gdsfactory/compare/v6.100.0...v6.99.0)

- add `get_polygon_bbox` and `get_polygon_enclosure` that return a shapely polygon [PR](https://github.com/gdsfactory/gdsfactory/pull/1709)

## [6.99.0](https://github.com/gdsfactory/gdsfactory/compare/v6.99.0...v6.98.2)

- improve difftest [PR](https://github.com/gdsfactory/gdsfactory/pull/1703)
- fix devsim [PR](https://github.com/gdsfactory/gdsfactory/pull/1704)
- update tidy3d to 2.2.2

## [6.98.2](https://github.com/gdsfactory/gdsfactory/compare/v6.98.2...v6.98.1)

- only use section.insets if they are not (0, 0) [PR](https://github.com/gdsfactory/gdsfactory/commit/9bffa6d84f4fe427e74cd6193c3afdc731bb0deb)
- fix webapp [PR](https://github.com/gdsfactory/gdsfactory/pull/1701)

## [6.98.1](https://github.com/gdsfactory/gdsfactory/compare/v6.98.1...v6.98.0)

- update kfactory [PR](https://github.com/gdsfactory/gdsfactory/pull/1695)
- fix tidy3d 2D FDTD sims and update tidy3d [PR](https://github.com/gdsfactory/gdsfactory/pull/1697)

## [6.98.0](https://github.com/gdsfactory/gdsfactory/compare/v6.98.0...v6.97.1)

- add verification functions to check for component overlap [PR](https://github.com/gdsfactory/gdsfactory/pull/1689/)

## [6.97.1](https://github.com/gdsfactory/gdsfactory/compare/v6.97.0...v6.97.1)

- fix snapping issues [PR](https://github.com/gdsfactory/gdsfactory/pull/1685)
- rename `write_gerbers` to `Component.write_gerber` [PR](https://github.com/gdsfactory/gdsfactory/pull/1687)

## [6.97.0](https://github.com/gdsfactory/gdsfactory/compare/v6.96.0...v6.97.0)

- improve declarative cell
- control Sbend resolution [PR](https://github.com/gdsfactory/gdsfactory/commit/db5a802c8161e2ea6c5cc1350bfc87a439bbd4ea)
- repair symmetries for tidy3d [PR](https://github.com/gdsfactory/gdsfactory/pull/1683)

## [6.96.0](https://github.com/gdsfactory/gdsfactory/compare/v6.95.0...v6.96.0)

- flatten invalid refs

## 6.95.0

- add pins and `flatten_offgrid_references` [PR](https://github.com/gdsfactory/gdsfactory/pull/1674)

## 6.95.0

- add `Component.write_gds(with_metadata=True) flag and deprecate `Component.write_gds_with_metadata()` [PR](https://github.com/gdsfactory/gdsfactory/pull/1668)
- tidy3d.ModeSolver supports multiple wavelengths [PR](https://github.com/gdsfactory/gdsfactory/pull/1673)
- remove S3 bucket [PR](https://github.com/gdsfactory/gdsfactory/pull/1670)
- Added CellDecoratorSettings to give ability to change default behavior of cell_without_validator function [PR](https://github.com/gdsfactory/gdsfactory/pull/1659)

## 6.94.0

- define connectivity for klayout [PR](https://github.com/gdsfactory/gdsfactory/pull/1635)
- add `gf web` [PR](https://github.com/gdsfactory/gdsfactory/pull/1584)

## 6.93.0

- add shapely polygon support and examples [PR](https://github.com/gdsfactory/gdsfactory/pull/1634)
- mode solver update [PR](https://github.com/gdsfactory/gdsfactory/pull/1628)

## 6.92.0

- show labels in klayout [PR](https://github.com/gdsfactory/gdsfactory/pull/1622)

## 6.91.0

- replace with database s3 bucket [PR](https://github.com/gdsfactory/gdsfactory/pull/1617)

## 6.89.7

- gdslib replacement with database s3 bucket [PR](https://github.com/gdsfactory/gdsfactory/pull/1594)
- add `zmin_tolerance` [PR](https://github.com/gdsfactory/gdsfactory/pull/1596)

## 6.89.4

- fix `grating_coupler_loss` structures [PR](https://github.com/gdsfactory/gdsfactory/pull/1589)

## 6.89.1

- improve manhattanization [PR](https://github.com/gdsfactory/gdsfactory/pull/1582)

## 6.89.0

- add manhattan [PR](https://github.com/gdsfactory/gdsfactory/pull/1579)
- fixes `rib_with_trenches` [PR](https://github.com/gdsfactory/gdsfactory/pull/1581)

## 6.88.1

- remove triangle dependency for M1 compatibility [PR](https://github.com/gdsfactory/gdsfactory/pull/1567)

## 6.88.0

- better greek cross [PR](https://github.com/gdsfactory/gdsfactory/pull/1561) [PR](https://github.com/gdsfactory/gdsfactory/pull/1560)

## 6.86.0

- add optional parameter cladding simplify [PR](https://github.com/gdsfactory/gdsfactory/pull/1558)

## 6.85.0

- add section inset [PR](https://github.com/gdsfactory/gdsfactory/pull/1555)

## 6.84.0

- better_cutback component [PR](https://github.com/gdsfactory/gdsfactory/pull/1554)

## 6.83.0

- extrude transitions [PR](https://github.com/gdsfactory/gdsfactory/pull/1548)
- add PSR [PR](https://github.com/gdsfactory/gdsfactory/pull/1546)

## 6.82.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/1544)

- fix tiling stitching issues

## 6.81.1 [PR](https://github.com/gdsfactory/gdsfactory/pull/1543)

- improve mode converter

## 6.81.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/1541)

- bring back multithreaded simulations with batch. Thanks to verbose flag is possible now.
- update tidy3d to latest version


## 6.80.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/1539)

- add function that returns labels with `GratingName-ComponentName-PortName`
- make it default for `gf.routing.add_fiber_single` and `gf.routing.add_fiber_array`
- for disabling automated measurement labels you can define `layer_label=None`

## 6.79.0

- add klayout fill [PR](https://github.com/gdsfactory/gdsfactory/pull/1535)
- improve spiral [PR](https://github.com/gdsfactory/gdsfactory/pull/1537)
- add `add_optical_ports_arms` flag to MZI [PR](https://github.com/gdsfactory/gdsfactory/pull/1538)

## 6.77.0

- add wire corner45 [PR](https://github.com/gdsfactory/gdsfactory/pull/1529)
- improve detector [PR](https://github.com/gdsfactory/gdsfactory/pull/1523)

## 6.76.0

- add gerber file support [PR](https://github.com/gdsfactory/gdsfactory/pull/1521)
- bends bbox works now for -90 and -180 deg [PR](https://github.com/gdsfactory/gdsfactory/pull/1522)

## 6.75.0

- Layer views update [PR](https://github.com/gdsfactory/gdsfactory/pull/1518)
- add `Component.add_ref_container()` convenient method to add reference into a new Component (container), [PR](https://github.com/gdsfactory/gdsfactory/pull/1519)

## 6.74.0

- add flag to Remove original layer in maskprep [PR](https://github.com/gdsfactory/gdsfactory/pull/1516)
- Adding an additional parameter `ang_res` to control the angular resolution of each section. [PR](https://github.com/gdsfactory/gdsfactory/pull/1513)

## 6.73.2

- extend kwargs #1507 to all `cross_section` functions [PR](https://github.com/gdsfactory/gdsfactory/pull/1508)

## 6.73.0

- better `maskprep` [PR](https://github.com/gdsfactory/gdsfactory/pull/1500)
- add names to trench (for taper) [PR](https://github.com/gdsfactory/gdsfactory/pull/1501)

## 6.72.9

- better `via_stack_with_offset` [PR](https://github.com/gdsfactory/gdsfactory/pull/1499)

## 6.72.8

- gf.components.array has optional size

## 6.72.7

- Mirroring drop waveguide for asymmetric waveguides in section based ring [PR](https://github.com/gdsfactory/gdsfactory/pull/1498)

## 6.72.6

- fix mzi with new routing defaults [PR](https://github.com/gdsfactory/gdsfactory/pull/1496)
- better error messages when vias do not fit in `via_stacks`

## 6.72.5

- better routing defaults [PR](https://github.com/gdsfactory/gdsfactory/pull/1495)

## 6.72.4

- better `via_stack_with_offset` [PR](https://github.com/gdsfactory/gdsfactory/pull/1494)
- adding radius as info for the section based ring [PR](https://github.com/gdsfactory/gdsfactory/pull/1493)

## 6.72.1

- fixes to `gf.components.ring_section_based` [PR](https://github.com/gdsfactory/gdsfactory/pull/1489)

## 6.72.0

- add `gf.components.regular_polygon` [PR](https://github.com/gdsfactory/gdsfactory/pull/1487)
- add function to calculate FSR [PR](https://github.com/gdsfactory/gdsfactory/pull/1488)

## 6.71.0

- mzi does not rename electrical ports [PR](https://github.com/gdsfactory/gdsfactory/pull/1485)
- add klayout fill [PR](https://github.com/gdsfactory/gdsfactory/pull/1484)

## 6.70.0

- test cutbacks2x2 [PR](https://github.com/gdsfactory/gdsfactory/pull/1479)
- improve `stack_with_offset` [PR](https://github.com/gdsfactory/gdsfactory/pull/1480)

## 6.69.0

- Some changes to section based ring and ViaStacks [PR](https://github.com/gdsfactory/gdsfactory/pull/1475)
- add dataprep [PR](https://github.com/gdsfactory/gdsfactory/pull/1470)
- `centre` can now also be specified when `port` is used when using add_port. [PR](https://github.com/gdsfactory/gdsfactory/pull/1478)
- Add corner rounding to dataprep notebook [PR](https://github.com/gdsfactory/gdsfactory/pull/1477)

## 6.68.0

- add gf.geometry.fillet and gf.geometry.boolean_polygons [PR](https://github.com/gdsfactory/gdsfactory/pull/1464)

## 6.67.0

- add modulator example [PR](https://github.com/gdsfactory/gdsfactory/pull/1463)

## 6.66.0

- add `Component.offset` [PR](https://github.com/gdsfactory/gdsfactory/pull/1462)

## 6.65.3

- make sure via stack fits at least one via [PR](https://github.com/gdsfactory/gdsfactory/pull/1461)
- add `Component.get_layer_names`
- make rich-click and optional dependency

## 6.65.0

- add `gf.components.add_trenches` with some components that use it (`coupler_trenches`, `bend_euler_trenches`, `ring_single_trenches`, `ring_double_trenches`) [PR](https://github.com/gdsfactory/gdsfactory/pull/1457)
- Component.move raises Error [PR](https://github.com/gdsfactory/gdsfactory/pull/1459)

## 6.64.2

- option to expose internal ports spiral_external_io [PR](https://github.com/gdsfactory/gdsfactory/pull/1452)
- fix delay_snake3 [PR](https://github.com/gdsfactory/gdsfactory/pull/1453)

## [6.64.1](https://github.com/gdsfactory/gdsfactory/pull/1451)

- add setter for `Port.x` and `Port.y`
- metals have default gap = 10
- `gf.routing.get_bundle(separation=None)` defaults to `cross_section.gap` and `cross_section.width`
- nicer CLI with rich-click
- remove CLI from docs

## 6.64.0

- add `add_pads_bot` and add_pads_top [PR](https://github.com/gdsfactory/gdsfactory/pull/1446)
- improve Component.add_ports [PR](https://github.com/gdsfactory/gdsfactory/pull/1448)
- add ports info to tests, and add pad_rectangular [PR](https://github.com/gdsfactory/gdsfactory/pull/1449)

## [6.63.0](https://github.com/gdsfactory/gdsfactory/pull/1445)

- add `bend_euler_s.info['length']`
- add `mzi(mirror_bot=False)` so you can mirror the bottom arm.

## 6.61.0

- Fix font free type [PR](https://github.com/gdsfactory/gdsfactory/pull/1438)
- ViaStack accounts for offset when placing vias [PR](https://github.com/gdsfactory/gdsfactory/pull/1437)

## 6.60.1

- Fixing connections on spiral_heater_fixed_length and removing duplicate remove_layers [PR](https://github.com/gdsfactory/gdsfactory/pull/1431)

## 6.60.0

- make klayout an optional dependency
- fix cutback_2x2 [PR](https://github.com/gdsfactory/gdsfactory/pull/1430)

## 6.59.1

- Improve cutback_2x2 [PR](https://github.com/gdsfactory/gdsfactory/pull/1422)
- change l_with_trenches `orient` to `mirror` to be consistent [PR](https://github.com/gdsfactory/gdsfactory/pull/1425)
- Switch to using a dict for angular extent in section based rings [PR](https://github.com/gdsfactory/gdsfactory/pull/1423)

## 6.58.0

- Bump jupyter-book from 0.14.0 to 0.15.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/1416)
- Bump tidy3d from 1.9.0 to 1.9.3
- [PR](https://github.com/gdsfactory/gdsfactory/pull/1419)
    - Port cross_section can be defined by name
    - snap `center` to grid in ring_section_based
    - increase bend radius for nitride and rib waveguides

## 6.57.2

- add `gf.geometry.layer_priority` operation [PR](https://github.com/gdsfactory/gdsfactory/pull/1413) [PR](https://github.com/gdsfactory/gdsfactory/pull/1415)

## 6.57.1

- add ports to `ring_section_based`
- removed required layers from pdk

## 6.57.0

- add angle parameter to ring component [PR](https://github.com/gdsfactory/gdsfactory/pull/1407)
- Added ring resonators with interleaved junction rings [PR](https://github.com/gdsfactory/gdsfactory/pull/1408)

## 6.56.0

- fix polygons at gdstk level before shapely meshing preprocessing [PR](https://github.com/gdsfactory/gdsfactory/pull/1396)
- add spiral for given length to `gf.components.spiral_racetrack_fixed_length` [PR](https://github.com/gdsfactory/gdsfactory/pull/1397)
- Don't warn on show [PR](https://github.com/gdsfactory/gdsfactory/pull/1402)
- adding a method in LayerStack to filter conveniently [PR](https://github.com/gdsfactory/gdsfactory/pull/1300)
- add `check_min_radius` to bend_s and rename `nb_points` to `npoints` for consistency [PR](https://github.com/gdsfactory/gdsfactory/pull/1406)

## 6.55.0

- Add `via_stack_from_rules` [PR](https://github.com/gdsfactory/gdsfactory/pull/1391)
- add `gf.fill.fill_rectangle_custom` which allows you to fill a region with a custom cell [PR](https://github.com/gdsfactory/gdsfactory/pull/1392)

## [6.54.0](https://github.com/gdsfactory/gdsfactory/pull/1388)

- add gap parameter to via

## [6.53.0](https://github.com/gdsfactory/gdsfactory/pull/1387)

- better error messages for `delay_snake`

## [6.52.0](https://github.com/gdsfactory/gdsfactory/pull/1385)

- install rich, jupytext and klayout, by default with `pip install gdsfactory`
- logger.level = 'WARNING' by default
- gf --version prints version of all packages
- add `gf.config.print_version()` and `gf.config.print_version_pdks()`

## 6.51.0

- Allow to change the crossection of spiral_racetrack [PR](https://github.com/gdsfactory/gdsfactory/pull/1384)
- Add sidewall tolerance [PR](https://github.com/gdsfactory/gdsfactory/pull/1382)

## 6.50.0

- Fix Component.to_stl() [PR](https://github.com/gdsfactory/gdsfactory/pull/1383)
- raise warning with uncached cells [PR](https://github.com/gdsfactory/gdsfactory/pull/1381)
- improve STL export [PR](https://github.com/gdsfactory/gdsfactory/pull/1380)
- allow choosing femwell solver in plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/1377)

## [6.49.0](https://github.com/gdsfactory/gdsfactory/pull/1375)

- install uses copy instead of symlink if symlink fails
- add gf.cross_section.rib_with_trenches
- use python instead of jupyter notebooks to build docs
- logger does not automatically print the version of the tool
- gf --version prints all plugin versions

## 6.48.0

- Transition does not inherit from CrossSection [PR](https://github.com/gdsfactory/gdsfactory/pull/1366)
- Pin conda and add layer_offsets to via_stack [PR](https://github.com/gdsfactory/gdsfactory/pull/1364)
- Rework Meep get_materials logic [PR](https://github.com/gdsfactory/gdsfactory/pull/1361)

## 6.47.1

- Fix meep neff fdtd [PR](https://github.com/gdsfactory/gdsfactory/pull/1359)
- make sure derived layers also work with Lumerical FDTD plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/1360)

## 6.47.0

- fix meep MPI [PR](https://github.com/gdsfactory/gdsfactory/pull/1358)
- add Component.plot_widget() [PR](https://github.com/gdsfactory/gdsfactory/pull/1356)
- Update S-parameter plotting [PR](https://github.com/gdsfactory/gdsfactory/pull/1354)

## 6.45.0

- propagate cross_section arg into taper [PR](https://github.com/gdsfactory/gdsfactory/pull/1348)
- use simple image when rendering docs [PR](https://github.com/gdsfactory/gdsfactory/pull/1348)
- Distributed model building / corner analysis [PR](https://github.com/gdsfactory/gdsfactory/pull/1343)

## 6.44.0

- Add layer selector widget to notebook view [PR](https://github.com/gdsfactory/gdsfactory/pull/1312)
- Sweep Fabrication [PR](https://github.com/gdsfactory/gdsfactory/pull/1309)
- Adding mode converter [PR](https://github.com/gdsfactory/gdsfactory/pull/1325)
- Add cascaded MZI filter example [PR](https://github.com/gdsfactory/gdsfactory/pull/1323)
- Derived layers work for more than one etched layer [PR](https://github.com/gdsfactory/gdsfactory/pull/1318)
- switch notebooks to [jupytext](https://marketplace.visualstudio.com/items?itemName=donjayamanne.vscode-jupytext) [PR](https://github.com/gdsfactory/gdsfactory/pull/1335)
- add mode converter [PR](https://github.com/gdsfactory/gdsfactory/pull/1325)
- add length to spiral heater [PR](https://github.com/gdsfactory/gdsfactory/pull/1334)
- Fix port in grating simulation to avoid substrate overlap [PR](https://github.com/gdsfactory/gdsfactory/pull/1329)

## [6.43.0](https://github.com/gdsfactory/gdsfactory/pull/1297)

- if no active PDK logger outputs warning, and activates generic PDK. In the future we won't activate the generic pdk by default.
- dither_pattern and hatch_pattern can be None
- Replace _ACTIVE_PDK with `gf.get_active_pdk()` as underscored variables are not supposed to be read or modified directly.
- Add coupler_straight_asymmetric [PR](https://github.com/gdsfactory/gdsfactory/pull/1299)
- fix devsim plugin and add a test [PR](https://github.com/gdsfactory/gdsfactory/pull/1304) [PR](https://github.com/gdsfactory/gdsfactory/pull/1303)
- add `Component.get_ports_pandas()` [PR](https://github.com/gdsfactory/gdsfactory/pull/1308) [PR](https://github.com/gdsfactory/gdsfactory/pull/1307/)

## 6.42.0

- don't expose generic LAYER_VIEWS [PR](https://github.com/gdsfactory/gdsfactory/pull/1295)

## 6.41.0

- Improve decorator docs [PR](https://github.com/gdsfactory/gdsfactory/pull/1287)
- Neural net interpolation & MEEP corner iterator [PR](https://github.com/gdsfactory/gdsfactory/pull/1285)
- add `mmi1x2_with_sbend` and `mmi2x2_with_sbend` [PR](https://github.com/gdsfactory/gdsfactory/pull/1283), [PR](https://github.com/gdsfactory/gdsfactory/pull/1286) and [PR](https://github.com/gdsfactory/gdsfactory/pull/1288)

## [6.40.0](https://github.com/gdsfactory/gdsfactory/pull/1281)

- fix devsim plugin
- run meshing notebooks
- add variability analysis [PR](https://github.com/gdsfactory/gdsfactory/pull/1280)

## 6.38.0

- update tidy3d from 1.8.3 to 1.8.4
- Add database simulation example notebook [PR](https://github.com/gdsfactory/gdsfactory/pull/1217)
- Add jinja based YAML circuits [PR](https://github.com/gdsfactory/gdsfactory/pull/1275)
- [PR](https://github.com/gdsfactory/gdsfactory/pull/1277)
    - add `Pdk.bend_points_distance` parameter for bends
    - add database to docs
    - make sure euler and arc have always enough points, as suggested by Orion in https://github.com/gdsfactory/gdsfactory/issues/1090


## [6.37.0](https://github.com/gdsfactory/gdsfactory/pull/1264)

- better port names for grating couplers, include `opt_te_wavelength_fiberAngle`
- remove grating coupler markers
- remove `Component.unlock()` in some functions that add labels. Cached components should never be modified.
- `add_fiber_single` and `add_fiber_array` don't add labels by default

## 6.36.0

- add tests and better docs for get_bundle_all_angle [PR](https://github.com/gdsfactory/gdsfactory/pull/1257)
- Fix flatten refs recursive [PR](https://github.com/gdsfactory/gdsfactory/pull/1258)

## [6.35.1](https://github.com/gdsfactory/gdsfactory/pull/1254)

- use constants in pdk
- default max points to 4e3

## 6.35.0

- Set KLayout technology to active PDK if available [PR](https://github.com/gdsfactory/gdsfactory/pull/1250)
- add thermal disk [PR](https://github.com/gdsfactory/gdsfactory/pull/1250)

## 6.34.0

- deprecate gdsfactory.types in favor of gdsfactory.typings as it was shadowing builtin types module [PR](https://github.com/gdsfactory/gdsfactory/pull/1241)
    - gdsfactory.types -> gdsfactory.typings
    - gf.types -> gdsfactory.typings
- Created meandered heater using doped silicon [PR](https://github.com/gdsfactory/gdsfactory/pull/1242)

## 6.33.0

- fix kweb and set log level to warning instead of debug [PR](https://github.com/gdsfactory/gdsfactory/pull/1237)
- Fix handling of gdstk.FlexPath in remap_layers [PR](https://github.com/gdsfactory/gdsfactory/pull/1238)
- make Component.remap_layers safe [PR](https://github.com/gdsfactory/gdsfactory/pull/1240)

## 6.32.0

- add general JAX interpolator [PR](https://github.com/gdsfactory/gdsfactory/pull/1230)
- add ring_single_pn and ring_double_pn [PR](https://github.com/gdsfactory/gdsfactory/pull/1228)

## 6.31.0

- rename show_jupyter to plot_jupyter [PR](https://github.com/gdsfactory/gdsfactory/pull/1226)
- use ComponentSpec string for via_stack_heater_mtop and resistance_meander [PR](https://github.com/gdsfactory/gdsfactory/pull/1224)

## 6.30.2

- Fix wafer layer in 3D meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1222)

## 6.30.1

- fix extract layers [PR](https://github.com/gdsfactory/gdsfactory/pull/1221)

## 6.30.0

- add all angle router [PR](https://github.com/gdsfactory/gdsfactory/pull/1216)
- add kweb integration [PR](https://github.com/gdsfactory/gdsfactory/pull/1220)

## 6.29.0

- add plot_klayout, plot_holoviews, plot_matplotlib. plot defaults to plot_klayout [PR](https://github.com/gdsfactory/gdsfactory/pull/1216)

## 6.28.0

- add flatten_offgrid_references as a write_gds flag [PR](https://github.com/gdsfactory/gdsfactory/pull/1211)
- add non-manhattan routing doc and document flatten_offgrid_references docs [PR](https://github.com/gdsfactory/gdsfactory/pull/1214)

## 6.27.0

- add options for 3D meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1207)
- [PR](https://github.com/gdsfactory/gdsfactory/pull/1206)
    - remove add_bbox_siepic
    - move tests from module to repo
    - update tidy3d to 1.8.3
    - document plugins in README.md https://github.com/gdsfactory/gdsfactory/issues/1205

## 6.26.0

- use klayout as the default jupyter notebook viewer [PR](https://github.com/gdsfactory/gdsfactory/pull/1200)

## 6.25.2
- fix mesh with 3D holes [PR](https://github.com/gdsfactory/gdsfactory/pull/1196)

## 6.25.0

- add flat netlist to `Component.get_netlist_flat()` [PR](https://github.com/gdsfactory/gdsfactory/pull/1192)

## 6.24.0

- add devcontainer similar to as in [this package](https://github.com/microsoft/azure-quantum-tgp)
- `pdk.get_cross_section` can accept CrossSection kwargs if cross-section already created [PR](https://github.com/gdsfactory/gdsfactory/pull/1189)

## 6.23.0

- femwell mode solver improvements [PR](https://github.com/gdsfactory/gdsfactory/pull/1166)
- Add interdigital capacitor [PR](https://github.com/gdsfactory/gdsfactory/pull/1167)

## 6.22.2

- move gdslib to home and use tidy3d instead of MPB for SAX tutorial [PR](https://github.com/gdsfactory/gdsfactory/pull/1153)
- Add "wafer_padding" argument to meshers to define how much larger than bounding box the LAYER.WAFER layers should be [PR](https://github.com/gdsfactory/gdsfactory/pull/1160)
- reduce femwell cache size [PR](https://github.com/gdsfactory/gdsfactory/pull/1158)
- schematic editor exposes port widget and load ports from YAML [PR](https://github.com/gdsfactory/gdsfactory/pull/1161)

## [6.22.0](https://github.com/gdsfactory/gdsfactory/pull/1151)

- fix mode hashing for tidy3d mode solver. Add `plot_power=True` flag that can also plot field if False.
- change gdslib branch to main
- install gdslib in $HOME/.gdsfactory
- remove empty cells from notebooks
- move gdslib to home and use tidy3d instead of MPB for SAX tutorial [PR](https://github.com/gdsfactory/gdsfactory/pull/1153)

## [6.21.0](https://github.com/gdsfactory/gdsfactory/pull/1150)
- Fix lumerical FDTD notebooks
- add gf.config.rich_output
- layer stack can be addressed as a dict

## [6.20.0](https://github.com/gdsfactory/gdsfactory/pull/1144)

- update femwell from 0.0.3 to 0.0.4
- update tidy3d from 1.8.1 to 1.8.2

## 6.19.4

- add example notebook for a generic optimiser implementation with Ray Tune [PR](https://github.com/gdsfactory/gdsfactory/pull/1137)
- add ring_single_coupler_bend [PR](https://github.com/gdsfactory/gdsfactory/pull/1137)
- add ray optimiser example [PR](https://github.com/gdsfactory/gdsfactory/pull/1139)
- fix phidl issues with bends with not enough points [PR](https://github.com/gdsfactory/gdsfactory/pull/1140)
- add name argument added to cross_section function [PR](https://github.com/gdsfactory/gdsfactory/pull/1142)

## [6.19.2](https://github.com/gdsfactory/gdsfactory/pull/1122)

- expose plot options in meep solver

## [6.19.1](https://github.com/gdsfactory/gdsfactory/pull/1121)

- install tidy3d at the end in make plugins so it installs shapely 1.8.4
- add LayerStack and LayerLevel to gf.typings
- silent logger by default

## 6.19.0

- simplify technology into a separate module [PR](https://github.com/gdsfactory/gdsfactory/pull/1014) [PR](https://github.com/gdsfactory/gdsfactory/pull/1102) [PR](https://github.com/gdsfactory/gdsfactory/pull/1109). You need to rename:
    - LAYER_COLORS -> LAYER_VIEWS
    - layer_colors -> layer_views
    - gf.layers.load_lyp -> gf.technology.LayerViews.from_lyp
    - GENERIC -> GENERIC_PDK
    - gdsfactory.tech -> gdsfactory.technology
    - gdsfactory.geometry.get_xsection_script import get_xsection_script -> gdsfactory.generic_tech.get_klayout_pyxs import get_klayout_pyxs
    - get_xsection_script -> get_klayout_pyxs
- move klayout to gdsfactory/generic_tech/klayout, you will need to run `gf tool install` to link `klive`

- add Implant simulation [PR](https://github.com/gdsfactory/gdsfactory/pull/1106)
- The fill_rectangle function wants either a list or layers for fill_inverter or a list of bools. Before the code did not properly accept a list of layers. [PR](https://github.com/gdsfactory/gdsfactory/pull/1107)

## 6.18.4

- fix path_smooth with near-collinear points [PR](https://github.com/gdsfactory/gdsfactory/pull/1093)

## 6.18.3

- fix transition also tapers cladding_layers and cladding offsets [PR](https://github.com/gdsfactory/gdsfactory/pull/1089)

## 6.18.2

- add port can infer width from cross_section [PR](https://github.com/gdsfactory/gdsfactory/pull/1089)

## 6.18.1

- transition also tapers cladding_layers and cladding offsets [PR](https://github.com/gdsfactory/gdsfactory/pull/1082)

## 6.18.0

- TCAD adaptative meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1074)
- fix CLI docs [PR](https://github.com/gdsfactory/gdsfactory/pull/1076)
- cache FEM modes [PR](https://github.com/gdsfactory/gdsfactory/pull/1079)

## [6.17.0](https://github.com/gdsfactory/gdsfactory/pull/1065)

- basic conformal 3D meshing

## [6.16.3](https://github.com/gdsfactory/gdsfactory/pull/1064)

- update tidy3d from 1.8.0 to 1.8.1
- fix write_sparameters_meep_mpi, after default `read_metadata=False` in import_gds

## 6.16.2

- default read_metadata=False in import_gds
- fix write_labels_klayout to be consistent with write_labels_gdstk (return angle as well)

## 6.16.1

- remove straight Pcell parameter from many components that do not need it [PR](https://github.com/gdsfactory/gdsfactory/pull/1061)

## 6.16.0

- remove bbox and pins siepic layers from strip cross-section  [PR](https://github.com/gdsfactory/gdsfactory/pull/1057)
- add more database examples and rename circuit to component for consistency [PR](https://github.com/gdsfactory/gdsfactory/pull/1058)

## 6.15.3

- Cache S-parameters from EME similarly to what is done for FDTD [PR](https://github.com/gdsfactory/gdsfactory/pull/1052)
- mirroring cross-section at class level [PR](https://github.com/gdsfactory/gdsfactory/pull/1050)

## 6.15.2

- add mirroring option to pn xs [PR](https://github.com/gdsfactory/gdsfactory/pull/1048)

## 6.15.1

- Pn xs offset [PR](https://github.com/gdsfactory/gdsfactory/pull/1043)
- allow multiple rows in straight_heater_meander [PR](https://github.com/gdsfactory/gdsfactory/pull/1046)
- update straight_pin docs [PR](https://github.com/gdsfactory/gdsfactory/pull/1045)

## 6.15.0

- Add femwell mode solver [PR](https://github.com/gdsfactory/gdsfactory/pull/1032)
- MEOW improvements [PR](https://github.com/gdsfactory/gdsfactory/pull/1033)

## 6.14.1

- fix bend_s width [issue](https://github.com/gdsfactory/gdsfactory/discussions/1028)

## 6.14.0

- fix tidy3d plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/1011)
- parse empty polygons uz meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1023)
- add Crow variant with couplers [PR](https://github.com/gdsfactory/gdsfactory/pull/1021)
- remove simphony [PR](https://github.com/gdsfactory/gdsfactory/pull/1024)

## 6.13.0

- Add z_to_bias processing in xy meshing [PR](https://github.com/gdsfactory/gdsfactory/pull/1011)
- Add MEOW EME plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/1015)
- update tidy3d to 1.8.0

## 6.12.0

- Add coupled rings (CROW) [PR](https://github.com/gdsfactory/gdsfactory/pull/1009)
- add LICENSE to pypi package [PR](https://github.com/gdsfactory/gdsfactory/pull/1010)

## [6.11.0](https://github.com/gdsfactory/gdsfactory/pull/1007)

- add option to register materials_index into a PDK.
- tidy3d and meep can use the same material_index from the registered materials.

## 6.10.0

- If sidewall, can specific at which height gds layer width equals structural width [PR](https://github.com/gdsfactory/gdsfactory/pull/994)
- fix for shapely 2.0 [PR](https://github.com/gdsfactory/gdsfactory/pull/996)
- improve klayout tech for 0.28 [PR](https://github.com/gdsfactory/gdsfactory/pull/998)
- add Option to have the mesh elements defined by material instead of layer [PR](https://github.com/gdsfactory/gdsfactory/pull/1002)
- add layer_grating to some of the gratings in case the grating needs to be defined with different layers. By default = None so it does not change existing gratings [PR](https://github.com/gdsfactory/gdsfactory/pull/1003)

## 6.9.0

- fix grating coupler slab and use better variable names [PR](https://github.com/gdsfactory/gdsfactory/pull/991)
- add database [PR](https://github.com/gdsfactory/gdsfactory/pull/990)

## 6.8.0

- write_cells works with multiple top level cells [PR](https://github.com/gdsfactory/gdsfactory/pull/978)
- add thickness_tolerance to LayerLevel [PR](https://github.com/gdsfactory/gdsfactory/pull/981)
- add z_to_bias map per layer level [PR](https://github.com/gdsfactory/gdsfactory/pull/983)
- remove many files thanks to pyproject.toml [PR](https://github.com/gdsfactory/gdsfactory/pull/977)

## 6.7.0

- Read raw cells [PR](https://github.com/gdsfactory/gdsfactory/pull/972)

## 6.6.0

- add greek cross component: Process control monitor to test implant sheet resistivity and linewidth variations. [PR](https://github.com/gdsfactory/gdsfactory/pull/965)
- add waveguide termination component. Terminate dangling waveguides with heavily doped narrow taper to protect modulator, detectors, and other active electronics from stray light. [PR](https://github.com/gdsfactory/gdsfactory/pull/964)
- rename master branch to main

## 6.5.0

- Add schematic symbols [PR](https://github.com/gdsfactory/gdsfactory/pull/958)
- Minor gmsh update [PR](https://github.com/gdsfactory/gdsfactory/pull/959)
- Fix assorted typos and formatting [PR](https://github.com/gdsfactory/gdsfactory/pull/961)

## [6.4.0](https://github.com/gdsfactory/gdsfactory/pull/955)

- improve meshing capabilities, add installer and docs

## 6.3.5

- add default 0 rotation to schematic and separation=5um for routes [PR](https://github.com/gdsfactory/gdsfactory/pull/945)
- Generic mesh refinement, refactoring [PR](https://github.com/gdsfactory/gdsfactory/pull/941)
- move python macros to python, (macros is for ruby), add klive path to print [PR](https://github.com/gdsfactory/gdsfactory/pull/943)
- fix position of rotated "no orientation" ports [PR](https://github.com/gdsfactory/gdsfactory/pull/951)
- Fix grating_coupler_elliptical_arbitrary component with how the ellipses are drawn. In the previous implementation, a constant neff is assumed for each grating unit cell, which is then used to determine the ellipticity of the teeth. However, for an apodized grating coupler, neff changes depending on the unit cell geometry (e.g. on duty cycle), so an apodized grating coupler would be drawn incorrectly. In addition, neff is not needed as an input, because it can be calculated from the period, wavelength, cladding index, and coupling angle, which are already inputs to the component. [PR](https://github.com/gdsfactory/gdsfactory/pull/953)

## [6.3.4](https://github.com/gdsfactory/gdsfactory/pull/939)

- replace lxml with the built-in xml modules to support python 3.11

## [6.3.3](https://github.com/gdsfactory/gdsfactory/pull/937)

- fix rotations in schematic
- gf tool install installs klayout as a salt package
- include material and name into 2.D view script


## 6.3.0

- Schematic-Driven Layout flow [PR](https://github.com/gdsfactory/gdsfactory/pull/920)
- from __future__ import annotations
from functools import partial to all files and makes the docs cleaner by rendering the type aliases rather than the expanded type [PR](https://github.com/gdsfactory/gdsfactory/pull/923)
- Add routes to gdsfactory klayout macro [PR](https://github.com/gdsfactory/gdsfactory/pull/918)
- fix missing conversion from rad (gdstk) to deg [PR](https://github.com/gdsfactory/gdsfactory/pull/927)
- better error message when failing to import missing gdscell [PR](https://github.com/gdsfactory/gdsfactory/pull/926)
- mzi lattice mmi [PR](https://github.com/gdsfactory/gdsfactory/pull/920)
- prepare release [PR](https://github.com/gdsfactory/gdsfactory/pull/929)
    * keep python3.7 compatibility, by removing `:=` [Walrus operator](https://realpython.com/python-walrus-operator/)
    * move schematic driven flow notebook from samples to docs
    * add test coverage for write_labels_gdstk

## 6.2.6

- import_gds can import any cell (not only top_level cells) [PR](https://github.com/gdsfactory/gdsfactory/pull/917)

## 6.2.5

- mode solvers get modes_path from PDK.modes_path [PR](https://github.com/gdsfactory/gdsfactory/pull/915)
- remove gf.CONFIG [PR](https://github.com/gdsfactory/gdsfactory/pull/916)

## 6.2.4

- straight propagates to route filter in get_bundle [PR](https://github.com/gdsfactory/gdsfactory/pull/914)
- update pre-commit hooks and simplify CI/CD [PR](https://github.com/gdsfactory/gdsfactory/pull/913)
- fix delay length [PR](https://github.com/gdsfactory/gdsfactory/pull/912)

## [6.2.3](https://github.com/gdsfactory/gdsfactory/pull/907)

- fix add ports from paths and polygons
- add tests

## [6.2.2](https://github.com/gdsfactory/gdsfactory/pull/905)

- fix import_gds works with arrays
- ComponentReference allows vectors v1 and v2

## [6.2.1](https://github.com/gdsfactory/gdsfactory/pull/903)

- difftest prompts you whether you want to do the xor diff

## [6.2.0](https://github.com/gdsfactory/gdsfactory/pull/895)

- mmi input waveguide width is optional and defaults to cross_section.width
- rename reflect_h to mirror_x and reflect_v to mirror_y [PR](https://github.com/gdsfactory/gdsfactory/pull/896)
- gdsfactory cells working on klayout [PR](https://github.com/gdsfactory/gdsfactory/pull/899)
- fix grid ports [PR](https://github.com/gdsfactory/gdsfactory/pull/900)

## 6.1.1

- fix Docker container by installing gdspy with mamba
- fix outline [issue](https://github.com/gdsfactory/gdsfactory/issues/888)
- fix None orientation connect [PR](https://github.com/gdsfactory/gdsfactory/pull/890)
- clean_value_json can handle Polygons [issue](https://github.com/gdsfactory/gdsfactory/issues/889)

## [6.1.0](https://github.com/gdsfactory/gdsfactory/pull/884)

- Native read and write oasis support

## [6.0.7](https://github.com/gdsfactory/gdsfactory/pull/882)

- fix ComponentReference.get_polygons broken when by_spec is layer and as_array is True
- fix Component.movex
- better names for transformed

## 6.0.6

- Remove lytest [PR](https://github.com/gdsfactory/gdsfactory/pull/878)
- Handle non-existing polygons [PR](https://github.com/gdsfactory/gdsfactory/pull/874/)
- fixing the port cross sections of an extruded transition and adding test [PR](https://github.com/gdsfactory/gdsfactory/pull/876)
- Fixing and simplifying remove_layers [PR](https://github.com/gdsfactory/gdsfactory/pull/873)
- Fix and improve speed of flatten and absorb [PR](https://github.com/gdsfactory/gdsfactory/pull/875)
- Components.__getitem__ is consistent with ComponentReference.__getitem__ [PR](https://github.com/gdsfactory/gdsfactory/pull/879)

## 6.0.5

- remove pytest from `requirements.txt` as it's already on `requirements_dev.txt`
- Ensure consistent u and z bounds when meshing a uz cross-section [PR](https://github.com/gdsfactory/gdsfactory/pull/871)

## 6.0.4

- expose uz_mesh functions [PR](https://github.com/gdsfactory/gdsfactory/pull/869)

## 6.0.3

- fixes ComponentReference.translate() [PR](https://github.com/gdsfactory/gdsfactory/pull/858)
- Keep reference names on copy, and allow setting on init [PR](https://github.com/gdsfactory/gdsfactory/pull/859)
  - Ensures that reference names are retained when copying a component
  - Fixes an issue when a ComponentReference's name is set before it has an owner
  - Adds name as a parameter to ComponentReference's constructor, so it can be set on initialization
- Fixes serialization of numpy arrays such that they are retained as numeric types (this was also the old, gdsfactory 5.x behaviour). Increases the default maximum # of digits retained when serializing arrays from 3 to 8 (this could cause side effects if not enough digits are retained, and 8 should still be plenty to ensure consistency across machines... 1e-8 is the default atol threshold in np.isclose()). [PR](https://github.com/gdsfactory/gdsfactory/pull/860)
- Devsim plugin with GMSH plugin backend [PR](https://github.com/gdsfactory/gdsfactory/pull/861)


## 6.0.2

- add trim [PR](https://github.com/gdsfactory/gdsfactory/pull/855)

## 6.0.1

- fix import ports from siepic pins [PR](https://github.com/gdsfactory/gdsfactory/pull/854)
- add adiabatic taper [PR](https://github.com/gdsfactory/gdsfactory/pull/853)
- Fix shortcut installation script [PR](https://github.com/gdsfactory/gdsfactory/pull/851)

## [6.0.0](https://github.com/gdsfactory/gdsfactory/pull/833)

- port gdsfactory from gdspy to gdstk: faster booleans, loading and writing GDS files.
- remove Inheritance of Component, ComponentReference, Polygon, Label from gdspy
    - use gdstk.Label and gdstk.Polygon directly (no inheritance)
    - Label.origin instead of Label.position
- ComponentReference, has rows and columns to represent removed `CellArray`
- add loss model for modesolver [PR](https://github.com/gdsfactory/gdsfactory/pull/831)
- fixes [PR](https://github.com/gdsfactory/gdsfactory/pull/835)
    * remove deprecated aliases
    * fix to_3d
    * fix quickplo
- Fix gmeep get_simulation center issue [PR](https://github.com/gdsfactory/gdsfactory/pull/834)
- get_polygons [PR](https://github.com/gdsfactory/gdsfactory/pull/846)
- replace deprecated reflect by mirror [PR](https://github.com/gdsfactory/gdsfactory/pull/838)
- remove aliases [PR](https://github.com/gdsfactory/gdsfactory/pull/835)

## 5.56.0

- rename add_fidutials to add_fiducials (it was misspelled before) [PR](https://github.com/gdsfactory/gdsfactory/pull/827)

## 5.55.0

- fix path defined port issues when absorbing [issue](https://github.com/gdsfactory/gdsfactory/issues/816)
- minor spiral fixes [PR](https://github.com/gdsfactory/gdsfactory/pull/822)
- klayout tech fixes [PR](https://github.com/gdsfactory/gdsfactory/pull/824)

## 5.54.0

- Add shortcut during installation [PR](https://github.com/gdsfactory/gdsfactory/pull/817/files)

## 5.53.0

- get_material from meep can also use tidy3d database [PR](https://github.com/gdsfactory/gdsfactory/pull/813)

## 5.51.0

- add devsim installation to installer

## 5.50.0

- sanitize generic pdk args [PR](https://github.com/gdsfactory/gdsfactory/pull/808)
- fix spiral racetrack [PR](https://github.com/gdsfactory/gdsfactory/pull/809)
- update meep adjoint default values [PR](https://github.com/gdsfactory/gdsfactory/pull/811)

## 5.49.0

- fix devsim [PR](https://github.com/gdsfactory/gdsfactory/pull/806)

## 5.47.2

- make package install work on Windows [PR](https://github.com/gdsfactory/gdsfactory/pull/805)

## 5.47.1

- improve simulation.plot.plot_sparameter kwargs [PR](https://github.com/gdsfactory/gdsfactory/pull/804)
    - replace with_sparameter_labels with with_simpler_input_keys

## 5.47.0

- integrate flayout to add all of generic components into KLayout [PR](https://github.com/gdsfactory/gdsfactory/pull/797)
- group gdsfactory klayout plugin items in a single menu [PR](https://github.com/gdsfactory/gdsfactory/pull/801)

## 5.46.0

- meep plot2D improvements [PR](https://github.com/gdsfactory/gdsfactory/pull/792)
- fix show_subports causes error with CellArray references [issue](https://github.com/gdsfactory/gdsfactory/issues/791)
- access reference ports more easily [PR](https://github.com/gdsfactory/gdsfactory/pull/794)

## 5.45.1

- add spiral heater phase shifter [PR](https://github.com/gdsfactory/gdsfactory/pull/786)

## 5.45.0

- add devsim and mkl to `pip install gdsfactory[full]`
- tidy3d modesolver sweep width computes also fraction_te [PR](https://github.com/gdsfactory/gdsfactory/pull/784)

## 5.44.0

- remove TECH, use gf.get_constant instead
- Add optional decay cutoff argument to write_sparameters_meep [issue](https://github.com/gdsfactory/gdsfactory/issues/779)

## 5.43.2

- add `gf.cross_section.metal_routing` and use `metal_routing` string instead of function for metal routing functions [PR](https://github.com/gdsfactory/gdsfactory/pull/778)
- Component.remove_layers returns flat copy when hierarchical=True

## 5.43.1

- upgrade tidy3d-beta from 1.7.0 to 1.7.1
- straight_heater_meander accepts straight_widths

## 5.43.0

- add devsim from pypi [PR](https://github.com/gdsfactory/gdsfactory/pull/776)
- dbr_tapered rewrite [PR](https://github.com/gdsfactory/gdsfactory/pull/775)
- Recursively netlist transformed (flattened) references [PR](https://github.com/gdsfactory/gdsfactory/pull/772) enables recursive netlisting of references which have been transformed and flattened via the flatten_offgrid_references decorator, such that they can be properly simulated, i.e. with sax.
- Don't re-apply PDK decorator in get_component [PR](https://github.com/gdsfactory/gdsfactory/pull/773)

## 5.42.0

- fix gf tool install when file already exists [PR](https://github.com/gdsfactory/gdsfactory/pull/769)

## 5.41.1

- remove Port.uid [PR](https://github.com/gdsfactory/gdsfactory/pull/768)
    - add gf.components.array(add_ports=True)

## 5.41.0

- `gf watch` watches python files as well as `pic.yml` [PR](https://github.com/gdsfactory/gdsfactory/pull/767)

## 5.40.0

- KLayout technology module improvements:
  - Added CustomPatterns class, which contains lists of CustomDitherPatterns and CustomLineStyles
  - Added yaml import/export for CustomPatterns and LayerDisplayProperties
  - LayerView group members are stored in a list rather than a dict
  - Added mebes reader options to KLayoutTechnology export since they aren't created by the KLayout API
  - Fixed some formatting issues when writing .lyt files
  - 2.5D section of .lyt now uses zmin, zmax as it should, rather than zmin, thickness as it did


## 5.39.0

- upgrade tidy3d to 1.7.0

## 5.38.0

- add inverse design capabilities with Meep [PR](https://github.com/gdsfactory/gdsfactory/pull/761)

## 5.37.2

- remove `__init__` from klayout
- move klayout.get_xsection_script to geometry.get_xsection_script
- improve klayout technology [PR](https://github.com/gdsfactory/gdsfactory/pull/757)
    * Allow users to set layer_pattern regex
    * Remove LayerStack from KLayoutTechnology and take as argument for export_technology_files instead.


## 5.37.1

- fix is_on_grid [PR](https://github.com/gdsfactory/gdsfactory/pull/754)

## 5.37.0

- remove Component.version and Component.changelog, change tutorial to recommend keeping Pcell changelog in the docstring. [PR](https://github.com/gdsfactory/gdsfactory/pull/752)

## 5.36.0

- Add thermal solver [PR](https://github.com/gdsfactory/gdsfactory/pull/739)
- remove phidl dependency [PR](https://github.com/gdsfactory/gdsfactory/pull/741)
- remove incremental naming from phidl
- remove Port.midpoint as it was deprecated since 5.14.0
- add freetype-py for using text with font and add components.text_freetype [PR](https://github.com/gdsfactory/gdsfactory/pull/743)

## 5.35.0

- Mesh2D renames boundaries [PR](https://github.com/gdsfactory/gdsfactory/pull/736)

## 5.34.0

- bump sax from 0.8.2 to 0.8.4
- add fixes for snapping to the grid, add parabolic_transition [PR](https://github.com/gdsfactory/gdsfactory/pull/733)
- add [installer for Windows, MacOs and Linux](https://github.com/gdsfactory/gdsfactory/releases)

## 5.33.0

- FEM mesher. Given component, line defined by (x_init, y_init), (x_final, y_final), and LayerStack, generate simple mesh of component cross-section. Mesh returned separately labels non-touching elements on different layers (for use in different solvers). Can provide dict with different resolution for different layers [PR](https://github.com/gdsfactory/gdsfactory/pull/729)
- add Coherent receiver (single and dual pol) Coherent transmitter (single and dual pol) [PR](https://github.com/gdsfactory/gdsfactory/pull/731)

## 5.32.0

- Read/write layer files (.lyp) and specify whether the layer number is displayed in KLayout [issue](https://github.com/gdsfactory/gdsfactory/issues/695) [PR](https://github.com/gdsfactory/gdsfactory/pull/724)
    - Read/write technology files (.lyt)
    - Requires the klayout.db module
    - 2.5D layer stacks can only be written due to incomplete API in 0.27.x (#1153)
- Bump sax from 0.8.1 to 0.8.2


## 5.31.0

- add new Pcells [PR](https://github.com/gdsfactory/gdsfactory/pull/717)
    * Dual polarization grating coupler
    * Straight Ge detector with contacts in the Si
    * MMI-based 90 degree hybrid
- add tests to new Pcells [PR](https://github.com/gdsfactory/gdsfactory/pull/720)
- add resolution_x and resolution_y Optional parameters to tidy3d modesolver. Fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/719) [PR](https://github.com/gdsfactory/gdsfactory/pull/721)

## 5.30.0

- Fix multilayer routing [PR](https://github.com/gdsfactory/gdsfactory/pull/707)
- Add tests and examples for multilayer routing [PR](https://github.com/gdsfactory/gdsfactory/pull/714)

## [5.29.0](https://github.com/gdsfactory/gdsfactory/pull/704)

- add sweep_neff, sweep_width, sweep_group_index gtidy3d.modes and ring model
- Absorption from DEVSIM [PR](https://github.com/gdsfactory/gdsfactory/pull/701)
    - DEVSIM PIN waveguides now return imaginary neff
    - Changes to tidy3D mode solver to allow running in a different interpreter for compatibility
- Added support for None orientation ports for get_bundle_from_steps. [PR](https://github.com/gdsfactory/gdsfactory/pull/702)
- bend_euler returns wire_corner if radius = None
- upgrade to tidy3d-beta 1.6.3

## [5.28.1](https://github.com/gdsfactory/gdsfactory/pull/698)

- upgrade to tidy3d-beta 1.6.2
- add functions to write a complete technology package for KLayout using the KLayoutTechnology class [PR](https://github.com/gdsfactory/gdsfactory/pull/696)
- remove unnamed layers

## [5.28.0](https://github.com/gdsfactory/gdsfactory/pull/691)

- Add avoid_layers, distance, and cost addition for turns for routing.get_route_astar [PR](https://github.com/gdsfactory/gdsfactory/pull/690)
- cross_sections `metal1`, `metal2`, `metal3` have `radius = None`
- routing.get_bundle uses `wire_corner` if `cross_section.radius=None`
- routing.get_route_astar uses `wire_corner` if `cross_section.radius=None`

## [5.27.1](https://github.com/gdsfactory/gdsfactory/pull/686)

- fix devsim TCAD units and examples.

## [5.27.0](https://github.com/gdsfactory/gdsfactory/pull/684)

- add A* router [PR](https://github.com/gdsfactory/gdsfactory/pull/683)

## 5.26.3

- fix tidy3d mode solver sweep width [PR](https://github.com/gdsfactory/gdsfactory/pull/682)
- devsim improvements. Add Modes with doping index perturbation [PR](https://github.com/gdsfactory/gdsfactory/pull/679)
    - Modify gtidy3D mode solver to handle local index perturbations.
    - New function to generate a Waveguide for mode solving from PIN semiconductor simulation.

## [5.26.2](https://github.com/gdsfactory/gdsfactory/pull/675)

- devsim TCAD improvements
    * remove wurlitzer
    * change devsim example classes CamelCase by snake_case

## 5.26.0

- add grating_coupler_elliptical uniform [PR](https://github.com/gdsfactory/gdsfactory/pull/668)
- generate KLayout technology files (.lyp) from the gdsfactory LayerColors, add structures that let you write (almost) all of the properties that .lyp files can take, including groups of layer properties. [PR](https://github.com/gdsfactory/gdsfactory/pull/662)
- via_stack has `port_type=placement` for intermediate ports and compass has default `port_type=placement` [PR](https://github.com/gdsfactory/gdsfactory/pull/661)
- get_netlist ignores ports with port_type='placement' [PR](https://github.com/gdsfactory/gdsfactory/pull/666)
- move gdsfactory.copy to Component.copy [PR](https://github.com/gdsfactory/gdsfactory/pull/660)
- clean install.py [PR](https://github.com/gdsfactory/gdsfactory/pull/657)
    - Fix a bug where calling make_symlink on an already-existing install would raise an error
    - Generalizes installing things to KLayout, and provides a new method for installing custom PDKs/technology to KLayout

## [5.25.1](https://github.com/gdsfactory/gdsfactory/pull/655)

- Component.plot() takes kwargs to configure the settings for matplotlib


## [5.25.0](https://github.com/gdsfactory/gdsfactory/pull/651)

- rewrite get_netlist() to be more robust and to warn about more issues in optical routing. [PR](https://github.com/gdsfactory/gdsfactory/pull/651)
- documentation improvements [PR](https://github.com/gdsfactory/gdsfactory/pull/654)

## [5.24.1](https://github.com/gdsfactory/gdsfactory/pull/650)

- fix lazy parallelism with new sparameter port naming conventions [PR](https://github.com/gdsfactory/gdsfactory/pull/649)

## [5.24.0](https://github.com/gdsfactory/gdsfactory/pull/644)

- write sparameters works with arbitrary port naming convention and different input modes. `o1@0,o2@0` for meep and tidy3d. where `o1` is in the input port `@0` is the first mode, and `o2@0` refers to `o2` port mode `0`
- add `csv_to_npz` function in `gf.simulation.convert_sparameters.py` to convert old sims into new ones.


## [5.23.1](https://github.com/gdsfactory/gdsfactory/pull/642)

- sort cells by name before writing gds to get a binary equivalent.


## [5.23.0](https://github.com/gdsfactory/gdsfactory/pull/641)

-  extended get_bundle to enable s_bend routing when there is no space for Manhattan routing. fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/55) [PR](https://github.com/gdsfactory/gdsfactory/pull/639)

## [5.22.3](https://github.com/gdsfactory/gdsfactory/pull/637)

- component_sequence has the same named scheme it used to have before adding named_references
- add flip option to component_sequence

## 5.22.2

- fail difftest when passing n to reject changes [PR](https://github.com/gdsfactory/gdsfactory/pull/635)
- don't cache import_gds in read.from_phidl. Add capability to pass a layer as a string to outline() [PR](https://github.com/gdsfactory/gdsfactory/pull/636)

## [5.22.0](https://github.com/gdsfactory/gdsfactory/pull/634)

- more robust geometric_hash [PR](https://github.com/amccaugh/phidl/pull/157) rounds to 0.1nm by default
- `Component.get_netlist()` returns a Dict instead of OmegaConf.DictConfig (faster)
- remove `Component.get_netlist_dict()` in favor of `Component.get_netlist()`

## [5.21.1](https://github.com/gdsfactory/gdsfactory/pull/633)

- reduce unnecessary usage of deepcopy

## [5.21.0](https://github.com/gdsfactory/gdsfactory/pull/631)

- Thanks to the counter `Component.add_array` has the same naming convention as `Component.add_ref` [PR](https://github.com/gdsfactory/gdsfactory/pull/630)
- remove picwriter dependency and lazy load scipy functions from init

## [5.20.0](https://github.com/gdsfactory/gdsfactory/pull/628)

- add file storage class that can use local file cache and cloud buckets [PR](https://github.com/gdsfactory/gdsfactory/pull/626)
- [PR](https://github.com/gdsfactory/gdsfactory/pull/624)
    * make reference naming more reliable, by replacing the alias and aliases attributes with `named_references` property called name, which is guaranteed to always stay in sync with Component.references. Because name and owner are both changed to properties, we also guard against strange cases where either are changed midway through creating a component.
    * add deprecation warning for alias/aliases
- add gf.read.import_gdspy
- add_ref adds default alias with incremental names using a counter.

## [5.19.1](https://github.com/gdsfactory/gdsfactory/pull/620)

- Rewrote model_from_gdsfactory to GDSFactorySimphonyWrapper. Similar to Simphony's SiPANN wrapper [PR](https://github.com/gdsfactory/gdsfactory/pull/619)
- `gf.routing.get_route()` has `with_sbend` that allows failing routes to route using Sbend.

## [5.19.0](https://github.com/gdsfactory/gdsfactory/pull/617)

- add_ref adds default alias with incremental names.
- get_netlist returns better instance names, uses aliases by default instead of labels.
- gf.read.from_yaml accepts a dict and DictConfig as well as filepath and string.
- Add s_bend option to gf.routing.get_route [PR](https://github.com/gdsfactory/gdsfactory/pull/616) closes [issue](https://github.com/gdsfactory/gdsfactory/issues/51)

## [5.18.5](https://github.com/gdsfactory/gdsfactory/pull/613)

- add layer_via and layer_metal to cross_section.pn. None by default.
- fix yaml rotation [issue](https://github.com/gdsfactory/gdsfactory/issues/295)
- add components.cutback_2x2
- add components.cutback_splitter
- make `pip install gdsfactory[tidy3d]` available instead of `pip install gdsfactory[full]`
- upgrade notebooks to `sax==0.8.0` and pin `sax==0.8.0` in `pip install gdsfactory[full]`

## 5.18.4

- fix AttributeError (component.info is sometimes None) [PR](https://github.com/gdsfactory/gdsfactory/pull/607)
- adding ports and copying child info in transformed cell wrapper [PR](https://github.com/gdsfactory/gdsfactory/pull/604)

## [5.18.3](https://github.com/gdsfactory/gdsfactory/pull/594)

- Remove picwriter dependencies [issue](https://github.com/gdsfactory/gdsfactory/issues/471)
- fix shear ports [PR](https://github.com/gdsfactory/gdsfactory/pull/596)
- fix recursive_netlist [PR](https://github.com/gdsfactory/gdsfactory/pull/595)
- add components.cutback_splitter
- adding ports and copying child info in transformed cell wrapper [PR](https://github.com/gdsfactory/gdsfactory/pull/604)

## [5.18.1](https://github.com/gdsfactory/gdsfactory/pull/594)

- add kwargs to bezier and change coupler_adiabatic from picwriter to adiabatic coupler

## [5.18.0](https://github.com/gdsfactory/gdsfactory/pull/593)

- include `rule_not_inside`, `rule_min_width_or_space` klayout DRC checks in gf.geometry.write_drc
- fix install instructions for SAX in docs

## [5.17.1](https://github.com/gdsfactory/gdsfactory/pull/590)

- copy_child_info updates component info
- mirror also propagates info
- add grating coupler info in add_fiber_single, add_fiber_array

## 5.17.0

- simplify write_sparameters code [PR](https://github.com/gdsfactory/gdsfactory/pull/581)
- clean code [PR](https://github.com/gdsfactory/gdsfactory/pull/582)
    - remove unused Coord2
    - move gds folder into tests/gds
    - move schemas folder into tests/schemas
    - move models to simulation/photonic_circuit_models
- simpler bend_s and bezier. User cross_section instead extrude_path [PR](https://github.com/gdsfactory/gdsfactory/pull/584)
- simplify code structure [PR](https://github.com/gdsfactory/gdsfactory/pull/586)
    - rename dft folder to labels
    - move write_labels from mask folder to `labels`
    - delete mask folder
- fix docstrings and apply pydocstyle pre-commit hook [PRs](https://github.com/gdsfactory/gdsfactory/pull/585)
- replace assert_on_2nm_grid to snap to 2nm grid [PR](https://github.com/gdsfactory/gdsfactory/pull/589)
- Add tcad cross-section simulation DEVSIM plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/587)
    - fully parametrized ridge PIN waveguide cross-section
    - basic DC simulation and visualization
    - example notebook, with DEVSIM installation instructions


## [5.16.0](https://github.com/gdsfactory/gdsfactory/pull/580)

- rename `gdsfactory/flatten_offgrid_references.py` to `gdsfactory/decorators.py`
- add test to flatten_offgrid_references and include documentation in the [PDK docs](https://gdsfactory.github.io/gdsfactory/notebooks/08_pdk.html#)
- pdk.activate clears cache

## [5.15.3](https://github.com/gdsfactory/gdsfactory/pull/579)

- [PR](https://github.com/gdsfactory/gdsfactory/pull/576)
    - fix offgrid ports connect
    - Component.copy() and Component.flatten() don't assign fixed names to the output Component which could create name collisions. It is safer to keep the new cell's default name (with UID) to prevent this. The thought is that even a copied or flattened cell should be eventually placed inside a cell function with @cell decorator, which will remove the anonymous and unpredictable names (for those of us who care). And for those who don't care as much, at least they won't get dangerous side effects!
- [PR](https://github.com/gdsfactory/gdsfactory/pull/578) changes get_netlist to only mate matching port types, and also to allow excluding specified port types from being netlisted (which is useful i.e. for non-logical placement ports)
- solve add_labels.get_labels mutability issue by returning list of labels

## [5.15.2](https://github.com/gdsfactory/gdsfactory/pull/575)

- fix move with string [issue](https://github.com/gdsfactory/gdsfactory/issues/572)  [PR](https://github.com/gdsfactory/gdsfactory/pull/573)
- fix docs

## [5.15.1](https://github.com/gdsfactory/gdsfactory/pull/571)

- expose import_oas to gf.read and Improve docs API
- klive shows gdsfactory version
- write_cells does not use partials

## [5.15.0](https://github.com/gdsfactory/gdsfactory/pull/567)

- read and write OASIS files using klayout API.

## [5.14.5](https://github.com/gdsfactory/gdsfactory/pull/565)

- fix mpirun path [PR](https://github.com/gdsfactory/gdsfactory/pull/561)
- fix model_from_csv [PR](https://github.com/gdsfactory/gdsfactory/pull/563/files)

## [5.14.5](https://github.com/gdsfactory/gdsfactory/pull/560)

- pack and grid expose components ports.
- add gdsfactory/samples/24_doe_2.py samples with unique labeling.

## [5.14.0](https://github.com/gdsfactory/gdsfactory/pull/554)

- Port has optional width that can get it from the cross_section
- Port does not inherit from phidl.Port. In python there should be only one obvious way to do things.
    * Deprecate Port.midpoint. Use Port.center instead.
    * Deprecate Port.position. Use Port.center instead.
    * Deprecate Port.angle. Use Port.orientation instead.
- [fix docs issues](https://github.com/gdsfactory/gdsfactory/issues/553)

## [5.13.0](https://github.com/gdsfactory/gdsfactory/pull/552)

- add gdsfactory.simulation.simphony.model_from_csv

## [5.12.28](https://github.com/gdsfactory/gdsfactory/pull/551)

- fix min_density and min_area Klayout write_drc

## [5.12.27](https://github.com/gdsfactory/gdsfactory/pull/550)

- add min area and min_density rules to klayout write drc deck
- upgrade phidl to 1.6.2
- lazy load matplotlib and pandas
- update tidy3d-beta from 1.4.2 to 1.5.0.
    - mode_solver has
        - precision: single or double.
        - filter_pol: te, tm or None.

## [5.12.22](https://github.com/gdsfactory/gdsfactory/pull/545)

- add xoffset1 and xoffset2 to straight_heater_doped_rib
- add taper_sc_nc to docs and tests
- add gdsfactory.simulation.add_simulation_markers, as well as example in the meep FDTD docs

## [5.12.21](https://github.com/gdsfactory/gdsfactory/pull/544)

- rename radius to min_bend_radius in spiral_double and spiral_archimedian
- add gdsfactory/klayout/get_xsection_script.py to get cross_section script for [klayout plugin](https://gdsfactory.github.io/klayout_pyxs/DocGrow.html)
- add pack and grid to documentation API

## [5.12.20](https://github.com/gdsfactory/gdsfactory/pull/542)

- rename double_spiral to spiral_double
- fix spiral_double.
    - start_angle = 0
    - end_angle = 180
- rename inner_radius to radius in spiral_double and spiral_archimedian

## [5.12.19](https://github.com/gdsfactory/gdsfactory/pull/539)

- lazy load matplotlib. Related to [issue](https://github.com/amccaugh/phidl/pull/159)
- make port.orientation None marker to be a cross.
- add settings label can ignore settings
- better message for symlinks in `gf tool install`
- fix write_cells code. Add a test.
- better mutability error message [fix issue](https://github.com/gdsfactory/gdsfactory/issues/538)
- add [YouTube Video link](https://www.youtube.com/channel/UCp4ZA52J1pH4XI5gvLjgB_g) to docs

## [5.12.16](https://github.com/gdsfactory/gdsfactory/pull/536)

- add klayout python cross_section scripts
- improve klayout drc decks. Add deep, tiled and default modes.
- add klayout drc and simulation docs API

## 5.12.14

- fix simphony MZI model for montecarlo simulations [PR](https://github.com/gdsfactory/gdsfactory/pull/530)
- add simphony montecarlo variability examples

## 5.12.13

- add extension_port_names to extend_ports [PR](https://github.com/gdsfactory/gdsfactory/pull/527)
- fix ring single simphony example [PR](https://github.com/gdsfactory/gdsfactory/pull/525)

## [5.12.12](https://github.com/gdsfactory/gdsfactory/pull/523)

- add `gdsfactory.simulation.gtidy3d.modes.WaveguideCoupler`

## [5.12.11](https://github.com/gdsfactory/gdsfactory/pull/522)

- add `gdsfactory.simulation.gtidy3d.modes.group_index`
- add `gdsfactory.simulation.gtidy3d.modes.sweep_width`
- add `gdsfactory.simulation.gtidy3d.modes.plot_sweep_width`

## [5.12.7](https://github.com/gdsfactory/gdsfactory/pull/513)

- get_sparameters_meep_mpi runs the mpirun command asynchronously. Direct stdout and stderr to a log file and console. [PR](https://github.com/gdsfactory/gdsfactory/pull/515)
    - It can't replace the current Popen call, as it doesn't handle the case of wait_to_finish=False, so it won't work with the get_sparameters_meep_batch code as-is.

## [5.12.6](https://github.com/gdsfactory/gdsfactory/pull/513)

- rename get_effective_index to get_effective_indices and add 2.5D FDTD demo
- [fix issue](https://github.com/gdsfactory/gdsfactory/issues/511)


## [5.12.5](https://github.com/gdsfactory/gdsfactory/pull/510)

- better docstrings with autodoc_typehints = "description"
- improve meep plugin.
    - remove port_field_monitor_name parameter (no longer needed) thanks to meep 1.23 introduced to use the energy in the whole simulation to determine when to terminate, which is a better termination condition than the energy at the ports. [PR](https://github.com/gdsfactory/gdsfactory/pull/495/files). Requires meep 1.23 or newer.
    - update termination condition for grating_coupler simulations.
    - rename effective permitivity to get_effective index. Change units from meters to um, and permitivities to refractive_index to be consistent with gdsfactory units in um.
- add `gf.generate_doe` [PR](https://github.com/gdsfactory/gdsfactory/pull/508/files)
- add add_center_section to CrossSection and cross_section for slot cross_section [PR](https://github.com/gdsfactory/gdsfactory/pull/509) [fixes](https://github.com/gdsfactory/gdsfactory/issues/506)

## [5.12.4](https://github.com/gdsfactory/gdsfactory/pull/502)

- function to calculate_effective_permittivity [PR](https://github.com/gdsfactory/gdsfactory/pull/501)
- Add MPB mode solver for cross sections [PR](https://github.com/gdsfactory/gdsfactory/pull/499)

## [5.12.2](https://github.com/gdsfactory/gdsfactory/pull/498)

- extract generating component list for doe into a separate function for use in pack_doe and elsewhere [fixes issue](https://github.com/gdsfactory/gdsfactory/issues/496)
- meep 1.23 introduced to use the energy in the whole simulation to determine when to terminate, which is a better termination condition than the energy at the ports. [PR](https://github.com/gdsfactory/gdsfactory/pull/495/files). Requires meep 1.23 or newer.

## [5.12.1](https://github.com/gdsfactory/gdsfactory/pull/494)

- layer_stack has a 2.5D information.
- fix xsection_planarized script
- add 2.5 info to generic.

## [5.12.0](https://github.com/gdsfactory/gdsfactory/pull/493)

- remove `gf.simulation.gtidy3d.modes.find_modes`, add cache and filepath to Waveguide
- remove many default parameters from `Waveguide`
- replace from pickle to np.savez_compressed()
- replace `from tqdm import tqdm` to `from tqdm.auto import tqdm`
- add Optional refractive_index to LayerLevel
- add Transition to docs API
- add archimedean spiral [PR](https://github.com/gdsfactory/gdsfactory/pull/492)
- add Google pydocstyle to docs/contribution.md

## [5.11.4](https://github.com/gdsfactory/gdsfactory/pull/491)

- add opacity 0.5 for dither I1
- Fix sweep_bend_loss, overlap integral code in gtidy3d.modes [PR](https://github.com/gdsfactory/gdsfactory/pull/490)
- replace Settings object in packed info by dict [PR](https://github.com/gdsfactory/gdsfactory/pull/489) fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/488)

## [5.11.3](https://github.com/gdsfactory/gdsfactory/pull/485)

- move dependencies from `pip install gdsfactory[full]`  to `pip install gdsfactory`
    - watchdog
    - qrcode
- increase test coverage
- remove `icyaml` webapp

## [5.11.2](https://github.com/gdsfactory/gdsfactory/pull/484)

- better docs
- simpler gf module namespace. unexpose some functions from module
    - port
    - klive
    - plot, quickplot, quickplot2, set_quickplot_options
    - dft
- add shear angle to Port.__str__

## [5.11.1](https://github.com/gdsfactory/gdsfactory/pull/481)

- add pytest and pytest-regressions to requirements

## [5.11.0](https://github.com/gdsfactory/gdsfactory/pull/480)

- add Pdk.warn_offgrid_ports
- move optional dependencies to `pip install gdsfactory[full]`
- move sipann dependency to `pip install gdsfactory[sipann]`
- parametric layer_stack

## [5.10.17](https://github.com/gdsfactory/gdsfactory/pull/479)

- [PR](https://github.com/gdsfactory/gdsfactory/pull/478) fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/474)
    - Use snap.snap_to_grid() to snap cross section points
    - Warning was not being raised if only one coordinate was off-grid
- [PR](https://github.com/gdsfactory/gdsfactory/pull/479) fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/476) offgrid manhattan connection gaps
- remove unused cache setting from Component.copy()
- fix phidl [issue](https://github.com/amccaugh/phidl/issues/154)
- make lytest as an optional dependency

## [5.10.16](https://github.com/gdsfactory/gdsfactory/pull/477)

- rename triangle to triangles, to avoid conflict names with triangle module [PR](https://github.com/gdsfactory/gdsfactory/pull/475)
- fix interconnect plugin notebook [PR](https://github.com/gdsfactory/gdsfactory/pull/473/files)
- add `Pdk.grid_size = 0.001` (1nm by default)
- raise warning when extruding paths with off-grid points
- raise warning when connecting components with non-manhattan (0, 90, 180, 270) orientation

## [5.10.15](https://github.com/gdsfactory/gdsfactory/pull/470)

- Update and document Interconnect plugin [PR](https://github.com/gdsfactory/gdsfactory/pull/469)

## [5.10.14](https://github.com/gdsfactory/gdsfactory/pull/468)

- simpler serialization.py
- difftest response is Yes by default when there is a GDSdiff error

## [5.10.13](https://github.com/gdsfactory/gdsfactory/pull/467)

- improve docs.
- add conda package.
- Cover any numpy numbers in serialization [PR](https://github.com/gdsfactory/gdsfactory/pull/466)
- Custom component labels in grid_with_text [PR](https://github.com/gdsfactory/gdsfactory/pull/465)

## [5.10.12](https://github.com/gdsfactory/gdsfactory/pull/463)

- speed up module gf/__init__.py thanks to scalene python profiler

## [5.10.8](https://github.com/gdsfactory/gdsfactory/pull/462)

- fix documentation (add `pip install jaxlib jax`) to `make plugins`
- fix some mypy issues

## [5.10.7](https://github.com/gdsfactory/gdsfactory/pull/460)

- [repo improvements](https://scikit-hep.org/developer/reporeview)
    - move mypy and pytest config to pyproject.toml
- rename extension_factory to extension

## [5.10.6](https://github.com/gdsfactory/gdsfactory/pull/459)

- raise ValueError if no polygons to render in 3D.
- add pad ports to functions that route to electrical pads, so new Component can still access the pad ports.
    - `gf.routing.add_electrical_pads_shortest`
    - `gf.routing.add_electrical_pads_top`
    - `gf.routing.add_electrical_pads_top_dc`
- add `gf.add_labels.add_labels_to_ports`
    - add `gf.add_labels.add_labels_to_ports_electrical`
    - add `gf.add_labels.add_labels_to_ports_optical`
    - add `gf.add_labels.add_labels_to_ports_vertical_dc` for pads
- fix colors in Component.plot()
- add `Component.plotqt()`
- add add_port_markers and read_labels_yaml to gf.read.labels

## [5.10.5](https://github.com/gdsfactory/gdsfactory/pull/457)

- quickplotter picks a random color if layer not defined in pdk.get_layer_color(). Before it was raising a ValueError.


## [5.10.4](https://github.com/gdsfactory/gdsfactory/pull/456)

- Use tidy3d.webapi.Batch instead of pool executor [PR](https://github.com/gdsfactory/gdsfactory/pull/455)
- update to latest tidy3d==1.4.1

## [5.10.3](https://github.com/gdsfactory/gdsfactory/pull/454)

- replace 'bend_euler' string with function in mzi

## [5.10.2](https://github.com/gdsfactory/gdsfactory/pull/453)

- fix tidy3d port orientation '+' or '-'

## [5.10.1](https://github.com/gdsfactory/gdsfactory/pull/452)

- works with latest simphony and Sipann

## [5.10.0](https://github.com/gdsfactory/gdsfactory/pull/449)

- rename LayerSet to LayerColors, as it is a more intuitive name. We only use this for defining 3D and 2D plot colors.
- add Pdk attributes
    - layer_stack: Optional[LayerStack] = None
    - layer_colors: Optional[LayerColors] = None
    - sparameters_path: PathType
- add Component.to_3d()
- add gf.pdk.get_layer_stack() for 3D rendering and simulation plugins
    - gf.simulation.lumerical.write_sparameters_lumerical
    - gf.simulation.gmeep.write_sparameters_meep
    - gf.simulation.tidy3d.write_sparameters
- modify Component.plot() to use colors from gf.pdk.get_layer_colors()

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
  - add layers dict to Pdk(layers=dict(LAYER)), and `pdk.get_layer`

## [5.7.1](https://github.com/gdsfactory/gdsfactory/pull/403)

- add cross_section_bot and cross_section_top to mzi, fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/402)
- add electrical ports to heater cross_sections, fixes [issue](https://github.com/gdsfactory/gdsfactory/issues/394)

## [5.7.0](https://github.com/gdsfactory/gdsfactory/pull/400)

- tidy3d mode solver accepts core_material and clad_material floats.
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

- add_fiber_single and add_fiber_array tries to add port with `vertical` prefix to the new component. It not adds the regular first port. This Keeps backwards compatibility with grating couplers that have no defined vertical ports.
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
- fix mirror issue in `gf.read.from_yaml` [PR](https://github.com/gdsfactory/gdsfactory/pull/341)

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

- remove default port width, layer and center

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

- add `has_routing_info` to CrossSection to ensure it has routing information
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

- add Component.get_netlist
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

- thread each tidy3d.write_sparameters simulation, so they run in parallel
- add tidy3d.write_sparameters_batch to run multiple sparameters simulations in parallel

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
- lumerical interface improvements (consistent with meep)
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
- placer uses Literal ('N', 'S', 'E', 'W') from gf.typings
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

- Component.show() writes component in a different tempfile every time. This avoids the `reload` question prompt from klayout.
- update klive to 0.0.7 to keep the same layers active between sessions

## 3.9.5

- imported cell names get incremented (starting on index = 1) with a `$` (based on Klayout naming convention)
- add test for flatten = True
- raise ValueError if the passed name is already on any CACHE (CACHE_IMPORTED or CACHE)
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
- add `gf.typings.ComponentSweep`
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
- ensure consistent name in YAML by hashing the dict in case no name is provided
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
- make consistent names in gf.read: `gf.read.from_phidl` `gf.read.from_picwriter` `gf.read.from_gds`

## 3.3.5

- `route_manhattan` ensures correct route connectivity
- replace `bend_factory` by `bend` to be more consistent with components
- replace `bend90_factory` by `bend90` to be more consistent with components
- replace `straight_factory` by `straight` to be more consistent with components
- replace `get_route_electrical_shortest_path` by `route_quad`
- gf.components.array raises error if columns > 1 and xspacing = 0
- gf.components.array raises error if rows > 1 and yspacing = 0
- simplify `gf.components.rectangle` definition, by default it gets 4 ports
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
- rename `gf.lys` to `gf.layers.LAYER_COLORS` to be consistent

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
- added filecmp for testing components width difftest, only does XOR if files are different. This speeds the check for larger files.

## 3.0.3

- change port naming convention from WNES to o1, o2, o3 for optical, and e1, e2, e3, e4 for electrical
- add Component.auto_rename_ports()
- add `ports_layer` property to Component and ComponentReference to get a map
- `Component.show()` `show_ports` and `show_subports` show a container (do not modify original component)
- add port_types to cross_section
- rename straight_horizontal_top to straight_x_top

## 3.0.2

- add straight_rib, straight_heater_metal and straight_heater_doped
- `xs2 = partial(cross_section)` does not require defining `xs2.__name__`
- replace gf.extend[.] with gf.components.extension.
- Component.show() uses `add_pins_triangle` as default to show port orientation
- add gf.containers.bend_port
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
- add pydantic.validate method to Port

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
- Load a LayerColors object from a Klayout lyp file
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
- replace some `c.show(show_ports=True)` by a simpler `c.show()`

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
    - you can define a route range (left,E:0:3 : right,W:0:3)
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

- added grating coupler decorator to assert polarization and wavelength
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
- adapted pp.pack from phidl as an easier way to pack masks
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
- new coupler with less snapping errors
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
