# Contributing

gdsfactory is an open source project that welcomes contributions from any users.

How can you contribute?

You can fork the repo, work on a feature, and then create a Pull Request asking permission to merge your feature into the `master` branch. As long as the tests pass on [GitHub Actions](https://github.com/gdsfactory/gdsfactory/actions) it is likely that your improvement will be included in the next release and benefit the community.

After your improvements the tests with `pytest` and syntax checker `flake8` must be passing.
To help you with code quality checks we install pre-commit hooks to maintain good code quality.

What are the easiest contributions to make?

You can take a look at the [open issues](https://github.com/gdsfactory/gdsfactory/issues) or you can also share some of your work:

- Any improvements you make (documentation, tutorials or code)
- Your layout/verification functions that you wrote recently
- A new device that you found on a paper. It will help your work get citations as other people build upon it.

The workflow is:

- Fork the repo
- `git clone` it into your computer and install it (`./install.bat` for Windows and `make install` for MacOs and Linux)
- `git add`,`git commit`, `git push` your work as many times as needed (make sure tests are passing)
- open a Pull request (PR)

## Tests

`pytest` runs 3 types of tests:

You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function that starts with `test_`. You can assert the number of polygons, the name, the length of a route or whatever you want.
- regressions tests: avoids unwanted regressions by storing Components port locations in CSV and metadata in YAML files. You can force to regenerate the reference files running `make test-force` from the repo root directory.
  - `tests/test_containers.py` stores container settings in YAML and port locations in a CSV file
  - `tests/components/test_components.py` stores all the component settings in YAML
  - `tests/components/test_ports.py` stores all port locations in a CSV file
  - `tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist.
    - converts the routed PIC into YAML and build back into the same PIC from its YAML definition
  - lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
    - when running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences.
    - you can check out any changes in your library with `gf gds diff ref_layouts/bbox.gds run_layouts/bbox.gds`
    - it will also store all differences in `diff_layouts` and you can combine and show them in klayout with `make diff`

## Testing your own component factories

As you create your component functions (known as factories because they return objects) I recommend that you also write tests for the all those new functions that you write. See for example the tests in the [ubc PDK](https://github.com/gdsfactory/ubc)

Pytest-regressions automatically creates the CSV and YAML files for you, as well `gdsfactory.gdsdiff` will store the reference GDS in ref_layouts

gdsfactory is **not** backwards compatible, which means that the package will keep improving and evolving.

1. To make your work stable you should install a specific version and [pin the version](https://martin-thoma.com/python-requirements/) in your `requirements.txt` as `gdsfactory==5.9.0` replacing `5.9.0` by whatever version you end up using.
2. Before you upgrade gdsfactory make sure you write and run regression tests on your work to check that things behave as expected


## gdsdiff

You can use the command line `gf gds diff gds1.gds gds2.gds` to overlay `gds1.gds` and `gds2.gds` files and show them in klayout.

For example, if you changed the mmi1x2 and made it 5um longer by mistake, you could `gf gds diff ref_layouts/mmi1x2.gds run_layouts/mmi1x2.gds` and see the GDS differences in Klayout.

![](images/git_diff_gds_ex2.png)


## Why does gdsfactory exists?

For Photonics IC layout I used [IPKISS](https://github.com/jtambasco/ipkiss) for 6 years. IPKISS is slow with big layouts, so in 2019 I tried all the commercial (Luceda, Cadence, Synopsys) and open source EDA tools (phidl, gdspy, picwriter, klayout-zero-pdk, nazca) looking for a fast and easy to use workflow.

The metrics for the benchmark were:

0. Fast
1. Easy to use and interface with other tools
2. Maintained / Documented / Popular

PHIDL won in speed, readability and easy of use. It is written on top of gdspy (which came second), so you can still leverage all the work from the gdspy community. Gdsfactory also leverages klayout and gdspy python APIs.

Gdsfactory leverages klayout and gdspy python APIs.

What nice things come from phidl?

- functional programming that follow UNIX philosophy
- nice API to create and modify Components
- Easy definition of paths, cross-sections and extrude them into Components
- Easy definition of ports, to connect components. Ports in phidl have name, position, width and orientation (in degrees)
  - gdsfactory expands phidl ports with layer, port_type (optical, electrical, vertical_te, vertical_tm ...) and cross_section
  - gdsfactory adds renaming ports functions (clockwise, counter_clockwise ...)

What nice things come from klayout?

- GDS viewer. gdsfactory can send GDS files directly to klayout, you just need to have klayout open
- layer colormaps for showing in klayout, matplotlib, trimesh (using the same colors)
- fast boolean xor to avoid geometric regressions on Components geometry. Klayout booleans are faster than gdspy ones
- basic DRC checks

What functionality does gdsfactory provide you on top phidl/gdspy/klayout?

- `@cell decorator` for decorating functions that create components
  - autonames Components with a unique name that depends on the input parameters
  - avoids duplicated names and faster runtime implementing a cache. If you try to call the same component function with the same parameters, you get the component directly from the cache.
  - automatically adds cell parameters into a `component.info` (`full`, `default`, `changed`) as well as any other `info` metadata (`polarization`, `wavelength`, `test_protocol`, `simulation_settings` ...)
  - writes component metadata in YAML including port information (name, position, width, orientation, type, layer)
- routing functions where the routes are composed of configurable bends and straight sections (for circuit simulations you want to maintain the route bends and straight settings)
  - `get_route`: for single routes between component ports
  - `get_route_from_steps`: for single routes between ports where we define the steps or bends
  - `get_bundle`: for bundles of routes (river routing)
  - `get_bundle_path_length_match`: for routes that need to keep the same path length
  - `get_route(auto_widen=True)`: for routes that expand to wider waveguides to reduce loss and phase errors
  - `get_route(impossible route)`: for impossible routes it warns you and returns a FlexPath on an error layer to clearly show you the impossible route
- testing framework to avoid unwanted regressions
  - checks geometric GDS changes by making a boolean difference between GDS cells
  - checks metadata changes, including port location and component settings
- large library of photonics and electrical components that you can easily customize to your technology
- read components from GDS, numpy, YAML
- export components to GDS, YAML or 3D (trimesh, STL ...)
- export netlist in YAML format
- plugins to compute Sparameters using for example Lumerical, meep or tidy3d



gdsfactory is written in python and requires some basic knowledge of python. If you are new to python you can find many free online resources to learn:

- [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html)
- [youTube videos](https://www.youtube.com/c/anthonywritescode)
- courses
    - [scientific computing](https://nbviewer.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-0-Scientific-Computing-with-Python.ipynb)
    - [numerical python](http://jrjohansson.github.io/numericalpython.html)
    - [python](https://dabeaz-course.github.io/practical-python/Notes/01_Introduction/01_Python.html)
