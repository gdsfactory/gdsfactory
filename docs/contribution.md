# Contributing

gdsfactory is an open source project that welcomes your contributions. How can you contribute?
You can fork the repo, work on a feature, and then create a Pull Request to merge your feature into the `main` branch.
This will benefit the project community and make you famous :).

How can you help? Take a look at the [open issues](https://github.com/gdsfactory/gdsfactory/issues) or add something you need to gdsfactory:

- Any improvements you make (documentation, tutorials or code). Just find a typo and submit a PR!
- Your design/verification/validation functions that you wrote, or rewrite some part tutorial that is not clear.
- A new device that you found on a paper so you can use it on your next tapeout. It helps get citations as other people start using or building on top of the work from the paper.

The workflow is:

- Fork the repo. This creates a copy into your GitHub account. `git clone` it into your computer and install it (`./install.bat` for Windows and `make install` for MacOs and Linux).
- `git add`, `git commit`, `git push` your work as many times as needed. Make sure [GitHub Actions](https://github.com/gdsfactory/gdsfactory/actions) pass so it all keeps working correctly.
- open a Pull request (PR) to merge your improvements to the main repository.

## Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). You can take a look at some existing gdsfactory code.
- Make sure tests pass on GitHub.
- Install pre-commit to get the pre-commit checks passing (autoformat the code, run linter ...). You can also install pre-commit:

```
cd gdsfactory
pip install -e . pre-commit
pre-commit install
```

Pre-commit makes sure the code is formatted correctly, runs linter (syntax check), checks docstrings ...
If you forgot to `pre-commit install` you can fix pre-commit issues by running

```
pre-commit run --all-files
```

## Tests

`pytest` runs 3 types of tests:

You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function that starts with `test_`. You can assert the number of polygons, the name, the length of a route or whatever you want.
- regressions tests: avoids unwanted regressions by storing Components port locations in CSV and metadata in YAML files. You can force to regenerate the reference files running `pytest --force-regen -s` from the repo root directory.
  - `tests/test_containers.py` stores container settings in YAML and port locations in a CSV file
  - `tests/components/test_components.py` stores all the component settings in YAML
  - `tests/components/test_ports.py` stores all port locations in a CSV file
  - `tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist.
    - converts the routed PIC into YAML and build back into the same PIC from its YAML definition
  - lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
    - when running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences.
    - you can check out any changes in your library with `gf gds diff ref_layouts/bbox.gds run_layouts/bbox.gds`

## Acks

Gdsfactory leverages KLayout and gdstk python APIs.

What nice things are inspired by gdstk and phidl?

- functional programming follows UNIX philosophy.
- Create and modify Components using simple functions that can build complexity one level at a time.
- Define paths and cross-sections and extrude them into Components
- Define ports, to connect components. Ports in phidl have name, position, width and orientation (in degrees)
  - gdsfactory ports have layer, `port_type` (`optical`, `electrical`, `vertical_te`, `vertical_tm` ...) and `cross_section`
  - gdsfactory adds renaming ports functions for ports (clockwise, counter-clockwise ...)

What nice things come from KLayout?

- GDS viewer. gdsfactory can send GDS files directly to KLayout, you just need to have KLayout open
- layer colormaps for showing in KLayout, matplotlib, trimesh (using the same colors)
- fast boolean xor to avoid geometric regressions on Components geometry. Klayout booleans are more robust.
- basic DRC checks

What functionality does gdsfactory provide you on top gdstk/KLayout?

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
- [open source best practices](https://opensource.guide/best-practices/)
