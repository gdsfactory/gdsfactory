# Tests


gdsfactory run tests with [pytest](https://docs.pytest.org/en/stable/index.html).
This will run 3 types of tests:


- pytest will test any function in the `pp` package that starts with `test_`
    - you can assert the number of polygons, the name, the length of a route or whatever you want.
- regressions tests: avoids unwanted regressions by storing Components ports position and metadata in YAML files. You can force to regenerate those files running `make test-force` from the repo root directory.
    - `pp/tests/test_containers.py` stores container function settings in YAML and port locations in a CSV file
    - `pp/tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist.
        - converts the routed PIC into YAML and build back into the same PIC from its YAML definition
    - `pp/tests/test_components.py` stores all the component settings in YAML and port locations in CSV files.
        * lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
            + when running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences.
            + you can check out any changes in the library with `pf diff ref_layouts/bbox.gds run_layouts/bbox.gds`


## Testing your own component factories

As you create your component functions (known as factories because they return objects). You can also store them in a dict so you can easily access their names and their functions.

I recommend that you also write tests for the all those new functions that you write.

See for example the tests in the [ubc PDK](https://github.com/gdsfactory/ubc)

Pytest-regressions automatically creates the CSV and YAML files for you, as well as the GDS in ref_layouts
