# Tests


gdsfactory run tests with [pytest](https://docs.pytest.org/en/stable/index.html).
This will run 3 types of tests:


- pytest will test any function in the `pp` package that starts with `test_`
- lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
    - you can check out any changes in the library with `pf diff ref_layouts/bbox.gds run_layouts/bbox.gds`
- regressions tests: avoids unwanted regressions by storing Components ports position and metadata in YAML files. You can force to regenerate those files running `make test-force` from the repo root directory.
    - `pp/test_containers.py` stores container function settings in YAML and port locations in a CSV file
    - `pp/components/test_components.py` stores all the component settings in YAML
    - `pp/components/test_ports.py` stores all port locations in a CSV file


I reccommend that you also write tests for the new cells that you write. See for example the tests in the [ubc PDK](https://github.com/gdsfactory/ubc)
