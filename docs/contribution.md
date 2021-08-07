# Contributing

gf.is an open source project that is open to any contributions from any of its users.

How can you contribute to gf.

You can fork the repo, work on a feature, and then create a merge request. As long as the tests pass on [GitHub Actions](https://github.com/gf.gf.actions) it is likely that your new improvement will be merged soon and included in the next gf.release.

gf.run tests with [pytest](https://docs.pytest.org/en/stable/index.html) and checks syntax errors with flake8.
To contribute to the project you will need to install it from GitHub and install it with `make install`. After your improvements `pytest` and `flake8` must be passing.
To help you with code quality checks `make install` will install some pre-commit hooks for you to ensure code is up to standards before you even commit with GIT.


## Tests


`pytest` runs 3 types of tests:


You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function in the `gf. package that starts with `test_`. You can assert the number of polygons, the name, the length of a route or whatever you want.
- regressions tests: avoids unwanted regressions by storing Components port locations in CSV and metadata in YAML files. You can force to regenerate the reference files running `make test-force` from the repo root directory.
    - `gf.tests/test_containers.py` stores container settings in YAML and port locations in a CSV file
    - `gf.tests/components/test_components.py` stores all the component settings in YAML
    - `gf.tests/components/test_ports.py` stores all port locations in a CSV file
    - `gf.tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist.
        - converts the routed PIC into YAML and build back into the same PIC from its YAML definition
    - lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
        * when running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences.
        * you can check out any changes in the library with `pf diff ref_layouts/bbox.gds run_layouts/bbox.gds`
        * it will also store all diferences in `diff_layouts` and you can combine and show them in klayout with `make diff`



## Testing your own component factories

As you create your component functions (known as factories because they return objects). You can also store them in a dict so you can easily access their names and their functions.

I recommend that you also write tests for the all those new functions that you write.

See for example the tests in the [ubc PDK](https://github.com/gf.ubc)

Pytest-regressions automatically creates the CSV and YAML files for you, as well gf.difftest will store the reference GDS in ref_layouts


## gdsdiff

You can use the command line `pf diff gds1.gds gds2.gds` to overlay `gds1.gds` and `gds2.gds` files and show them in klayout.

For example, if you changed the mmi1x2 and made it 5um longer by mistake, you could `pf diff ref_layouts/mmi1x2.gds run_layouts/mmi1x2.gds` and see the GDS differences in Klayout.

![](images/git_diff_gds_ex2.png)
