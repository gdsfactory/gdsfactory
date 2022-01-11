# Contributing

gdsfactory is an open source project that welcomes contributions from any users.

How can you contribute?


You can fork the repo, work on a feature, and then create a merge request. As long as the tests pass on [GitHub Actions](https://github.com/gdsfactory/gdsfactory/actions) it is likely that your new improvement will be merged soon and included in the next release.

[pytest](https://docs.pytest.org/en/stable/index.html) run tests and `flake8` checks syntax errors.
To contribute to the project you will need to install it from GitHub and install it with `make install`. After your improvements `pytest` and `flake8` must be passing.
To help you with code quality checks `make install` will install some pre-commit hooks for you to ensure code is up to standards before you even commit with GIT.

What are the easiest contributions to make?

You can take a look at the [open issues](https://github.com/gdsfactory/gdsfactory/issues) or you can also share some of your work:

- Any improvements you make (documentation, tutorials, docstrings, tests, type checkers, code quality ...
- Your layout/verification functions that you wrote recently
- A cool structure that you made found on a paper. It will help your work get citations as other people build upon it.

The workflow is:

- Fork the repo
- `git clone` it into your computer and install it
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

As you create your component functions (known as factories because they return objects). You can also store them in a dict so you can easily access their names and their functions.

I recommend that you also write tests for the all those new functions that you write.

See for example the tests in the [ubc PDK](https://github.com/gdsfactory/ubc)

Pytest-regressions automatically creates the CSV and YAML files for you, as well `gdsfactory.gdsdiff` will store the reference GDS in ref_layouts

gdsfactory is **not** backwards compatible, which means that the package will keep improving and evolving.

1. To make your work stable you should install a specific version and [pin the version](https://martin-thoma.com/python-requirements/) in your `requirements.txt` as `gdsfactory==3.9.26` replacing `3.9.26` by whatever version you end up using.
2. Before you upgrade gdsfactory make sure you write and run regression tests on your work to check that things behave as expected


## gdsdiff

You can use the command line `gf gds diff gds1.gds gds2.gds` to overlay `gds1.gds` and `gds2.gds` files and show them in klayout.

For example, if you changed the mmi1x2 and made it 5um longer by mistake, you could `gf gds diff ref_layouts/mmi1x2.gds run_layouts/mmi1x2.gds` and see the GDS differences in Klayout.

![](images/git_diff_gds_ex2.png)
