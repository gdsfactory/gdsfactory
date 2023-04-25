# Contributing

gdsfactory is an open source project that welcomes your contributions. How can you contribute?
You can fork the repo, work on a feature, and then create a Pull Request to merge your feature into the `main` branch.
This will benefit the project community and make you famous :).

How can you help? Take a look at the [open issues](https://github.com/gdsfactory/gdsfactory/issues) or add something you need to gdsfactory:

- Documentation, tutorials or code improvements. Just find a typo and submit a PR!
- Design/verification/validation improvements.
- A new device that you found on a paper so you can use it on your next tapeout. It helps get citations as other people start using or building on top of the work from the paper.

The workflow is:

- Fork the repo. This creates a copy into your GitHub account. `git clone` it into your computer and install it (`./install.bat` for Windows and `make install` for MacOs and Linux).
- `git add`, `git commit`, `git push` your work as many times as needed. Make sure [GitHub Actions](https://github.com/gdsfactory/gdsfactory/actions) pass so it all keeps working correctly.
- open a Pull request (PR) to merge your improvements to the main repository.

## Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and take a look at existing gdsfactory code.
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


If test failed because you modified the geometry you can regenerate the regression tests with:

```
pytest --force-regen -s
```
