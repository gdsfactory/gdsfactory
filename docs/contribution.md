# Contributing

We welcome your skills and enthusiasm at the gdsfactory project!. There are numerous opportunities to contribute beyond writing code.
All contributions, including bug reports, bug fixes, documentation improvements, enhancement suggestions, and other ideas are welcome.

If you have any questions on the process or how to fix something feel free to ask us!
The recommended place to ask a question is on [GitHub Discussions](https://github.com/gdsfactory/xarray/discussions), but we also have a [gitter matrix channel](https://matrix.to/#/#gdsfactory-dev_community:gitter.im) that you can use with any matrix client (such as [element](https://element.io/download)) and a [mailing list](https://groups.google.com/g/gdsfactory)

## Where to start?

You can fork the repo, work on a feature, and then create a Pull Request to merge your feature into the `main` branch.
This will benefit other project community members and make you famous :).

Take a look at the [open issues](https://github.com/gdsfactory/gdsfactory/issues) to find issues that interest you. Some issues are particularly suited for new contributors by the [good first issue label](https://github.com/gdsfactory/gdsfactory/labels/good first issue) where you could start out. These are well documented issues, that do not require a deep understanding of the internals of gdsfactory.

Here are some other ideas for possible contributions:

- Documentation, tutorials or code improvements. Just find a typo and submit a PR!
- Design/verification/validation improvements.
- A new device that you found on a paper so you can use it on your next tapeout. It helps get citations as other people start using or building on top of the work from the paper.

The workflow is:

- Fork the repo. This creates a copy into your GitHub account namespace. `git clone` it into your computer and install it.
- `git add`, `git commit`, `git push` your work as many times as needed. Make sure [GitHub Actions](https://github.com/gdsfactory/gdsfactory/actions) pass so it all keeps working correctly.
- open a Pull request (PR) to merge your improvements to the main repository.

![git flow](https://i.imgur.com/kNc40fI.png)

## Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and take a look at existing gdsfactory code.
- Make sure tests pass on GitHub.
- Install pre-commit to get the pre-commit checks passing (autoformat the code, run linter ...).

```
cd gdsfactory
pip install -e . pre-commit
pre-commit install
```

Pre-commit makes sure your code is formatted following black and checks syntax.
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

## Code of Conduct

This project is a community effort, and everyone is welcome to contribute. Everyone within the community is expected to abide by our [code of conduct](https://github.com/gdsfactory/gdsfactory/blob/main/docs/code_of_conduct.md)
