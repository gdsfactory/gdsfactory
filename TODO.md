# TODO

- add electrical router for DC where orientation is None
- enable rich output?
- more explicit Sparameter data format. Consider TE/TM modes.
- replace circular fiber marker by square
- better netlist extraction
- add DVC for version control data on GCS, instead of relying gdslib second repo.

## Plugins

- snakemake: for cloud workflows
- tidy3d: compute bend mode missmatch
- simphony, update to latest version?
- sipann, equivalent for SAX?
- Modes: include mode overlap examples, tidy3d and modesolverpy solver
- Lumerical:
    - batch mode or run with more cores
    - eme
    - mode-solver

## Someday

- add xdoctest
- type checker passing (mypy, pyre, pytype)
- klayout placer (north, west) does not work well with rotations

## Maybe not a good idea?

- remove kwargs from most components as a way to customize cross_sections to get more clear error messages
- add non-manhattan routing
  - enable routing with 180euler and Sbends
  - electrical routing with 45
- cell decorator includes hashes all the source code from a function to ensure no name conflicts happen when merging old and future cells. This was quite slow. Too slow
