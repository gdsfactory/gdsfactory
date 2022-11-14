# TODO

- native oasis compatibility
- flake8-bugbear
- more explicit Sparameter data format. Consider TE/TM modes.
- add examples for TE to TM conversion

## Plugins

- more routers
- tcad
- litho
- snakemake: for cloud workflows
- Lumerical:
    - batch mode or run with more cores
    - eme
    - mode

## Someday

- add xdoctest
- type checker passing (mypy, pyre, pytype)
- klayout placer (north, west) does not work well with rotations
- enable rich output?
- add cloud bucket or DVC for version control data on GCS, instead of relying gdslib second repo.

## Maybe not a good idea?

- remove kwargs from most components as a way to customize cross_sections to get more clear error messages
- add non-manhattan routing
  - enable routing with 180euler and Sbends
  - electrical routing with 45
- cell decorator includes hashes all the source code from a function to ensure no name conflicts happen when merging old and future cells. This was quite slow. Too slow
