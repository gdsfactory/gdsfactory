# TODO

- serializer
- fix FIXMEs
- add example on how to customize visualization of a component
- remove kwargs from most components as a way to customize cross_sections to get more clear error messages
- klayout placer north, west does not work well with rotations

## Plugins

Modes

- include mode overlap examples

Meep

- Add some heuristics for optimizing number of codes in MPI

Lumerical:

- batch mode or run with more cores
- eme
- mode-solver

## Someday

- add xdoctest
- type checker passing (mypy, pyre, pytype)
- cell decorator includes hashes all the source code from a function to ensure no name conflicts happen when merging old and future cells. This was quite slow.
- holoviews layout plot, similar to dphox

- add non-manhattan routing
  - enable routing with 180euler and Sbends
  - electrical routing with 45
