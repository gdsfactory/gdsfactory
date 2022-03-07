# TODO

- enable rich output?
- more explicit Sparameter data format. Considers TE/TM modes.
- add DVC for version control data on GCS, instead of relying gdslib second repo.
- replace circular fiber marker by square
- fix FIXMEs
- klayout klive refresh does not maintain the position of the view any more
- better netlist extraction
- interface with Siepic tools? add ports as FlexPath
- corners don't auto-taper for RF circuits

## Plugins

- tidy3d: compute bend mode missmatch
- sax. Backend that does not require JAX (for windows users)
- simphony, update to latest version?
- sipann

Modes

- include mode overlap examples
- include modesolverpy solver

Meep

- Add some heuristics for optimizing number of codes in MPI. How many cores to use?

Lumerical:

- batch mode or run with more cores
- eme
- mode-solver
- interconnect

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
