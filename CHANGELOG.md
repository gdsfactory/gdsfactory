# dev

- add_padding works over the same component
- import_gds can snap points to a design grid


# 1.1.5 2020-03-17

- added pre-commit hook for code consitency
- allows a list of cladding layers
- all layers are defined as tuples using pp.LAYER.WG, pp.LAYER.WGCLAD


# 1.1.4 2020-02-27

- bug fixes
- new coupler with less snaping errors
- adding Klayout generic DRC rule deck

# 1.1.1 2020-01-27

- first public release

# 1.0.2 2019-12-20

- test components using gdshash
- new CLI commands for `pf`
    - pf library lock
    - pf library pull

# 1.0.1 2019-12-01

- autoplacer and yaml placer
- mask_merge functions (merge metadata, test protocols)
- added mask samples
- all the mask can be build now from a config.yml in the current directory using `pf mask write`

# 1.0.0 2019-11-24

- first relase
