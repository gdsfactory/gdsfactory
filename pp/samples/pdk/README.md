# How to build/import an existing PDK?

Different foundries use different layers number for each process step.

Some require multiple layers to define just a simple waveguide.

You can easily create a PDK by modifying the basic components of gdsfactory to adapt to your target foundry.

- waveguide
- bend_circular
- add_gc (to connect to grating couplers)

In this sample import the [SiEPIC PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK)



# How to add regression tests?

You can create a python package with the contents of this folder, then run `write_tests` to write your regression tests.
