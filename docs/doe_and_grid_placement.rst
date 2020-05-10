DOE: Design of Experiment
===================================

A design Of Experiment combines several components to extract a component model or some other useful data.

You can define large parts of the layout in YAML:

A "placer" YAML file has two reserved key words:

- `placer` : contains the information about the origin of the grid and component spacing
- `placement` : describes how and where DOEs are placed within the grid

The grid is indexed from A to Z along the X axis and uses integers for the y axis.

All the other sections in the YAML file are assumed to be DOE sections.
A DOE contains:

- `doe_name`: the name of the DOE (one DOE can be defined using several sections if needed, the DOE sections must have a unique name. But they will point to the same `doe_name`)
- `component`: the name of the component factory to be invoked. This component has to be present within `pp.components.__init__.__all__`
- `do_permutation`: (optional), True by default. If True, performs all the permutations of the given parameters
- Any other field corresponds to a parameter of the component factory


This is the content of the YAML file called `does.yml`
It specifies component DOEs, and where they should be placed after building them.

.. code-block:: yaml

    mask:
      width: 10000
      height: 10000
      name: mask2

    mmi_width:
      doe_name: mmi_width
      component: mmi1x2
      settings:
        width_mmi: [4.5, 5.6]
        length_mmi: 10
      placer:
        type: pack_row
        x0: 0 # Absolute coordinate placing
        y0: 0 # Absolute coordinate placing
        align_x: W # x origin is west
        align_y: S # y origin is south

    mmi_width_length:
      doe_name: mmi_width_length
      component: mmi1x2
      do_permutation: False
      settings:
        length_mmi: [11, 12]
        width_mmi: [3.6, 7.8]

      placer:
        type: pack_row
        next_to: mmi_width
        x0: W # x0 is the west of the DOE specified in next_to
        y0: S # y0 is the south of the DOE specified in next_to
        align_x: W # x origin is west of current component
        align_y: N # y origin is south of current component
        inter_margin_y: 200 # y margin between this DOE and the one used for relative placement
        margin_x: 50. # x margin between the components within this DOE
        margin_y: 20. # y margin between the components within this DOE


Define DOE
--------------------

In the previous section, we saw how to create a DOE based on standard component factories available.
You may also want to create your own DOE based on arbitrary devices.
To do that you need to:

 - Create in your workspace a file containing your component factory
 - Add your factory to the dictionnary of component factories
 - Define your DOE based on this factory in the YAML file


In this example, we want to define a specific type of waveguide cutback

Define component factory
------------------------------

.. code-block:: python

    import pp
    from pp.components.delay_snake import delay_snake
    from pp.components.waveguide import waveguide
    from pp.components.bend_circular import bend_circular
    from pp.routing.connect_component import add_io_optical

    @pp.autoname
    def wg_te_cutback(L=2000.0, wg_width=0.5, bend_radius=10.0):
        """
        All the parameters from this function will be accessible for defining the DOE
            L: total length of the cutback
            wg_width: waveguide width
            bend_radius: bend radius
        """

        # Defining my component
        _delay = delay_snake(
            total_length=L,
            L0=1.0,
            n=5,
            taper=None,
            bend_factory=bend_circular,
            bend_radius=bend_radius,
            wg_width=wg_width,
            straight_factory=waveguide,
        )
        # Adding optical I/O for test and measurement
        component = add_io_optical(_delay)

        # The factory should return the component
        return component


We now need to make sure that the `does.yaml` placer knows about this new factory.

.. code-block:: python

    import pp
    from pp.placer import component_grid_from_yaml
    from wg_te_cutback import wg_te_cutback
    from pp.components import component_type2factory

    # Add the custom DOE to the dictionary of factories
    component_type2factory["wg_te_cutback"] = wg_te_cutback

    def main():
        # We will show this YAML file in the next section
        filepath = "mask_definition.yml"

        # Generate the cell following the instructions from  `"mask_definition.yml"`
        top_level = component_grid_from_yaml("top_level", filepath)

        # Save and show the GDS
        pp.show(top_level)


    if __name__ == "__main__":
        main()

This examples generates a full cell with all the DOEs defined in `mask_definition.yml`


Define DOE in a YAML file
-------------------------------------

.. code-block:: yaml

   wg_loss_te_1000:
     component: spiral_te
     settings:
       wg_width: [1.]
       length: [2, 4, 6]
     test: passive_optical_te_coarse
     analysis: spiral_loss_vs_length

   coupler_500_224:
     component: coupler_te
     settings:
       wg_width: [0.5]
       gap: [0.224]
       length: [0, 18.24, 36.48, 54.72, 72.96, 91.2]
     test: passive_optical_te_coarse
     analysis: coupling_vs_length
