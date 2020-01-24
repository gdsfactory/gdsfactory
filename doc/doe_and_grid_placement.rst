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
 

This is the content of the YAML file called `placer_example.yml`
It specifies a component grid and its DOEs.

.. code-block:: yaml

    placer:
        x_spacing: 1000 # x spacing origin to origin between the components
        y_spacing: 500 #  y spacing origin to origin between the components
        x_start: 500
        y_start: 100

    doe1:
        doe_name: doe1
        component: mmi1x2
        length_mmi: [11, 12]
        width_mmi: [3.6, 7.8]
        do_permutation: False
        
        
    doe2:
        doe_name: doe2
        component: mmi1x2
        length_mmi: [13, 14, 15]
        width_mmi: [3.6, 7.8, 9.4]
        do_permutation: False
        
    doe3:
        doe_name: doe3
        component: mzi2x2
        L0: [60, 80, 100]
        
        
    doe4:
        doe_name: doe4
        component: mzi2x2
        L0: [60, 80, 100]
        gap: [0.23, 0.234, 0.24]
        do_permutation: True
        
    placement:
        A1-2: doe1
        D-F4: doe2
        A3-5: doe3
        E-G5-7: doe4
        
Define DOE
--------------------

In the previous section, we saw how to create a DOE based on standard component factories available in the pdk.
You may also want to create your own DOE based on arbitrary devices.
To do that you need to:

 - Create in your workspace a file containing your component factory 
 - Add your factory to the dictionnary of component factories
 - Define your DOE based on this factory in the YAML file
 

In this example, we want to define a specific type of waveguide cutback

Define component factory
------------------------------

.. code-block:: python

    # Importing what I need from the pdk to make my new component factory
    import pp
    from pp.components.delay_snake import delay_snake
    from pp.components.waveguide import waveguide
    from pp.components.bend_circular import bend_circular
    
    # Importing the I/O connector function
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


We now need to make sure that the `component_grid_from_yaml` placer knows about this new factory.

.. code-block:: python
    
    import pp
    
    # Import the placer
    from pp.placer import component_grid_from_yaml
    
    # Import our custom factory
    from wg_te_cutback import wg_te_cutback
    
    # Import the dictionnary of factories
    from pp.components import component_type2factory

    # Add the custom DOE to the dictionnary of factories
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

