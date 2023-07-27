from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Lithography:
    """Simulates lithography by generating a logical on-wafer mask from many layers to be used in processing operations.

        (1) First, a mask is created from the current layer and argument layers:

        if no argument_layers are supplied to any of the boolean arguments, behaviour should be:

                this_layer                           mask opening
            <---------------->                    <---------------->
                                         ----->
            |________________|                    |________________|

            0       1        2         3          0       1        2         3
    w

        if argument_layers is provided to layers_or, for those layers:

                this_layer                               mask opening
            <---------------->                   <--------------------------->
                                         ----->
            |________________|                    |_______|________|_________|
                    |__________________|
            0       1        2         3          0       1        2         3
                    <------------------>
                       argument_layers


        if argument_layers is provided to layers_and, for those layers:

                this_layer                               mask opening
            <---------------->                            <--------->
                                         ----->
            |________________|                    |       |_________|        |
                    |__________________|
            0       1        2         3          0       1        2         3
                    <------------------>
                       argument_layers


        if argument_layers is provided to layers_xor, for those layers:

                this_layer                               mask opening
            <---------------->                   <-------->        <--------->
                                         ----->
            |________________|                    |_______|        |_________|
                    |__________________|
            0       1        2         3          0       1        2         3
                    <------------------>
                       argument_layers


        (2) Convert the logical mask into a wafer mask, opening up parts of the wafer for processing:

        if positive_tone:

               mask opening             wafer mask opened
                <------>                   <------>
                                       _____      _____
                                       |   |      |   |
                            ----->     |   |      |   | mask_thickness
            ________________           |___|______|___|


        else (negative tone):

              mask opening            wafer mask NOT opened
                <------>                   <------>
                                           ________
                                           |      |
                            ----->         |      | mask_thickness
            ________________           ____|______|____

        Args:
            layers_union (List[Layers]): other layers to use to form a mask (see diagram)
            layers_diff (List[Layers]): other layers to use to form a mask (see diagram)
            layers_intersect (List[Layers]): other layers to use to form a mask (see diagram)
            positive_tone (bool): whether to invert the resulting mask or not
            resist_thickness (float): resist mask thickness, used in some simulators
    """

    layers_or: List
    layers_and: List
    layers_xor: List
    positive_tone: bool
    resist_thickness: Optional[float]


@dataclass
class Deposit(Lithography):
    """Simulates masking + addition of material + liftoff.

    wafer mask opened           wafer mask opened
        <------>                   <------>
                                   ________
                                   |      |
                      ----->       | mat  | thickness
    ________________           ____|______|____


    Args:
        material (str): material tag
        thickness (float): thickness to add
        use_mask (bool): whether to use the mask, or operate on the entire wafer

    TODO (long term): Physical model and simulator for deposition
    """

    material: str
    thickness: float
    positive_tone = True
    use_mask: bool = True
    simulate: bool = False


@dataclass
class Etch(Lithography):
    """Simulates masking + removal of material + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________           _____      _____
    |              |               |      |
    |     mat      |    -----> mat | open | mat  depth
    |______________|           ____|______|____


    Args:
        depth (float): thickness to remove
        use_mask (bool): whether to use the mask, or operate on the entire wafer

    TODO (short term): add profile (sidewalls, etc.), add material selectivity
    TODO (long term): Physical model and simulator for etching
    """

    depth: float
    positive_tone = True
    use_mask = True
    simulate: bool = False


@dataclass
class ImplantProfile(Lithography):
    """Simulates masking + ion implantation + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________          __________________
    |              |          |                |
    |              |  ----->  |    ------- <---- range (max dist.)
    |______________|          |________________|

    Args:
        element (str): element tag
        simulate (bool): whether to use a provided profile (False), or calculate a profile from parameters (True)
        range (float): of the ions
        energy (float): thickness to remove
        resist_thickness (float): to account for shadowing in simulations
        use_mask (bool): whether to use the mask, or operate on the entire wafer

    TODO (long term): Physical model and simulator for implantation
    """

    energy: str
    vertical_straggle: float
    lateral_straggle: float
    simulate: bool = False
    profile: str = "Gaussian"
    use_mask = True
    simulate: bool = False


@dataclass
class ImplantPhysical(Lithography):
    """Simulates masking + ion implantation + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________          __________________
    |              |          |                |
    |              |  ----->  |    ------- <---- range (max dist.)
    |______________|          |________________|

    Args:
        element (str): element tag
        simulate (bool): whether to use a provided profile (False), or calculate a profile from parameters (True)
        range (float): of the ions
        energy (float): thickness to remove
        resist_thickness (float): to account for shadowing in simulations
        use_mask (bool): whether to use the mask, or operate on the entire wafer

    TODO (long term): Physical model and simulator for implantation
    """

    energy: str
    vertical_straggle: float
    lateral_straggle: float
    simulate: bool = False
    profile: str = "Gaussian"
    use_mask = True
    simulate: bool = False


@dataclass
class Anneal:
    """Simulates thermal diffusion of impurities and healing of defects. Does not use the masking.

    Args:
        time (float)
        temperature (float): temperature

    TODO (long term): heating/cooling time profiles
    TODO (long term): Physical model and simulator for diffusion
    """

    time: float
    temperature: float
    simulate: bool = False


@dataclass
class Planarize:
    """Simulates chip planarization, removing all material down to the smallest zmax value across the wafer (+overshoot) in order to recover a flat surface. Does not use masking.

          __  lowest zmax
    <-> _|  |___ <-->
     __|        |____          _ _ _ _ _ _ _ _ _
     |              |          _________________ <-- overshoot
     |              |  ----->  |               |
     |______________|          |_______________|


     Args:
         overshoot (float): how much more than the smallest zmax value to remove
    """

    overshoot: float = 0


if __name__ == "__main__":
    # ls = get_layer_stack(substrate_thickness=50.0)
    # ls = get_layer_stack()
    # script = ls.get_klayout_3d_script()
    # print(script)
    # print(ls.get_layer_to_material())
    # print(ls.get_layer_to_thickness())

    from gdsfactory.technology.layer_stack import get_process

    get_process()
