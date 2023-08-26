from pydantic.dataclasses import dataclass

from gdsfactory.technology.layer_views import Layer


@dataclass(kw_only=True)
class ProcessStep:
    """Generic process step."""

    name: str | None


@dataclass(kw_only=True)
class Lithography(ProcessStep):
    """Simulates lithography by generating a logical on-wafer mask from one or many layers to be used in processing operations.

    (1) First, a mask is created from the layer arguments:

    if layer=None, behaviour should be:

                                                 mask opening
                                              <---------------->
                                     ----->
        |________________|                    |________________|

        0       1        2         3          0       1        2         3


    if argument_layers is provided to layers_or, for those layers:

               layer                                 mask opening
        <---------------->                   <--------------------------->
                                     ----->
        |________________|                    |_______|________|_________|
                |__________________|
        0       1        2         3          0       1        2         3
                <------------------>
                    layers_or


    if argument_layers is provided to layers_and, for those layers:

               layer                                 mask opening
        <---------------->                            <--------->
                                     ----->
        |________________|                    |       |_________|        |
                |__________________|
        0       1        2         3          0       1        2         3
                <------------------>
                      layers_and

    if argument_layers is provided to layers_diff, for those layers:

               layer                                 mask opening
        <---------------->                    <------->
                                     ----->
        |________________|                    |________|        |        |
                |__________________|
        0       1        2         3          0       1        2         3
                <------------------>
                      layers_and

    if argument_layers is provided to layers_xor, for those layers:

              layer                                  mask opening
        <---------------->                   <-------->        <--------->
                                     ----->
        |________________|                    |_______|        |_________|
                |__________________|
        0       1        2         3          0       1        2         3
                <------------------>
                     layers_xor


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


    (3) (Optional) Planarize the resist


    Args:
        layer: main layer to use as a mask for this lithography step
        layers_union (List[Layers]): other layers to use to form the mask (see diagram)
        layers_diff (List[Layers]): other layers to use to form a mask (see diagram)
        layers_intersect (List[Layers]): other layers to use to form a mask (see diagram)
        positive_tone (bool): whether to invert the resulting mask (False) or not (True)
        resist_thickness (float): resist mask thickness, used in some simulators
        planarization_height (float): height at which to "clip" the resist above the wafer
    """

    layer: Layer | None = None
    layers_or: list | None = None
    layers_diff: list | None = None
    layers_and: list | None = None
    layers_xor: list | None = None
    resist_thickness: float | None = 0
    positive_tone: bool = True
    planarization_height: float = None


@dataclass(kw_only=True)
class Grow(Lithography):
    """Simulates masking + addition of material + liftoff.

    wafer mask opened           wafer mask opened
        <------>                   <------>
                                   ________
                                   |      |
                      ----->       | mat  | thickness
    ________________           ____|______|____

    Args:
        material (str): material tag of material to add
        thickness (float): thickness to add [nm]
        type (str): of growth/deposition (isotropic, anisotropic, etc.)
        rate (float): of growth [nm/s]
    """

    thickness: float
    material: str
    type: str
    rate: float = None


@dataclass(kw_only=True)
class Etch(Lithography):
    """Simulates masking + removal of material + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________           _____      _____
    |              |               |      |
    |     mat      |    -----> mat | etch | mat  depth
    |______________|           ____|______|____


    Args:
        material (str): material tag to etch into
        thickness (float): thickness to remove [nm]
        type (str): of etch (isotropic, anisotropic, etc.)
        rate (float): of removal [nm/s]

    """

    material: str
    depth: float
    type: str = "anisotropic"
    rate: float = None


@dataclass(kw_only=True)
class ImplantPhysical(Lithography):
    """Simulates masking + physical ion implantation + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________          __________________
    |              |          |                |
    |              |  ----->  |    ------- <---- range (depends on energy)
    |______________|          |________________|

    Args:
        ion (str): ion tag
        energy (float): of the ions
        dose (float): in /cm^2
        tilt (float): ion angle from out-of-plane axis. in degrees. If None, uses simulator default
        twist (float): ion angle from wafer "x-axis", in degrees. If None, uses simulator default
        rotation (float): if twist is None, toggle to split the dose 4-ways between 4 cardinal twist angles (simulates substrate rotation during implantation)
    """

    ion: str
    energy: float
    dose: float
    tilt: float = None
    twist: float = None
    rotation: float = None


@dataclass(kw_only=True)
class ImplantAnalytical(Lithography):
    """Simulates masking + physical ion implantation + strip.

    wafer mask opened          wafer mask opened
        <------>                   <----->
    ________________          __________________
    |              |          |                |
    |              |  ----->  |    ------- <---- range (depends on energy)
    |______________|          |________________|

    Args:
        ion (str): ion tag
        peak_conc (float): peak concentration
        range (float): of the ions (center of distribution)
        straggle (float): of the ions (spread of distribution)
    """

    ion: str
    peak_conc: float
    range: float
    straggle: float = None
    tilt: float = None


@dataclass(kw_only=True)
class Anneal(ProcessStep):
    """Simulates thermal diffusion of impurities and healing of defects.

    Args:
        time (float)
        temperature (float): temperature

    TODO (long term): heating/cooling time profiles
    """

    time: float
    temperature: float


@dataclass(kw_only=True)
class Planarize(ProcessStep):
    """Simulates chip planarization, "clipping" the structure above some height. Does not use masking.

         __
       _|  |___
    __|        |____
    |              |  ___                     _________________
    |              |   |  height    ----->    |               |
    |______________|  _|_  z=0                |_______________|


    Args:
        depth (float): how much to remove
    """

    height: float = 0


@dataclass(kw_only=True)
class ArbitraryStep(ProcessStep):
    """Arbitrary process step, used to handle all other cases."""

    info: str
