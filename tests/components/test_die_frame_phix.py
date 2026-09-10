import gdsfactory as gf


def test_die_frame_phix_accepts_individual_custom_fiducials() -> None:
    """Each legacy fiducial can be independently replaced."""
    layers = [(999, datatype) for datatype in range(4)]
    fiducials = [
        gf.c.rectangle(size=(30, 30), layer=layer, centered=True) for layer in layers
    ]

    die = gf.c.die_frame_phix_dc(
        fiducial_top_left=fiducials[0],
        fiducial_top_right=fiducials[1],
        fiducial_bottom_left=fiducials[2],
        fiducial_bottom_right=fiducials[3],
    )

    polygons = die.get_polygons(layers=layers, by="tuple")
    assert {layer: len(polygons[layer]) for layer in layers} == dict.fromkeys(layers, 1)
