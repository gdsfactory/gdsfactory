def assert_grating_coupler_properties(gc: object) -> None:
    assert hasattr(
        gc, "polarization"
    ), f"{gc.name} does not have polarization attribute"
    assert gc.polarization in [
        "te",
        "tm",
    ], f"{gc.name} polarization  should be 'te' or 'tm'"
    assert hasattr(
        gc, "wavelength"
    ), f"{gc.name} wavelength does not have wavelength attribute"
    assert (
        500 < gc.wavelength < 2000
    ), f"{gc.name} wavelength {gc.wavelength} should be in nm"
    if "W0" not in gc.ports:
        print(f"grating_coupler {gc.name} should have a W0 port. It has {gc.ports}")
    if "W0" in gc.ports and gc.ports["W0"].orientation != 180:
        print(
            f"grating_coupler {gc.name} W0 port should have orientation = 180 degrees. It has {gc.ports['W0'].orientation}"
        )


if __name__ == "__main__":
    import pp

    c = pp.c.grating_coupler_elliptical_te()
    assert_grating_coupler_properties(c)
