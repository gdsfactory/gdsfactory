import gdsfactory as gf


@gf.cell
def io_os_en(
    no: int = 4,
    ne: int = 2,
    with_loopback: bool = True,
    pad_spacing: float = 100.0,
    gc_spacing: float = 127.0,
    pad_gc_spacing: float = 250.0,
) -> gf.Component:
    """Returns a IO with optical switches and electrical pads

    Args:
        no: number of optical ports.
        ne: number of electrical ports.
        with_loopback: whether to add a loopback port.
        pad_spacing: spacing between pads.
        gc_spacing: spacing between grating couplers.
        pad_gc_spacing: spacing between pads and grating couplers.
    """
    c = gf.Component()
    return c


if __name__ == "__main__":
    c = io_os_en()
    c.show()
