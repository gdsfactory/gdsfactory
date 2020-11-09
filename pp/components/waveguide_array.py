import pp
from pp.port import deco_rename_ports
from pp.components import waveguide


@pp.autoname
@deco_rename_ports
def waveguide_array(n_waveguides: int = 4, spacing: float = 4.0, waveguide=waveguide):
    """array of waveguides connected with grating couplers
    useful to align the 4 corners of the chip
    """

    c = pp.Component()
    w = pp.call_if_func(waveguide)

    for i in range(n_waveguides):
        wref = c.add_ref(w)
        wref.y += i * (spacing + w.width)
        c.ports["E" + str(i)] = wref.ports["E0"]
        c.ports["W" + str(i)] = wref.ports["W0"]
    return c


if __name__ == "__main__":
    c = waveguide_array()
    pp.show(c)
