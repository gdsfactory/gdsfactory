from gdspy import Path
from phidl import Device as Component
from scipy.special import binom

import gdsfactory as gf
from gdsfactory.cross_section import strip
from gdsfactory.types import CrossSectionSpec


# Base bezier function
def _bezier_curve(t, control_points):
    """Returns bezier coordinates.

    Args:
        t: 1D array of points varying between 0 and 1.
    """
    xs = 0.0
    ys = 0.0
    n = len(control_points) - 1
    for k in range(n + 1):
        ank = binom(n, k) * (1 - t) ** (n - k) * t ** k
        xs += ank * control_points[k][0]
        ys += ank * control_points[k][1]

    return (xs, ys)


def _generate_S_bend(wg_width, bezier_function, final_width, layer, datatype):
    # Top input S-bend
    bend = Path(width=wg_width)
    bend.parametric(
        bezier_function,
        max_points=199,
        tolerance=0.00001,
        final_width=final_width,
        layer=layer,
        datatype=datatype,
    )

    return bend


@gf.cell
def coupler_adiabatic(
    length1: float = 20.0,
    length2: float = 50.0,
    length3: float = 30.0,
    wg_sep: float = 1.0,
    input_wg_sep: float = 3.0,
    output_wg_sep: float = 3.0,
    dw: float = 0.1,
    cross_section: CrossSectionSpec = strip,
    **kwargs
):
    """Returns 50/50 adiabatic coupler.

    Design based on asymmetric adiabatic 3dB coupler designs, such as those.
    - https://doi.org/10.1364/CLEO.2010.CThAA2,
    - https://doi.org/10.1364/CLEO_SI.2017.SF1I.5
    - https://doi.org/10.1364/CLEO_SI.2018.STh4B.4

    Has input Bezier curves, with poles set to half of the x-length of the S-bend.
    I is the first half of input S-bend where input widths taper by +dw and -dw
    II is the second half of the S-bend straight with constant, unbalanced widths
    III is the region where the two asymmetric straights gradually come together
    IV  straights taper back to the original width at a fixed distance from one another
    IV is the output S-bend straight.

    Args:
        length1: region that gradually brings the two asymmetric straights together.
            In this region the straight widths gradually change to be different by `dw`.
        length2: coupling region, where asymmetric straights gradually
            become the same width.
        length3: output region where the two straights separate.
        wg_sep: Distance between center-to-center in the coupling region (Region 2).
        input_wg_sep: Separation of the two straights at the input, center-to-center.
        output_wg_sep: Separation of the two straights at the output, center-to-center.
        dw: Change in straight width.
            In Region 1, top arm tapers to width+dw/2.0, bottom taper to width-dw/2.0.
        cross_section: spec.

    Keyword Args:
        cross_section kwargs.
    """
    # Control points for input and output S-bends
    control_points_input_top = [
        (0, 0),
        (length1 / 2.0, 0),
        (length1 / 2.0, -input_wg_sep / 2.0 + wg_sep / 2.0),
        (length1, -input_wg_sep / 2.0 + wg_sep / 2.0),
    ]

    control_points_input_bottom = [
        (0, -input_wg_sep),
        (length1 / 2.0, -input_wg_sep),
        (length1 / 2.0, -input_wg_sep / 2.0 - wg_sep / 2.0),
        (length1, -input_wg_sep / 2.0 - wg_sep / 2.0),
    ]

    control_points_output_top = [
        (length1 + length2, -input_wg_sep / 2.0 + wg_sep / 2.0),
        (
            length1 + length2 + length3 / 2.0,
            -input_wg_sep / 2.0 + wg_sep / 2.0,
        ),
        (
            length1 + length2 + length3 / 2.0,
            -input_wg_sep / 2.0 + output_wg_sep / 2.0,
        ),
        (
            length1 + length2 + length3,
            -input_wg_sep / 2.0 + output_wg_sep / 2.0,
        ),
    ]

    control_points_output_bottom = [
        (length1 + length2, -input_wg_sep / 2.0 - wg_sep / 2.0),
        (
            length1 + length2 + length3 / 2.0,
            -input_wg_sep / 2.0 - wg_sep / 2.0,
        ),
        (
            length1 + length2 + length3 / 2.0,
            -input_wg_sep / 2.0 - output_wg_sep / 2.0,
        ),
        (
            length1 + length2 + length3,
            -input_wg_sep / 2.0 - output_wg_sep / 2.0,
        ),
    ]

    # Bend specific bezier functions
    def _bezier_inp_top(t):
        return _bezier_curve(t, control_points_input_top)

    def _bezier_inp_bottom(t):
        return _bezier_curve(t, control_points_input_bottom)

    def _bezier_output_bottom(t):
        return _bezier_curve(t, control_points_output_bottom)

    def _bezier_output_top(t):
        return _bezier_curve(t, control_points_output_top)

    c = Component()

    x = gf.get_cross_section(cross_section, **kwargs)

    wg_width = x.width

    c.add(
        _generate_S_bend(
            wg_width=wg_width,
            bezier_function=_bezier_inp_top,
            final_width=wg_width + dw / 2.0,
            layer=gf.get_layer(x.layer)[0],
            datatype=gf.get_layer(x.layer)[1],
        )
    )

    # Bottom input S-bend
    c.add(
        _generate_S_bend(
            wg_width=wg_width,
            bezier_function=_bezier_inp_bottom,
            final_width=wg_width - dw / 2.0,
            layer=gf.get_layer(x.layer)[0],
            datatype=gf.get_layer(x.layer)[1],
        )
    )

    # Top output S-bend
    c.add(
        _generate_S_bend(
            wg_width=wg_width,
            bezier_function=_bezier_output_top,
            final_width=wg_width,
            layer=gf.get_layer(x.layer)[0],
            datatype=gf.get_layer(x.layer)[1],
        )
    )

    # Bottom output S-bend
    c.add(
        _generate_S_bend(
            wg_width=wg_width,
            bezier_function=_bezier_output_bottom,
            final_width=wg_width,
            layer=gf.get_layer(x.layer)[0],
            datatype=gf.get_layer(x.layer)[1],
        )
    )

    # Get cladding layers
    try:
        if kwargs["cladding_layers"] is None:
            kwargs["cladding_layers"] = (111, 0)
        else:
            layers = []
            for layer in kwargs["cladding_layers"]:
                layers.append(gf.get_layer(layer))
            kwargs["cladding_layers"] = layers
    except KeyError:
        kwargs["cladding_layers"] = (111, 0)

    # Instantiate S-bend cladding objects
    try:
        c.add(
            _generate_S_bend(
                wg_width=2 * x.cladding_offsets[0] + wg_width + dw / 2.0,
                bezier_function=_bezier_inp_top,
                final_width=2 * x.cladding_offsets[0] + wg_width + dw / 2.0,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * x.cladding_offsets[0] + wg_width - dw / 2.0,
                bezier_function=_bezier_inp_bottom,
                final_width=2 * x.cladding_offsets[0] + wg_width - dw / 2.0,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * x.cladding_offsets[0] + wg_width,
                bezier_function=_bezier_output_top,
                final_width=2 * x.cladding_offsets[0] + wg_width,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * x.cladding_offsets[0] + wg_width,
                bezier_function=_bezier_output_bottom,
                final_width=2 * x.cladding_offsets[0] + wg_width,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
    except (TypeError, KeyError):
        c.add(
            _generate_S_bend(
                wg_width=2 * 2 + wg_width + dw / 2.0,
                bezier_function=_bezier_inp_top,
                final_width=2 * 2 + wg_width + dw / 2.0,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * 2 + wg_width - dw / 2.0,
                bezier_function=_bezier_inp_bottom,
                final_width=2 * 2 + wg_width + dw / 2.0,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * 2 + wg_width,
                bezier_function=_bezier_output_top,
                final_width=2 * 2 + wg_width,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )
        c.add(
            _generate_S_bend(
                wg_width=2 * 2 + wg_width,
                bezier_function=_bezier_output_bottom,
                final_width=2 * 2 + wg_width,
                layer=kwargs["cladding_layers"][0],
                datatype=kwargs["cladding_layers"][1],
            )
        )

    # Kwargs for top and bottom taper. Derive from input
    # kwargs, and then modify if necessary
    kwargs_top_taper = kwargs
    kwargs_bot_taper = kwargs

    try:
        update_offset_top = []
        update_offset_bot = []

        for offset in kwargs["cladding_offsets"]:
            update_offset_top.append(offset)
            update_offset_bot.append(offset)
        kwargs_top_taper["cladding_offsets"] = update_offset_top
        kwargs_bot_taper["cladding_offsets"] = update_offset_bot
    except KeyError:
        kwargs_top_taper["cladding_offsets"] = [2 * x.cladding_offsets[0]]
        kwargs_bot_taper["cladding_offsets"] = [2 * x.cladding_offsets[0]]

    try:
        update_layer_top = []
        update_layer_bot = []
        for layer in kwargs["cladding_layers"]:
            update_layer_top.append(gf.get_layer(layer))
            update_layer_bot.append(gf.get_layer(layer))
        kwargs_top_taper["cladding_layers"] = update_layer_top
        kwargs_bot_taper["cladding_layers"] = update_layer_bot
    except (TypeError, KeyError):
        kwargs_top_taper["cladding_layers"] = [(111, 0)]
        kwargs_bot_taper["cladding_layers"] = [(111, 0)]

    # Define taper sections
    taper_top = gf.components.taper2(
        length=length2,
        width1=wg_width + dw / 2.0,
        width2=wg_width,
        cross_section=cross_section,
        **kwargs_top_taper
    )

    taper_bottom = gf.components.taper2(
        length=length2,
        width1=wg_width - dw / 2.0,
        width2=wg_width,
        cross_section=cross_section,
        **kwargs_bot_taper
    )

    # Generate gdsfactory component from phidl object
    c = gf.read.from_phidl(c)

    # Add ports to easily connect taper objects to S-bend
    c.add_port(
        "o2", (0, 0), wg_width, 180, layer="PORT", cross_section=x.copy(width=wg_width)
    )
    c.add_port(
        "o1",
        (0, -input_wg_sep),
        wg_width,
        180,
        layer="PORT",
        cross_section=x.copy(width=wg_width),
    )
    try:
        c.add_port(
            "top_taper_connect",
            (length1, -input_wg_sep / 2.0 + wg_sep / 2.0),
            wg_width,
            0,
            layer="PORT",
            cross_section=x.copy(width=2 * x.cladding_offsets[0] + wg_width),
        )
        c.add_port(
            "bottom_taper_connect",
            (length1, -input_wg_sep / 2.0 - wg_sep / 2.0),
            wg_width,
            0,
            layer="PORT",
            cross_section=x.copy(width=2 * x.cladding_offsets[0] + wg_width),
        )
    except TypeError:
        c.add_port(
            "top_taper_connect",
            (length1, -input_wg_sep / 2.0 + wg_sep / 2.0),
            wg_width,
            0,
            layer="PORT",
            cross_section=x.copy(width=2 * 2 + wg_width),
        )
        c.add_port(
            "bottom_taper_connect",
            (length1, -input_wg_sep / 2.0 - wg_sep / 2.0),
            wg_width,
            0,
            layer="PORT",
            cross_section=x.copy(width=2 * 2 + wg_width),
        )

    # Add tapers to component
    taper_top_ref = c << taper_top

    taper_bottom_ref = c << taper_bottom

    # Connect tapers
    taper_top_ref.connect("o1", c.ports["top_taper_connect"])
    taper_bottom_ref.connect("o1", c.ports["bottom_taper_connect"])

    # Redefine components ports
    c.ports = {
        "o1": c.ports["o1"],
        "o2": c.ports["o2"],
    }

    c.add_port(
        "o3",
        (
            length1 + length2 + length3,
            -input_wg_sep / 2.0 + output_wg_sep / 2.0,
        ),
        wg_width,
        0,
        layer="PORT",
        cross_section=x.copy(width=wg_width),
    )

    c.add_port(
        "o4",
        (
            length1 + length2 + length3,
            -input_wg_sep / 2.0 - output_wg_sep / 2.0,
        ),
        wg_width,
        0,
        layer="PORT",
        cross_section=x.copy(width=wg_width),
    )

    c.absorb(taper_top_ref)
    c.absorb(taper_bottom_ref)

    # Create new component and combine all polygons to form final polygon
    c2 = gf.Component()

    c2 << gf.geometry.union(c, by_layer=True)

    if x.info:
        c2.info.update(x.info)

    if x.add_pins:
        c2 = x.add_pins(c2)

    if x.add_bbox:
        c2 = x.add_bbox(c2)

    c2.ports = c.ports

    return c2


if __name__ == "__main__":

    c = coupler_adiabatic(length3=5, cladding_offsets=[0.5])
    print(c.ports)
    c.show(show_ports=True)
