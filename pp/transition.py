"""Pack a list of components into as few components as possible.
Adapted from phidl.Geometry.
"""
from phidl.device_layout import CrossSection
from phidl.path import _linear_transition, _sinusoidal_transition


def transition(
    cross_section1: CrossSection, cross_section2: CrossSection, width_type: str = "sine"
) -> CrossSection:
    """Creates a CrossSection that smoothly transitions between two input
    CrossSections. Only cross-sectional elements that have the `name` (as in
    X.add(..., name = 'wg') ) parameter specified in both input CrosSections
    will be created. Port names will be cloned from the input CrossSections in
    reverse.

    Raises an error when there is no name in any of the CrossSection sections.

    Args:
        cross_section1: First input CrossSection
        cross_section2: Second input CrossSection
        width_type: {'sine', 'linear'}
            Sets the type of width transition used if any widths are different
            between the two input CrossSections.

    Returns
        A smoothly-transitioning CrossSection
    """

    x1 = cross_section1
    x2 = cross_section2
    xtrans = CrossSection()

    if not x1.aliases or not x2.aliases:
        raise ValueError("No named sections in X1 and X2")

    for alias in x1.aliases.keys():
        if alias in x2.aliases:

            offset1 = x1[alias]["offset"]
            offset2 = x2[alias]["offset"]
            width1 = x1[alias]["width"]
            width2 = x2[alias]["width"]

            if callable(offset1):
                offset1 = offset1(1)
            if callable(offset2):
                offset2 = offset2(0)
            if callable(width1):
                width1 = width1(1)
            if callable(width2):
                width2 = width2(0)

            offset_fun = _sinusoidal_transition(offset1, offset2)

            if width_type == "sine":
                width_fun = _sinusoidal_transition(width1, width2)
            elif width_type == "linear":
                width_fun = _linear_transition(width1, width2)
            else:
                raise ValueError(
                    "[PHIDL] transition() width_type "
                    + "argument must be one of {'sine','linear'}"
                )

            xtrans.add(
                width=width_fun,
                offset=offset_fun,
                layer=x1[alias]["layer"],
                ports=(x2[alias]["ports"][0], x1[alias]["ports"][1]),
                name=alias,
            )

    return xtrans


def demo_fail():
    import pp

    X = pp.CrossSection()
    P = pp.Path()
    P.append(pp.path.arc(radius=10, angle=90))
    P.append(pp.path.straight(length=10))
    P.append(pp.path.euler(radius=3, angle=-90))
    P.append(pp.path.straight(length=40))
    P.append(pp.path.arc(radius=8, angle=-45))
    P.append(pp.path.straight(length=10))
    P.append(pp.path.arc(radius=8, angle=45))
    P.append(pp.path.straight(length=10))

    X.add(width=1, offset=0, layer=0)

    x2 = pp.CrossSection()
    x2.add(width=2, offset=0, layer=0)

    T = transition(X, x2)
    c3 = pp.path.component(P, T)
    c3.show()


if __name__ == "__main__":
    import pp

    X1 = pp.CrossSection()
    X1.add(width=1.2, offset=0, layer=2, name="wg", ports=("in1", "out1"))
    X1.add(width=2.2, offset=0, layer=3, name="etch")
    X1.add(width=1.1, offset=3, layer=1, name="wg2")

    # Create the second CrossSection that we want to transition to
    X2 = pp.CrossSection()
    X2.add(width=1, offset=0, layer=2, name="wg", ports=("in2", "out2"))
    X2.add(width=3.5, offset=0, layer=3, name="etch")
    X2.add(width=3, offset=5, layer=1, name="wg2")

    xtrans = pp.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")

    P1 = pp.path.straight(length=5)
    P2 = pp.path.straight(length=5)
    wg1 = pp.path.component(P1, X1)
    wg2 = pp.path.component(P2, X2)

    P4 = pp.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    wg_trans = pp.path.component(P4, xtrans)
    # WG_trans = P4.extrude(xtrans)

    c = pp.Component()
    wg1_ref = c << wg1
    wg2_ref = c << wg2
    wgt_ref = c << wg_trans

    wgt_ref.connect("W0", wg1_ref.ports["E0"])
    wg2_ref.connect("W0", wgt_ref.ports["E0"])
