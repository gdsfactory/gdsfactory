import pp
from pp.components.pcm.cd import CENTER_SHAPES_MAP


@pp.cell
def cd_straight(
    spacing_h=5.0,
    spacing_v=8.0,
    gaps=(0.224, 0.234, 0.246),
    length=10.0,
    width_center=0.5,
    layer=pp.LAYER.WG,
):

    c = pp.Component()
    x = 0
    i = 0

    widths = [width_center * 0.92, width_center, width_center * 1.08]

    for width, marker_type in zip(widths, ["D", "S", "U"]):
        y = 0
        # iso line
        _r = pp.components.rectangle(size=(length, width), layer=layer, centered=True)
        _r_ref = c.add_ref(_r)
        _r_ref.move((x, y))
        c.absorb(_r_ref)

        marker = CENTER_SHAPES_MAP[marker_type]()
        _marker = c.add_ref(marker)
        _marker.move((x, -2.0))
        c.absorb(_marker)

        y += width + spacing_v
        # dual lines
        for gap, col_marker_type in zip(gaps, ["S", "D", "U"]):
            _r1_ref = c.add_ref(_r)
            _r1_ref.move((x, y))

            _r2_ref = c.add_ref(_r)
            _r2_ref.move((x, y + gap + width))
            c.absorb(_r1_ref)
            c.absorb(_r2_ref)

            if i < 2:
                marker = CENTER_SHAPES_MAP[col_marker_type]()
                _marker = c.add_ref(marker)
                _marker.move((x + length / 2 + spacing_h / 2, y + width + gap / 2))
                c.absorb(_marker)

            y += 2 * width + gap + spacing_v
        i += 1

        x += length + spacing_h

    c.move(c.size_info.cc, (0, 0))
    return c


if __name__ == "__main__":
    c = cd_straight()
    c.show()
