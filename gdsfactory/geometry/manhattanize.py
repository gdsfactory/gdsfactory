import gdstk
import numpy as np
from gdstk import Polygon

import gdsfactory as gf


def manhattanize_polygon(
    p: Polygon,
    target_step: float = 0.05,
):
    """Return a Manhattanized version of the input polygon (where non-x and non-y parallel segments are decomposed into a staircase of small x and y-parallel segments)

    Implemented in pure Python, and hence only suited to small polygons.

    Args:
        p: input polygon.
        target_step: target staircase step size.

    Returns:
        manhattanized polygon

    .. plot::
      :include-source:

      import gdsfactory as gf
      c = gf.Component()

      poly = gdstk.rectangle((-2, -2), (2, 2))
      poly.rotate(np.pi / 4)
      poly.scale(1, 0.5)
      init_poly = c.add_polygon(poly, layer=1)
      final_poly = gf.geometry.manhattanize_polygon(poly)
      c.add_polygon(final_poly, layer=2)
      c.plot_matplotlib()

    """
    p_manhattan = []
    points = list(p.points)
    points.append(p.points[0])
    x_polarity = 0
    for pt1, pt2 in zip(points[:-1], points[1:]):
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        if dx == 0 or dy == 0:
            p_manhattan.append(pt1)
            p_manhattan.append(pt2)
        else:
            num_x_steps = abs(int(np.ceil(dx / target_step)))
            num_y_steps = abs(int(np.ceil(dy / target_step)))
            num_steps = np.min([num_x_steps, num_y_steps]) + 1
            x_step = dx / num_steps
            y_step = dy / num_steps
            cur_x = pt1[0]
            cur_y = pt1[1]
            for _ in range(num_steps):
                if not x_polarity % 2:
                    p_manhattan.extend(
                        (
                            (cur_x + 0.5 * x_step, cur_y),
                            (cur_x + 0.5 * x_step, cur_y + y_step),
                            (cur_x + x_step, cur_y + y_step),
                        )
                    )
                else:
                    p_manhattan.extend(
                        (
                            (cur_x, cur_y + 0.5 * y_step),
                            (cur_x + x_step, cur_y + 0.5 * y_step),
                            (cur_x + x_step, cur_y + y_step),
                        )
                    )
                cur_x += x_step
                cur_y += y_step
            x_polarity += 1
    return gdstk.Polygon(p_manhattan)


def test_manhattanize() -> None:
    c = gf.Component("route")
    poly = gdstk.rectangle((-2, -2), (2, 2))
    poly.rotate(np.pi / 4)
    poly.scale(1, 0.5)
    init_poly = c.add_polygon(poly, layer=1)
    final_poly = c.add_polygon(manhattanize_polygon(poly, target_step=0.05), layer=2)

    assert len(init_poly.points) == 4
    assert len(final_poly.points) == 354


if __name__ == "__main__":
    c = gf.Component()

    poly = gdstk.rectangle((-2, -2), (2, 2))
    poly.rotate(np.pi / 4)
    poly.scale(1, 0.5)
    init_poly = c.add_polygon(poly, layer=1)
    final_poly = c.add_polygon(manhattanize_polygon(poly), layer=2)
    c.show()

    # test_manhattanize()
