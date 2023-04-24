import gdsfactory as gf
from gdstk import Polygon
import numpy as np
import gdstk


def manhattanize_polygon(
    p: Polygon,
    min_step: float = 0.05,
):
    """Return a Manhattanized version of the input polygon (where non-x and non-y parallel segments are decomposed into a staircase of small x and y-parallel segments)

    Implemented in pure Python, and hence only suited to small polygons.

    Arguments:
        p: input polygon
        min_step: minimum

    Returns:
        manhattanized polygon
    """
    p_manhattan = []
    points = list(p.points)
    points.append(p.points[0])
    for pt1, pt2 in zip(points[:-1], points[1:]):
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        if dx == 0 or dy == 0:
            p_manhattan.append(pt1)
        else:
            num_x_steps = abs(int(np.floor(dx / min_step)))
            num_y_steps = abs(int(np.floor(dy / min_step)))
            num_steps = np.min([num_x_steps, num_y_steps])
            x_step = dx / num_steps
            y_step = dy / num_steps
            cur_x = pt1[0] - 0.5 * x_step
            cur_y = pt1[1]
            for _ in range(num_steps):
                p_manhattan.extend(
                    (
                        (cur_x + x_step, cur_y),
                        (cur_x + x_step, cur_y + y_step),
                    )
                )
                cur_x += x_step
                cur_y += y_step
    return gdstk.Polygon(p_manhattan)


def test_manhattanize():
    c = gf.Component("route")
    poly = gdstk.rectangle((-2, -2), (2, 2))
    poly.rotate(np.pi / 4)
    poly.scale(1, 0.5)
    init_poly = c.add_polygon(poly, layer=1)
    final_poly = c.add_polygon(manhattanize_polygon(poly, min_step=0.05), layer=2)

    assert len(init_poly.points) == 4
    assert len(final_poly.points) == 228


if __name__ == "__main__":
    c = gf.Component()

    poly = gdstk.rectangle((-2, -2), (2, 2))
    poly.rotate(np.pi / 4)
    poly.scale(1, 0.5)

    init_poly = c.add_polygon(poly, layer=1)

    final_poly = c.add_polygon(manhattanize_polygon(poly), layer=2)

    c.show()

    test_manhattanize()
