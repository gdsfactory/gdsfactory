from typing import Tuple

import numpy as np
from numpy import float64


class Coord2:
    def __init__(self, x: float64, y: float64) -> None:
        self.point = np.array([x, y])

    def __getitem__(self, i: int) -> float64:
        return self.point[i]

    @property
    def x(self) -> float64:
        return self.point[0]

    @property
    def y(self) -> float64:
        return self.point[1]

    @property
    def xy(self) -> Tuple[float64, float64]:
        return (self.x, self.y)

    def __add__(self, c2):
        return Coord2(self[0] + c2[0], self[1] + c2[1])

    def __mul__(self, a):
        return Coord2(self[0] * a, self[1] * a)

    def __rmul__(self, a):
        return Coord2(self[0] * a, self[1] * a)

    def __str__(self):
        return "({}, {})".format(self[0], self[1])


if __name__ == "__main__":
    p0 = Coord2(1.0, 1.5)
    p1 = Coord2(2.0, 0.0)

    p2 = p0 + p1
    p3 = p0 * 2
    p4 = 2 * p0

    p5 = p3 + (0.0, 5.0)

    print(p2)
    print(p5)

    print(p3.x, p3.y)
