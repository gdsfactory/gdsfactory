"""fixme."""

import numpy as np
import gdstk

path = gdstk.FlexPath(width=0.5, points=[(-0.1, 0), (+0.1, 0)])
# print(path.points)
# AttributeError: 'gdstk.FlexPath' object has no attribute 'points'


polygons = path.to_polygons()
polygon = polygons[0]
p = polygon.points
center = np.sum(p, 0) / 2

p1 = np.sum(p[:2], 0) / 2
p2 = np.sum(p[2:], 0) / 2
