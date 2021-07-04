import math

# Some global settings
PADDING = 10000  # 10um padding
STEP_SIZE = 1
GRID = 1
WORKING_MEMORY = {}
DEVREC_LAYER = 68
FLOORPLAN_LAYER = 64
DICING_LAYERS = [(1, 0), (41, 30), (45, 30), (49, 30)]
TEXT = 66
COUNTER = 0
AUTOPLACER_REGISTRY = {}

VERTICAL = 101
HORIZONTAL = 102

NORTH = 201
SOUTH = 202
EAST = 203
WEST = 204
MIDDLE = 0

HEIGHT = 301
WIDTH = 302
BOTH = 303
AREA = 304
NEITHER = None

GRANULARITY = PADDING * 8

NORTH_EAST = (NORTH, EAST)
NORTH_WEST = (NORTH, WEST)
SOUTH_EAST = (SOUTH, EAST)
SOUTH_WEST = (SOUTH, WEST)
CENTER = (MIDDLE, MIDDLE)
CORNERS = [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST]


class CellsNotFoundError(Exception):
    pass


class OutOfSpaceError(Exception):
    pass


def mean(values):
    """Just compute the mean"""
    return sum(values) / len(values)


def area(cell):
    """Get the area of a cell, used for sorting"""
    return cell.bbox().width()


def factors(n):
    """Get all the factors of n"""
    return [i for i in range(1, n + 1) if n % i == 0]


def estimate_cols(n, width, height, target):
    """
    Estimate the number of cols required
    to acheive a certain target aspect ratio
    """
    cols = factors(n)

    # Hack to deal with prime numbers
    if cols == [1, n]:
        cols = [math.ceil(math.sqrt(n))]
    aspect = {c: c * width / (height * n / c) for c in cols}
    return sorted(cols, key=lambda c: abs(aspect[c] - target))[0]


def longest_common_prefix(names):
    """Used to infer group names"""
    if not names:
        return "Empty"
    length = max(len(n) for n in names)
    for i in range(length, 0, -1):
        prefix = names[0][:i]
        if all(n.startswith(prefix) for n in names):
            return prefix
        return prefix
