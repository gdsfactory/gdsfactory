import sys
from pathlib import Path

import numpy as np

import gdsfactory as gf

_LAYER = gf.gpdk.LAYER.WG

_CALLS: list[tuple[list[tuple[float, float]], float, float]] = [
    (
        [(0.0, 7.5), (0.0, 38.5), (-4.0, 38.5), (-6.0, 38.5), (-6.0, 43.0)],
        1.0,
        0.5,
    ),
    (
        [(0.0, 7.5), (0.0, -4.0), (0.0, -5.0), (1.0, -6.0)],
        1.0,
        0.5,
    ),
    (
        [
            (0.0, 0.0),
            (0.0, -2.5),
            (-2.5, -4.0),
            (-2.5, -7.5),
            (-1.5, -9.0),
            (-1.5, -9.5),
        ],
        1.5,
        0.5,
    ),
    (
        [(0.0, -11.0), (0.0, -38.5), (4.0, -38.5), (6.0, -38.5), (6.0, -43.0)],
        1.0,
        0.5,
    ),
]


def _smooth_path(
    points: list[tuple[float, float]],
    width: float,
    radius: float,
) -> gf.Component:
    path = gf.path.smooth(points=np.array(points), radius=radius, bend=gf.path.euler)
    section = gf.Section(width=width, offset=0.0, layer=_LAYER, port_names=(None, None))
    return gf.path.extrude(path, gf.CrossSection(sections=(section,)))


def _main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "out"
    out = gf.Component()
    for index, (points, width, radius) in enumerate(_CALLS):
        ref = out.add_ref(_smooth_path(points, width, radius))
        ref.ymin = 0.0
        ref.dmove((index * 10.0, 0.0))

    Path("breakage_artifacts").mkdir(exist_ok=True)
    out.write_gds(f"breakage_artifacts/mwe_round_{label}.gds")
    print(f"gdsfactory={gf.__version__}  wrote mwe_round_{label}.gds")
    out.show()


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    _main()
