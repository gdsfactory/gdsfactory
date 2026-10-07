from __future__ import annotations

__all__ = ["torus_wave"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def torus_wave(
    inner_radius: float = 5.0,
    outer_radius: float = 10.0,
    amplitude: float = 0.5,
    n_oscillations: int = 8,
    in_phase: bool = True,
    angle_resolution: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a torus (full ring) with sinusoidal boundary modulation.

    Inner and outer boundaries oscillate sinusoidally. When in_phase=True,
    both boundaries are modulated in phase; when False, they are pi/2 out
    of phase.

    Args:
        inner_radius: mean inner radius.
        outer_radius: mean outer radius.
        amplitude: amplitude of boundary oscillation.
        n_oscillations: number of oscillations around the boundary.
        in_phase: if True, inner and outer modulations are in phase.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    return cf.torus_wave(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        amplitude=amplitude,
        n_oscillations=n_oscillations,
        in_phase=in_phase,
        angle_resolution=angle_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = torus_wave()
    c.show()
