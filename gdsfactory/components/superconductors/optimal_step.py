from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def optimal_step(
    start_width: float = 10,
    end_width: float = 22,
    num_pts: int = 50,
    width_tol: float = 1e-3,
    anticrowding_factor: float = 1.2,
    symmetric: bool = False,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns an optimally-rounded step geometry.

    Args:
        start_width: Width of the connector on the left end of the step.
        end_width: Width of the connector on the right end of the step.
        num_pts: number of points comprising the entire step geometry.
        width_tol: Point at which to terminate the calculation of the optimal step
        anticrowding_factor: Factor to reduce current crowding by elongating
            the structure and reducing the curvature
        symmetric: If True, adds a mirrored copy of the step across the x-axis to the
            geometry and adjusts the width of the ports.
        layer: layer spec to put polygon geometry on.

    based on phidl.geometry
    Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1-27.
    """
    import numpy as np

    # Vectorized computation of step points
    def step_points_vectorized(
        eta: np.ndarray, W: complex, a: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        gamma = (a**2 + W**2) / (a**2 - W**2)
        w = np.exp(1j * eta)
        zeta = (
            4
            * 1j
            / np.pi
            * (
                W * np.arctan(np.sqrt((w - gamma) / (gamma + 1)))
                + a * np.arctan(np.sqrt((gamma - 1) / (w - gamma)))
            )
        )
        return np.real(zeta), np.imag(zeta)

    def step_points(eta: float, W: complex, a: complex) -> tuple[float, float]:
        """Returns step points.

        Returns points from a unit semicircle in the w (= u + iv) plane to
        the optimal curve in the zeta (= x + iy) plane which transitions
        a wire from a width of 'W' to a width of 'a'
        eta takes value 0 to pi
        """
        gamma = (a**2 + W**2) / (a**2 - W**2)
        w = np.exp(1j * eta)
        zeta = (
            4
            * 1j
            / np.pi
            * (
                W * np.arctan(np.sqrt((w - gamma) / (gamma + 1)))
                + a * np.arctan(np.sqrt((gamma - 1) / (w - gamma)))
            )
        )
        return np.real(zeta), np.imag(zeta)

    if start_width > end_width:
        reverse = True
        s_w, e_w = end_width, start_width
    else:
        reverse = False
        s_w, e_w = start_width, end_width

    D = Component()
    if s_w == e_w:  # Just return a square
        if symmetric:
            ypts = [
                -s_w / 2,
                s_w / 2,
                s_w / 2,
                -s_w / 2,
            ]
            xpts = [0, 0, s_w, s_w]
        else:
            ypts = [0, s_w, s_w, 0]
            xpts = [0, 0, s_w, s_w]
        D.info["num_squares"] = 1
        # Only minimal work, no optimization possible here
    else:
        # Use vectorized computation for optimal step points
        # First, find minimum and maximum x (corresponding to given y)
        def invert_step_point_y(
            y_desired: float, W: float, a: float
        ) -> tuple[float, float]:
            # Only called twice; keep as-is for clarity.
            from scipy.optimize import fminbound

            def fh(eta: float) -> float:
                guessed_x, guessed_y = step_points(eta, W=W + 0j, a=a + 0j)
                return (guessed_y - y_desired) ** 2

            found_eta = fminbound(fh, 0, np.pi)
            return step_points(found_eta, W=W + 0j, a=a + 0j)

        xmin, _ = invert_step_point_y(y_desired=s_w * (1 + width_tol), W=s_w, a=e_w)
        xmax, _ = invert_step_point_y(y_desired=e_w * (1 - width_tol), W=s_w, a=e_w)

        # Instead of slow root-finding per x, vectorize the solution:
        # Compute step curve as parametric function with uniform sampling of eta, then interpolate
        eta_samples = np.linspace(0, np.pi, num_pts)
        step_x, step_y = step_points_vectorized(eta_samples, W=s_w + 0j, a=e_w + 0j)
        # Rescale so that step_x[0] ≈ xmin, step_x[-1] ≈ xmax
        x_targets = np.linspace(xmin, xmax, num_pts)
        # Interpolate y for target x
        y_targets = np.interp(x_targets, step_x, step_y)
        xpts = x_targets.tolist()
        ypts = y_targets.tolist()

        ypts[-1] = e_w
        ypts[0] = s_w
        x_num_sq = np.array(xpts)
        y_num_sq = np.array(ypts)

        if not symmetric:
            xpts += [xpts[-1], xpts[0]]
            ypts += [0, 0]
        else:
            # Efficient symmetric copy
            xpts_new = xpts + xpts[::-1]
            ypts_new = ypts + [-y for y in ypts[::-1]]
            xpts = [x / 2 for x in xpts_new]
            ypts = [y / 2 for y in ypts_new]

        # Apply anticrowding_factor efficiently
        if anticrowding_factor != 1.0:
            xpts = [x * anticrowding_factor for x in xpts]

        if reverse:
            xpts = [-x for x in xpts]
            # Swap for correct port assignment at the end
            s_w, e_w = e_w, s_w

        D.info["num_squares"] = float(
            np.round(
                np.sum(np.diff(x_num_sq) / ((y_num_sq[:-1] + y_num_sq[1:]) / 2)), 3
            )
        )

    D.add_polygon(list(zip(xpts, ypts)), layer=layer)
    if not symmetric:
        D.add_port(
            name="e1",
            center=(min(xpts), s_w / 2),
            width=s_w,
            orientation=180,
            layer=layer,
        )
        D.add_port(
            name="e2",
            center=(max(xpts), e_w / 2),
            width=e_w,
            orientation=0,
            layer=layer,
        )
    else:
        D.add_port(
            name="e1",
            center=(min(xpts), 0),
            width=s_w,
            orientation=180,
            layer=layer,
        )
        D.add_port(
            name="e2",
            center=(max(xpts), 0),
            width=e_w,
            orientation=0,
            layer=layer,
        )

    return D


if __name__ == "__main__":
    c = optimal_step()
    c.show()
