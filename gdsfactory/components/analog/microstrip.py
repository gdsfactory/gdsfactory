"""Helper functions for RF layout.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from numpy import exp, log, pi, sinh, sqrt


def _G_integrand(xip: float, B: float) -> float:
    """Special function for microstrip calculations.

    Args:
        xip: Integration variable.
        B: Controls the intensity of the taper.

    Returns:
        Value of the integrand at xip.
    """
    try:
        from scipy.special import iv as besseli
    except ImportError as err:
        raise ImportError(
            "To run this function you need scipy, please install it with "
            "`pip install scipy`"
        ) from err
    return float(besseli(0, B * sqrt(1 - xip**2)))


def _G(xi: float, B: float) -> float:
    """Special function for microstrip calculations.

    Args:
        xi: Normalized position along the taper [-1 to +1].
        B: Controls the intensity of the taper.

    Returns:
        Value of the G function at xi.
    """
    try:
        import scipy.integrate
    except ImportError as err:
        raise ImportError(
            "To run the microstrip functions you need scipy, please install it with "
            "`pip install scipy`"
        ) from err
    return float(B / sinh(B) * scipy.integrate.quad(_G_integrand, 0, xi, args=(B,))[0])


def _microstrip_Z(
    wire_width: float, dielectric_thickness: float, eps_r: float
) -> tuple[float, float]:
    """Calculate impedance of a microstrip.

    Args:
        wire_width: Width of the conducting strip.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.

    Returns:
        Tuple of (impedance, effective dielectric constant).

    References:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design. http://doi.org/10.1109/MWSYM.1980.1124303
    """
    u = wire_width / dielectric_thickness
    eta = 376.73  # Vacuum impedance

    a = (
        1
        + log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
        + log(1 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3)) ** 0.053
    F = 6 + (2 * pi - 6) * exp(-((30.666 / u) ** 0.7528))
    eps_eff = 0.5 * (eps_r + 1) + 0.5 * (eps_r - 1) * (1 + 10 / u) ** (-a * b)
    Z = eta / (2 * pi) * log(F / u + sqrt(1 + (2 / u) ** 2)) / sqrt(eps_eff)
    return Z, eps_eff


def _microstrip_LC_per_meter(
    wire_width: float, dielectric_thickness: float, eps_r: float
) -> tuple[float, float]:
    """Calculate inductance and capacitance per meter of a microstrip.

    Args:
        wire_width: Width of the conducting strip.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.

    Returns:
        Tuple of (inductance per meter, capacitance per meter).

    References:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design. http://doi.org/10.1109/MWSYM.1980.1124303
    """
    # Use the fact that v = 1/sqrt(L_m*C_m) = 1/sqrt(eps*mu) and
    # Z = sqrt(L_m/C_m)   [Where L_m is inductance per meter]
    Z, eps_eff = _microstrip_Z(wire_width, dielectric_thickness, eps_r)
    eps0 = 8.854e-12
    mu0 = 4 * pi * 1e-7

    eps = eps_eff * eps0
    mu = mu0
    L_m = sqrt(eps * mu) * Z
    C_m = sqrt(eps * mu) / Z
    return L_m, C_m


def _microstrip_Z_with_Lk(
    wire_width: float, dielectric_thickness: float, eps_r: float, Lk_per_sq: float
) -> float:
    """Calculate impedance of a microstrip with kinetic inductance.

    Args:
        wire_width: Width of the conducting strip.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.
        Lk_per_sq: Kinetic inductance per square of the microstrip.

    Returns:
        Impedance of the microstrip.

    References:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design. http://doi.org/10.1109/MWSYM.1980.1124303
    """
    # Add a kinetic inductance and recalculate the impedance, be careful
    # to input Lk as a per-meter inductance
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq * (1.0 / wire_width)
    Z = sqrt((L_m + Lk_m) / C_m)
    return float(Z)


def _microstrip_v_with_Lk(
    wire_width: float, dielectric_thickness: float, eps_r: float, Lk_per_sq: float
) -> float:
    """Calculate propagation velocity in a microstrip.

    Args:
        wire_width: Width of the conducting strip.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.
        Lk_per_sq: Kinetic inductance per square of the microstrip.

    Returns:
        Propagation velocity in the microstrip.

    References:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design. http://doi.org/10.1109/MWSYM.1980.1124303
    """
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq * (1.0 / wire_width)
    v = 1 / sqrt((L_m + Lk_m) * C_m)
    return float(v)


def _find_microstrip_wire_width(
    Z_target: float, dielectric_thickness: float, eps_r: float, Lk_per_sq: float
) -> float:
    """Calculate wire width for a target impedance.

    Args:
        Z_target: Target impedance of the microstrip.
        dielectric_thickness: Thickness of the substrate.
        eps_r: Dielectric constant of the substrate.
        Lk_per_sq: Kinetic inductance per square of the microstrip.

    Returns:
        Wire width of the microstrip.

    References:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design. http://doi.org/10.1109/MWSYM.1980.1124303
    """

    def error_fun(wire_width: float) -> float:
        Z_guessed = _microstrip_Z_with_Lk(
            wire_width, dielectric_thickness, eps_r, Lk_per_sq
        )
        return (Z_guessed - Z_target) ** 2  # The error

    x0 = dielectric_thickness
    try:
        from scipy.optimize import fmin
    except ImportError as err:
        raise ImportError(
            "To run the microstrip functions you need scipy, please install it with "
            "`pip install scipy`"
        ) from err
    w = fmin(error_fun, x0, args=(), disp=False)
    return float(w[0])
