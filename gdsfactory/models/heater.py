"""equations from paper https://doi.org/10.1364/OE.27.010456
"""

dn_dt_si = 1.87e-4
dn_dt_sio2 = 8.5e-6


def delta_temperature_pi(
    length: float, wavelength: float = 1.55, dndT: float = 1.8e-4
) -> float:
    return wavelength / (2.0 * length * dndT)


if __name__ == "__main__":
    for length in [320, 600]:
        dT = delta_temperature_pi(length=length)
        print(f"length = {length}, dT = {dT:.3f} K")
