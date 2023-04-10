from __future__ import annotations


def fsr(
    ng: float = 4.2,
    delta_length: float = 40,
    wavelength: float = 1.55,
) -> float:
    """Returns Free Spectral Range.

    Args:
        ng: group index.
        delta_length: in um.
        wavelength: in um.
    """
    return wavelength**2 / delta_length / ng


if __name__ == "__main__":
    print(int(fsr() * 1e3))
