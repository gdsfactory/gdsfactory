"""Directional Couplers."""

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import sax
import xarray as xr
from jaxtyping import ArrayLike

from .waveguides import bend_euler, straight_strip

if TYPE_CHECKING:
    SDict = sax.SDict
else:
    SDict = "sax.SDict"

CWD = Path(__file__).resolve().parent

with jax.ensure_compile_time_eval():
    xarr_dc_strip = (
        xr.open_dataarray(CWD / "directional_coupler_strip.nc")
        .load()
        .expand_dims({"kappa": ["kappa"]}, -1)
    )
    xarr_racetrack_strip = (
        xr.open_dataarray(CWD / "coupler_racetrack_strip.nc")
        .load()
        .expand_dims({"kappa": ["kappa"]}, -1)
    )


def _interpolate_kappa(xarr: xr.DataArray, **kwargs: ArrayLike) -> jnp.ndarray:
    # Extract interpolation dims from xarray
    dims = [d for d in xarr.coords if d != "kappa"]

    # Ensure required args are provided
    missing = [d for d in dims if d not in kwargs]
    if missing:
        raise ValueError(f"Missing required interpolation inputs: {missing}")

    # Broadcast all input arrays
    arrays = [jnp.asarray(kwargs[dim]) for dim in dims]
    broadcasted = jnp.broadcast_arrays(*arrays)
    shape = broadcasted[0].shape

    # Prepare kwargs for interpolation
    interp_args = {dim: arr.ravel() for dim, arr in zip(dims, broadcasted, strict=True)}

    # Interpolate
    result = sax.interpolate_xarray(xarr, **interp_args)["kappa"]
    return result.reshape(shape)


def directional_coupler_no_phase(
    *,
    wl: float = 1.3,
    coupler_length: float = 10.0,
    gap: float = 0.5,
    offset: float = 20,
    bend_radius: float = 25,
    width: float = 1.0,
    cross_section: str = "strip",
) -> SDict:
    r"""Ring coupler model.

    Semi-analytical model for ring couplers developed by GDSFactory.
    Use at your own risk.

    Args:
        wl: wavelength [µm]; between 1.5 and 1.6 µm.
        gap: gap between the two waveguides [µm]; between 0.05 and 1.5 µm.
        coupler_length: length of the ring coupler [µm]; between 0 and 100 µm.
        offset: offset between the two waveguides [µm]; between 5 and 100 µm.
        bend_radius: bend radius of the ring coupler [µm]; between 5 and 100 µm.
        width: width of the waveguides [µm]; between 0.1 and 10 µm.
        cross_section: cross section of the waveguide.
    """
    if cross_section == "strip":
        xarr = xarr_dc_strip
    kappa = _interpolate_kappa(
        xarr=xarr,
        wavelength=wl,
        radius=bend_radius,
        gap=gap,
        length_x=coupler_length,
        v_offset=offset,
    )

    tau = jnp.sqrt(1 - jnp.array(kappa) ** 2)

    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def directional_coupler(
    *,
    wl: float = 1.3,
    length: float = 10.0,
    gap: float = 0.5,
    offset: float = 20,
    bend_radius: float = 25,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: str = "strip",
) -> SDict:
    r"""Directional coupler model.

    Semi-analytical model for directional couplers developed by GDSFactory.
    Use at your own risk.

    Args:
        wl: wavelength [µm]
        gap: gap between the two waveguides [µm]
        length: length of the ring coupler [µm]
        offset: offset between the two waveguides [µm]
        bend_radius: bend radius of the ring coupler [µm]
        with_euler: if True, the directional coupler will have an Euler bend.
        width: width of the waveguides [µm].
        cross_section: cross section of the waveguide.
    """
    if with_euler:
        raise NotImplementedError("Euler bend is not implemented yet")

    coupler_length = length

    def sbend_length(radius: float, offset: float) -> float:
        return float(2 * radius * jnp.arccos(1 - offset / 2 / radius))

    coupler_circuit, info = sax.circuit(
        netlist={
            "instances": {
                "s1": "straight",
                "s2": "straight",
                "s3": "straight",
                "s4": "straight",
                "dc": "coupling_area",
            },
            "connections": {
                "s1,o1": "dc,o1",
                "s2,o1": "dc,o2",
                "s3,o1": "dc,o3",
                "s4,o1": "dc,o4",
            },
            "ports": {
                "o1": "s1,o2",
                "o2": "s2,o2",
                "o3": "s3,o2",
                "o4": "s4,o2",
            },
        },
        models={
            "straight": straight_strip,
            "coupling_area": directional_coupler_no_phase,
        },
    )

    s = coupler_circuit(
        wl=wl,
        dc={
            "coupler_length": coupler_length,
            "gap": gap,
            "bend_radius": bend_radius,
            "cross_section": cross_section,
        },
        s1={"length": coupler_length / 2 + sbend_length(bend_radius, offset)},
        s2={"length": coupler_length / 2 + sbend_length(bend_radius, offset)},
        s3={"length": coupler_length / 2 + sbend_length(bend_radius, offset)},
        s4={"length": coupler_length / 2 + sbend_length(bend_radius, offset)},
    )

    return sax.reciprocal(
        {
            ("o1", "o4"): s["o1", "o4"],
            ("o1", "o3"): s["o1", "o3"],
            ("o2", "o4"): s["o2", "o4"],
            ("o2", "o3"): s["o2", "o3"],
        }
    )


coupler_strip = directional_coupler
coupler = directional_coupler


def coupler_ring_coupling_area(
    *,
    wl: float = 1.3,
    gap: float = 0.1,
    radius: float = 5.0,
    length_x: float = 1.0,
    cross_section: str = "strip",
) -> SDict:
    r"""Ring coupler model.

    This is a semi-analytical model developed by GDSFactory.
    GDSFactory does not guaranee the accuracy of this model.
    This model has not been validated by the foundry.
    Please use at your own discretion.

    Args:
        wl: wavelength [µm]; between 1.2 and 1.4 µm.
        gap: gap between the two waveguides [µm]; between 0.05 and 1.1 µm.
        radius: radius of the ring [µm]; between 5 and 200 µm.
        length_x: length of the ring coupler [µm]; between 0 and 20 µm.
        cross_section: cross section of the waveguide.
    """
    if cross_section == "strip":
        xarr = xarr_racetrack_strip

    kappa = _interpolate_kappa(
        xarr=xarr,
        wavelength=wl,
        gap=gap,
        radius=radius,
        length_x=length_x,
    )

    tau = jnp.sqrt(1 - jnp.array(kappa) ** 2)

    kappa *= 0.95  # adding 5% loss to the coupler
    tau *= 0.95  # adding 5% loss to the coupler

    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def coupler_ring(  # this is not the complete model!!!!
    *,
    wl: float = 1.3,
    gap: float = 0.1,
    radius: float = 40.0,
    length_x: float = 1.0,
    p: float = 0,
    wl0: float = 0,  # this is not used in the model
    cross_section: str = "strip",
) -> SDict:
    r"""Ring coupler model.

    This is a semi-analytical model developed by GDSFactory.
    GDSFactory does not guaranee the accuracy of this model.
    This model has not been validated by the foundry.
    Please use at your own discretion.

    Args:
        wl: wavelength [µm]; between 1.5 and 1.6 µm.
        gap: gap between the two waveguides [µm]; between 0.01 and 1.5 µm.
        radius: radius of the ring [µm]; between 25 and 200 µm.
        length_x: length of the ring coupler [µm]; between 0 and 20 µm.
        p: bend parameter percentage (0: circular, 1: euler).
        wl0: center wavelength (um).
        cross_section: cross section of the waveguide.
    """
    coupler_circuit, info = sax.circuit(
        netlist={
            "instances": {
                "bl": "bend_euler",
                "c": "coupler_ring",
                "br": "bend_euler",
            },
            "connections": {
                "bl,o2": "c,o2",
                "c,o3": "br,o1",
            },
            "ports": {
                "o1": "c,o1",
                "o2": "bl,o1",
                "o3": "br,o2",
                "o4": "c,o4",
            },
        },
        models={
            "coupler_ring": coupler_ring_coupling_area,
            "bend_euler": bend_euler,
        },
    )

    s = coupler_circuit(
        wl=wl,
        c={
            "length_x": length_x,
            "gap": gap,
            "radius": radius,
            "cross_section": cross_section,
        },
        bl={"radius": radius},
        br={"radius": radius},
    )

    return sax.reciprocal(
        {
            ("o1", "o4"): s["o1", "o4"],
            ("o1", "o3"): s["o1", "o3"],
            ("o2", "o4"): s["o2", "o4"],
            ("o2", "o3"): s["o2", "o3"],
        }
    )
