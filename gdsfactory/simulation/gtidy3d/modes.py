"""tidy3d mode solver.

tidy3d has a powerful open source mode solver.

tidy3d can:

- compute bend modes.
- compute mode overlaps.

"""

from __future__ import annotations

import hashlib
import itertools
import pathlib
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pydantic
import tidy3d as td
import xarray
from tidy3d.plugins import waveguide
from tqdm.auto import tqdm
from typing_extensions import Literal

from gdsfactory.config import logger
from gdsfactory.pdk import MaterialSpec, get_modes_path
from gdsfactory.serialization import clean_value_name
from gdsfactory.simulation.gtidy3d.materials import get_medium
from gdsfactory.typings import PathType

Precision = Literal["single", "double"]
nm = 1e-3


class Waveguide(pydantic.BaseModel):
    """Waveguide Model.

    All dimensions must be specified in μm (1e-6 m).

    Parameters:
        wavelength: wavelength in free space.
        core_width: waveguide core width.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.
        cache: controls the use of cached results.
        overwrite: overwrite cache.

    ::

        ________________________________________________
                                                ^
                                                ¦
                                                ¦
                                          clad_thickness
                       |<--core_width-->|       ¦
                                                ¦
                       .________________.      _v_
                       |       ^        |
        <-side_margin->|       ¦        |
                       |       ¦        |
        _______________'       ¦        '_______________
              ^          core_thickness
              ¦                ¦
        slab_thickness         ¦
              ¦                ¦
              v                v
        ________________________________________________
                               ^
                               ¦
                         box_thickness
                               ¦
                               v
        ________________________________________________
    """

    wavelength: Union[float, Sequence[float], Any]
    core_width: float
    core_thickness: float
    core_material: Union[MaterialSpec, td.CustomMedium]
    clad_material: MaterialSpec
    box_material: Optional[MaterialSpec] = None
    slab_thickness: float = 0.0
    clad_thickness: Optional[float] = None
    box_thickness: Optional[float] = None
    side_margin: Optional[float] = None
    sidewall_angle: float = 0.0
    sidewall_thickness: float = 0.0
    sidewall_k: float = 0.0
    surface_thickness: float = 0.0
    surface_k: float = 0.0
    bend_radius: Optional[float] = None
    num_modes: int = 2
    group_index_step: Union[bool, float] = False
    precision: Precision = "double"
    grid_resolution: int = 20
    max_grid_scaling: float = 1.2
    cache: bool = True
    overwrite: bool = False

    _cached_data = pydantic.PrivateAttr()
    _waveguide = pydantic.PrivateAttr()

    class Config:
        """pydantic config."""

        extra = "forbid"

    @pydantic.validator("wavelength")
    def _fix_wavelength_type(cls, value):
        return np.array(value, dtype=float)

    @property
    def cache_path(self) -> Optional[PathType]:
        """Cache directory"""
        return get_modes_path()

    @property
    def filepath(self) -> Optional[pathlib.Path]:
        """Cache file path"""
        if not self.cache:
            return None
        cache_path = pathlib.Path(self.cache_path)
        cache_path.mkdir(exist_ok=True, parents=True)

        settings = [
            f"{setting}={clean_value_name(getattr(self, setting))}"
            for setting in sorted(self.__fields__.keys())
        ]

        named_args_string = "_".join(settings)
        h = hashlib.md5(named_args_string.encode()).hexdigest()[:16]
        return cache_path / f"{self.__class__.__name__}_{h}.npz"

    @property
    def waveguide(self):
        """Tidy3D waveguide used by this instance."""
        # if (not hasattr(self, "_waveguide")
        #         or isinstance(self.core_material, td.CustomMedium)):
        if not hasattr(self, "_waveguide"):
            # To include a dn -> custom medium
            if isinstance(self.core_material, td.CustomMedium):
                core_medium = self.core_material
            else:
                core_medium = get_medium(self.core_material)
            clad_medium = get_medium(self.clad_material)
            box_medium = get_medium(self.box_material) if self.box_material else None

            freq0 = td.C_0 / np.mean(self.wavelength)
            n_core = core_medium.eps_model(freq0) ** 0.5
            n_clad = clad_medium.eps_model(freq0) ** 0.5

            sidewall_medium = (
                td.Medium.from_nk(
                    n=n_clad.real, k=n_clad.imag + self.sidewall_k, freq=freq0
                )
                if self.sidewall_k != 0.0
                else None
            )
            surface_medium = (
                td.Medium.from_nk(
                    n=n_clad.real, k=n_clad.imag + self.surface_k, freq=freq0
                )
                if self.surface_k != 0.0
                else None
            )

            mode_spec = td.ModeSpec(
                num_modes=self.num_modes,
                target_neff=n_core.real,
                bend_radius=self.bend_radius,
                bend_axis=1,
                num_pml=(12, 12) if self.bend_radius else (0, 0),
                precision=self.precision,
                group_index_step=self.group_index_step,
            )

            self._waveguide = waveguide.RectangularDielectric(
                wavelength=self.wavelength,
                core_width=self.core_width,
                core_thickness=self.core_thickness,
                core_medium=core_medium,
                clad_medium=clad_medium,
                box_medium=box_medium,
                slab_thickness=self.slab_thickness,
                clad_thickness=self.clad_thickness,
                box_thickness=self.box_thickness,
                side_margin=self.side_margin,
                sidewall_angle=self.sidewall_angle,
                sidewall_thickness=self.sidewall_thickness,
                sidewall_medium=sidewall_medium,
                surface_thickness=self.surface_thickness,
                surface_medium=surface_medium,
                propagation_axis=2,
                normal_axis=1,
                mode_spec=mode_spec,
                grid_resolution=self.grid_resolution,
                max_grid_scaling=self.max_grid_scaling,
            )

        return self._waveguide

    @property
    def _data(self):
        """Mode data for this waveguide (cached if cache is enabled)."""
        if not hasattr(self, "_cached_data"):
            filepath = self.filepath
            if filepath and filepath.exists():
                if not self.overwrite:
                    logger.info(f"load data from {filepath}.")
                    self._cached_data = np.load(filepath)
                    return self._cached_data

            wg = self.waveguide

            fields = wg.mode_solver.data._centered_fields
            self._cached_data = {
                f + c: fields[f + c].squeeze(drop=True).values
                for f in "EH"
                for c in "xyz"
            }

            self._cached_data["x"] = fields["Ex"].coords["x"].values
            self._cached_data["y"] = fields["Ex"].coords["y"].values

            self._cached_data["n_eff"] = wg.n_complex.squeeze(drop=True).values
            self._cached_data["mode_area"] = wg.mode_area.squeeze(drop=True).values

            fraction_te = np.zeros(self.num_modes)
            fraction_tm = np.zeros(self.num_modes)

            for i in range(self.num_modes):
                e_fields = (
                    fields["Ex"].sel(mode_index=i),
                    fields["Ey"].sel(mode_index=i),
                )
                areas_e = [np.sum(np.abs(e) ** 2) for e in e_fields]
                areas_e /= np.sum(areas_e)
                areas_e *= 100
                fraction_te[i] = areas_e[0] / (areas_e[0] + areas_e[1])
                fraction_tm[i] = areas_e[1] / (areas_e[0] + areas_e[1])

            self._cached_data["fraction_te"] = fraction_te
            self._cached_data["fraction_tm"] = fraction_tm

            if wg.n_group is not None:
                self._cached_data["n_group"] = wg.n_group.squeeze(drop=True).values

            if filepath:
                logger.info(f"store data into {filepath}.")
                np.savez(filepath, **self._cached_data)

        return self._cached_data

    @property
    def fraction_te(self):
        """Fraction of TE polarization."""
        return self._data["fraction_te"]

    @property
    def fraction_tm(self):
        """Fraction of TM polarization."""
        return self._data["fraction_tm"]

    @property
    def n_eff(self):
        """Effective propagation index."""
        return self._data["n_eff"]

    @property
    def n_group(self):
        """Group index.

        This is only present it the parameter `group_index_step` is set.
        """
        return self._data.get("n_group", None)

    @property
    def mode_area(self):
        """Effective mode area."""
        return self._data["mode_area"]

    @property
    def loss_dB_per_cm(self):
        """Propagation loss for computed modes in dB/cm."""
        wavelength = self.wavelength * 1e-6  # convert to m
        alpha = 2 * np.pi * np.imag(self.n_eff).T / wavelength  # lin/m loss
        return 20 * np.log10(np.e) * alpha.T * 1e-2  # dB/cm loss

    @property
    def index(self) -> None:
        """Refractive index distribution on the simulation domain."""
        plane = self.waveguide.mode_solver.plane
        wavelength = (
            self.wavelength[self.wavelength.size // 2]
            if self.wavelength.size > 1
            else self.wavelength
        )
        eps = self.waveguide.mode_solver.simulation.epsilon(
            plane, freq=td.C_0 / wavelength
        )
        return eps.squeeze(drop=True).T ** 0.5

    def overlap(self, waveguide: Waveguide, conjugate: bool = True):
        """Calculate the mode overlap between waveguide modes.

        Parameters:
            waveguide: waveguide with which to overlap modes.
            conjugate: use the conjugate form of the overlap integral.
        """
        self_data = self.waveguide.mode_solver.data
        other_data = waveguide.waveguide.mode_solver.data
        # self_data = self._data
        # other_data = waveguide._data
        return self_data.outer_dot(other_data, conjugate).squeeze(drop=True).values

    def plot_grid(self) -> None:
        """Plot the waveguide grid."""
        self.waveguide.plot_grid(z=0)

    def plot_index(self, **kwargs) -> None:
        """Plot the waveguide index distribution.

        Keyword arguments are passed to xarray.DataArray.plot.
        """
        artist = self.index.real.plot(**kwargs)
        artist.axes.set_aspect("equal")

    def plot_field(
        self,
        field_name: str,
        value: str = "real",
        mode_index: int = 0,
        wavelength: float = None,
        **kwargs,
    ) -> None:
        """Plot the selected field distribution from a waveguide mode.

        Parameters:
            field_name: one of 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'.
            value: component of the field to plot. One of 'real',
                'imag', 'abs', 'phase', 'dB'.
            mode_index: mode selection.
            wavelength: wavelength selection.
            kwargs: keyword arguments passed to xarray.DataArray.plot.
        """
        data = self._data[field_name]

        if self.num_modes > 1:
            data = data[..., mode_index]
        if self.wavelength.size > 1:
            i = (
                np.argmin(np.abs(wavelength - self.wavelength))
                if wavelength
                else self.wavelength.size // 2
            )
            data = data[..., i]

        if value == "real":
            data = data.real
        elif value == "imag":
            data = data.imag
        elif value == "abs":
            data = np.abs(data)
        elif value == "dB":
            data = 20 * np.log10(np.abs(data))
            data -= np.max(data)
        elif value == "phase":
            data = np.arctan2(data.imag, data.real)
        else:
            raise ValueError(
                "value must be one of 'real', 'imag', 'abs', 'phase', 'dB'"
            )
        data_array = xarray.DataArray(
            data.T, coords={"y": self._data["y"], "x": self._data["x"]}
        )
        data_array.name = field_name
        artist = data_array.plot(**kwargs)
        artist.axes.set_aspect("equal")

    def _ipython_display_(self) -> None:
        """Show index in matplotlib for Jupyter Notebooks."""
        self.plot_index()

    def __repr__(self) -> str:
        """Show waveguide representation."""
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__fields__.keys())
            + ")"
        )

    def __str__(self) -> str:
        """Show waveguide representation."""
        return self.__repr__()


class WaveguideCoupler(Waveguide):
    """Waveguide coupler Model.

    All dimensions must be specified in μm (1e-6 m).

    Parameters:
        wavelength: wavelength in free space.
        core_width: with of each core.
        gap: inter-core separation.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.
        cache: controls the use of cached results.

    ::

        _____________________________________________________________

                ._________________.       ._________________.
                |                 |       |                 |
                |<-core_width[0]->|       |<-core_width[1]->|
                |                 |<-gap->|                 |
        ________'                 '_______'                 '________

        _____________________________________________________________



        _____________________________________________________________
    """

    core_width: Tuple[float, float]
    gap: float

    @property
    def waveguide(self):
        """Tidy3D waveguide used by this instance."""
        if not hasattr(self, "_waveguide"):
            core_medium = get_medium(self.core_material)
            clad_medium = get_medium(self.clad_material)
            box_medium = get_medium(self.box_material) if self.box_material else None

            freq0 = td.C_0 / np.mean(self.wavelength)
            n_core = core_medium.eps_model(freq0) ** 0.5
            n_clad = clad_medium.eps_model(freq0) ** 0.5

            sidewall_medium = (
                td.Medium.from_nk(
                    n=n_clad.real, k=n_clad.imag + self.sidewall_k, freq=freq0
                )
                if self.sidewall_k != 0.0
                else None
            )
            surface_medium = (
                td.Medium.from_nk(
                    n=n_clad.real, k=n_clad.imag + self.surface_k, freq=freq0
                )
                if self.surface_k != 0.0
                else None
            )

            mode_spec = td.ModeSpec(
                num_modes=self.num_modes,
                target_neff=n_core.real,
                bend_radius=self.bend_radius,
                bend_axis=1,
                num_pml=(12, 12) if self.bend_radius else (0, 0),
                precision=self.precision,
                group_index_step=self.group_index_step,
            )

            self._waveguide = waveguide.RectangularDielectric(
                wavelength=self.wavelength,
                core_width=self.core_width,
                core_thickness=self.core_thickness,
                core_medium=core_medium,
                clad_medium=clad_medium,
                box_medium=box_medium,
                slab_thickness=self.slab_thickness,
                clad_thickness=self.clad_thickness,
                box_thickness=self.box_thickness,
                side_margin=self.side_margin,
                sidewall_angle=self.sidewall_angle,
                gap=self.gap,
                sidewall_thickness=self.sidewall_thickness,
                sidewall_medium=sidewall_medium,
                surface_thickness=self.surface_thickness,
                surface_medium=surface_medium,
                propagation_axis=2,
                normal_axis=1,
                mode_spec=mode_spec,
                grid_resolution=self.grid_resolution,
                max_grid_scaling=self.max_grid_scaling,
            )

        return self._waveguide

    def coupling_length(self, power_ratio: float = 1.0) -> float:
        """Coupling length calculated from the effective mode indices.

        Args:
            power_ratio: desired coupling power ratio.
        """
        m = (self.n_eff.size // 2) * 2
        n_even = self.n_eff[:m:2].real
        n_odd = self.n_eff[1:m:2].real
        return (
            self.wavelength / (np.pi * (n_even - n_odd)) * np.arcsin(power_ratio**0.5)
        )


def _sweep(waveguide: Waveguide, attribute: str, **sweep_kwargs) -> xarray.DataArray:
    """Return an attribute for a range of waveguide geometries.

    The returned array uses the sweep arguments and the mode index as
    coordinates to organize the data.

    Args:
        waveguide: base waveguide geometry.
        attribute: desired waveguide attribute (retrieved with getattr).
        sweep_kwargs: Waveguide arguments and values to sweep.
    """
    for prohibited in ("wavelength", "num_modes"):
        if prohibited in sweep_kwargs:
            raise ValueError(f"Parameter '{prohibited}' cannot be swept.")

    kwargs = {
        k: getattr(waveguide, k) for k in waveguide.__fields__ if k not in sweep_kwargs
    }

    keys = tuple(sweep_kwargs.keys())
    values = tuple(sweep_kwargs.values())

    shape = [len(v) for v in values]
    if waveguide.wavelength.size > 1:
        shape.append(waveguide.wavelength.size)
        sweep_kwargs["wavelength"] = waveguide.wavelength.tolist()
    if waveguide.num_modes > 1:
        shape.append(waveguide.num_modes)
        sweep_kwargs["mode_index"] = list(range(waveguide.num_modes))

    variations = tuple(itertools.product(*values))
    neff = np.array(
        [
            getattr(Waveguide(**kwargs, **dict(zip(keys, values))), attribute)
            for values in tqdm(variations)
        ]
    ).reshape(shape)

    return xarray.DataArray(neff, coords=sweep_kwargs, name=attribute)


def sweep_n_eff(waveguide: Waveguide, **sweep_kwargs) -> np.ndarray:
    """Return the effective index for a range of waveguide geometries.

    The returned array uses the sweep arguments and the mode index as
    coordinates to organize the data.

    Args:
        waveguide: base waveguide geometry.

    Keyword Args:
        sweep_kwargs: Waveguide arguments and values to sweep.
        wavelength: wavelength in free space.
        core_width: waveguide core width.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.

    Example:
        >>> sweep_n_eff(
        ...     my_waveguide,
        ...     core_width=[0.40, 0.45, 0.50],
        ...     core_thickness=[0.22, 0.25],
        ... )
    """
    return _sweep(waveguide, "n_eff", **sweep_kwargs)


def sweep_fraction_te(waveguide: Waveguide, **sweep_kwargs) -> np.ndarray:
    """Return the te fraction for a range of waveguide geometries.

    Args:
        waveguide: base waveguide geometry.

    Keyword Args:
        sweep_kwargs: Waveguide arguments and values to sweep.
        wavelength: wavelength in free space.
        core_width: waveguide core width.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.

    Example:
        >>> sweep_fraction_te(
        ...     my_waveguide,
        ...     core_width=[0.40, 0.45, 0.50],
        ...     core_thickness=[0.22, 0.25],
        ... )
    """
    return _sweep(waveguide, "fraction_te", **sweep_kwargs)


def sweep_n_group(waveguide: Waveguide, **sweep_kwargs) -> np.ndarray:
    """Return the group index for a range of waveguide geometries.

    The returned array uses the sweep arguments and the mode index as
    coordinates to organize the data.

    Args:
        waveguide: base waveguide geometry.

    Keyword Args:
        sweep_kwargs: Waveguide arguments and values to sweep.
        wavelength: wavelength in free space.
        core_width: waveguide core width.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.

    Example:
        >>> sweep_n_group(
        ...     my_waveguide,
        ...     core_width=[0.40, 0.45, 0.50],
        ...     core_thickness=[0.22, 0.25],
        ... )
    """
    return _sweep(waveguide, "n_group", **sweep_kwargs)


def sweep_mode_area(waveguide: Waveguide, **sweep_kwargs) -> np.ndarray:
    """Return the mode area for a range of waveguide geometries.

    The returned array uses the sweep arguments and the mode index as
    coordinates to organize the data.

    Args:
        waveguide: base waveguide geometry.

    Keyword Args:
        sweep_kwargs: Waveguide arguments and values to sweep.
        wavelength: wavelength in free space.
        core_width: waveguide core width.
        core_thickness: waveguide core thickness (height).
        core_material: core material. One of:
            - string: material name.
            - float: refractive index.
            - float, float: refractive index real and imaginary part.
            - function: function of wavelength.
        clad_material: top cladding material.
        box_material: bottom cladding material.
        slab_thickness: thickness of the slab region in a rib waveguide.
        clad_thickness: thickness of the top cladding.
        box_thickness: thickness of the bottom cladding.
        side_margin: domain extension to the side of the waveguide core.
        sidewall_angle: angle of the core sidewall w.r.t. the substrate
            normal.
        sidewall_thickness: thickness of a layer on the sides of the
            waveguide core to model side-surface losses.
        sidewall_k: absorption coefficient added to the core material
            index on the side-surface layer.
        surface_thickness: thickness of a layer on the top of the
            waveguide core and slabs to model top-surface losses.
        surface_k: absorption coefficient added to the core material
            index on the top-surface layer.
        bend_radius: radius to simulate circular bend.
        num_modes: number of modes to compute.
        group_index_step: if set to `True`, indicates that the group
            index must also be calculated. If set to a positive float
            it defines the fractional frequency step used for the
            numerical differentiation of the effective index.
        precision: computation precision.
        grid_resolution: wavelength resolution of the computation grid.
        max_grid_scaling: grid scaling factor in cladding regions.

    Example:
        >>> sweep_mode_area(
        ...     my_waveguide,
        ...     core_width=[0.40, 0.45, 0.50],
        ...     core_thickness=[0.22, 0.25],
        ... )
    """
    return _sweep(waveguide, "mode_area", **sweep_kwargs)


def sweep_bend_mismatch(
    waveguide: Waveguide, bend_radii: Tuple[float, ...]
) -> np.ndarray:
    """Overlap integral squared for the bend mode mismatch loss.

    The loss is squared because you hit the bend loss twice
    (from bend to straight and from straight to bend).

    Args:
        waveguide: base waveguide geometry.
        bend_radii: radii values to sweep.
    """
    kwargs = dict(waveguide)
    kwargs.pop("bend_radius")
    straight = Waveguide(**kwargs)

    results = []
    for radius in tqdm(bend_radii):
        bend = Waveguide(bend_radius=radius, **kwargs)
        overlap = bend.overlap(straight)
        results.append(
            np.diagonal(overlap) ** 2 if straight.num_modes > 1 else overlap**2
        )

    return np.abs(results) ** 2


def sweep_coupling_length(
    coupler: WaveguideCoupler, gaps: Tuple[float, ...], power_ratio: float = 1.0
) -> np.ndarray:
    """Calculate coupling length for a series of gap sizes.

    Parameters:
        coupler: base waveguide coupler geometry.
        gaps: gap values to use for coupling length calculation.
        power_ratio: desired coupling power ratio.
    """
    kwargs = {k: getattr(coupler, k) for k in coupler.__fields__}
    length = []
    for gap in tqdm(gaps):
        kwargs["gap"] = gap
        c = WaveguideCoupler(**kwargs)
        length.append(c.coupling_length(power_ratio))
    return np.array(length)


if __name__ == "__main__":
    # from matplotlib import pyplot

    # for num_modes in (1, 2):
    #     for wavelength in (1.55, [1.54, 1.55, 1.56]):
    #         strip = Waveguide(
    #             wavelength=wavelength,
    #             core_width=0.5,
    #             core_thickness=0.22,
    #             slab_thickness=0.0,
    #             core_material="si",
    #             clad_material="sio2",
    #             num_modes=num_modes,
    #         )
    #         pyplot.figure()
    #         strip.plot_field(field_name="Ex", mode_index=0, wavelength=1.55, value="real")
    # rib = Waveguide(
    #     wavelength=1.55,
    #     core_width=0.5,
    #     core_thickness=0.25,
    #     slab_thickness=0.07,
    #     core_material="si",
    #     clad_material="sio2",
    #     group_index_step=True,
    #     num_modes=2,
    # )
    # print("\nRib:", rib)
    # print("Effective indices:", rib.n_eff)
    # print("Group indices:", rib.n_group)
    # print("Mode areas:", rib.mode_area)
    #
    # fig, ax = pyplot.subplots(2, rib.num_modes + 1, tight_layout=True, figsize=(12, 8))
    # rib.plot_index(ax=ax[0, 0])
    # rib.waveguide.plot_structures(z=0, ax=ax[1, 0])
    # rib.waveguide.plot_grid(z=0, ax=ax[1, 0])
    # for i in range(rib.num_modes):
    #     rib.plot_field("Ex", mode_index=i, ax=ax[0, i + 1])
    #     rib.plot_field("Ey", mode_index=i, ax=ax[1, i + 1])
    #     ax[0, i + 1].set_title(f"Mode {i}")
    # fig.suptitle("Rib waveguide")
    # # Strip waveguide coupler
    #
    # coupler = WaveguideCoupler(
    #     wavelength=1.55,
    #     core_width=(0.45, 0.45),
    #     core_thickness=0.22,
    #     core_material="si",
    #     clad_material="sio2",
    #     num_modes=4,
    #     gap=0.1,
    # )
    #
    # print("\nCoupler:", coupler)
    # print("Effective indices:", coupler.n_eff)
    # print("Mode areas:", coupler.mode_area)
    # print("Coupling length:", coupler.coupling_length())
    #
    # gaps = np.linspace(0.05, 0.15, 11)
    # lengths = sweep_coupling_length(coupler, gaps)
    #
    # _, ax = pyplot.subplots(1, 1)
    # ax.plot(gaps, lengths)
    # ax.set(xlabel="Gap (μm)", ylabel="Coupling length (μm)")
    # ax.legend(["TE", "TM"])
    # ax.grid()
    # # Strip bend mismatch
    #
    # radii = np.arange(7, 21)
    # bend = Waveguide(
    #     wavelength=1.55,
    #     core_width=0.5,
    #     core_thickness=0.25,
    #     core_material="si",
    #     clad_material="sio2",
    #     num_modes=1,
    #     bend_radius=radii.min(),
    # )
    # mismatch = sweep_bend_mismatch(bend, radii)
    #
    # fig, ax = pyplot.subplots(1, 2, tight_layout=True, figsize=(9, 4))
    # bend.plot_field("Ex", ax=ax[0])
    # ax[1].plot(radii, 10 * np.log10(mismatch))
    # ax[1].set(xlabel="Radius (μm)", ylabel="Mismatch (dB)")
    # ax[1].grid()
    # fig.suptitle("Strip waveguide bend")
    # Effective index sweep

    # wg = Waveguide(
    #     wavelength=1.55,
    #     core_width=0.5,
    #     core_thickness=0.22,
    #     core_material="si",
    #     clad_material="sio2",
    #     num_modes=2,
    #     overwrite=True
    # )

    strip = Waveguide(
        wavelength=1.55,
        core_width=1.0,
        slab_thickness=0.0,
        core_material="si",
        clad_material="sio2",
        core_thickness=220 * nm,
        num_modes=4,
    )
    w = np.linspace(400 * nm, 1000 * nm, 7)
    n_eff = sweep_n_eff(strip, core_width=w)
    fraction_te = sweep_fraction_te(strip, core_width=w)

    # t = np.linspace(0.2, 0.25, 6)
    # w = np.linspace(0.4, 0.6, 5)
    # n_eff = sweep_n_eff(wg, core_width=w, core_thickness=t)

    # fig, ax = pyplot.subplots(1, 2, tight_layout=True, figsize=(9, 4))
    # n_eff.sel(mode_index=0).real.plot(ax=ax[0])
    # n_eff.sel(mode_index=1).real.plot(ax=ax[1])
    # fig.suptitle("Effective index sweep")

    # pyplot.show()
