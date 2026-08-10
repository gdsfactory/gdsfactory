"""Statistical process variation declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _VariationBase(BaseModel):
    """Fields shared by every variation distribution."""

    model_config = ConfigDict(extra="forbid")

    sigma_basis: Literal["one_sigma", "three_sigma_norm"]
    lower_spec_limit: float | None = None
    upper_spec_limit: float | None = None
    scope: Literal["process", "mismatch"]
    random_var: str | None = None


class NormalVariation(_VariationBase):
    """Gaussian spread around an attached nominal value."""

    distribution: Literal["normal"] = "normal"
    sigma: float = Field(ge=0)
    percent: bool
    truncate_sigma: float | None = Field(default=None, ge=0)


class UniformVariation(_VariationBase):
    """Bounded uniform spread around an attached nominal value."""

    distribution: Literal["uniform"] = "uniform"
    half_range: float = Field(gt=0)


class AsymmetricVariation(_VariationBase):
    """Two-sided spread with explicit best, nominal, and worst values."""

    distribution: Literal["asymmetric"] = "asymmetric"
    best: float
    nominal: float
    worst: float


type Variation = Annotated[
    NormalVariation | UniformVariation | AsymmetricVariation,
    Field(discriminator="distribution"),
]
