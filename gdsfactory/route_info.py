from gdsfactory.serialization import clean_value_json
from gdsfactory.typings import CrossSectionSpec


def route_info(
    cs_type: str,
    length: float,
    length_eff: float | None = None,
    taper: bool = False,
    **kwargs,
) -> dict[str, float | str]:
    """Returns route info dict for pathlength analysis.

    Args:
        cs_type: cross section type.
        length: length.
        length_eff: effective length (i.e. an equivalent straight length of a bend).
        taper: True if this component is a taper.
        kwargs: other attributes to track.

    Returns:
        A dictionary of routing attributes.
    """
    if length_eff is None:
        length_eff = length

    d = {
        "type": clean_value_json(cs_type),
        "length": length_eff,
        f"{cs_type}_length": length_eff,
        "weight": length_eff,
    }
    if taper:
        d[f"{cs_type}_taper_length"] = length
    d |= kwargs
    return d


def route_info_from_cs(
    cs: CrossSectionSpec, length: float, length_eff: float | None = None, **kwargs
) -> dict[str, float | str]:
    """Returns route info dict for pathlength analysis.

    Args:
        cs: cross section object or spec.
        length: length.
        length_eff: effective length (i.e. an equivalent straight length of a bend).
        kwargs: other attributes to track.

    Returns:
        A dictionary of routing attributes.
    """
    from gdsfactory import get_cross_section

    x = get_cross_section(cs)
    cs_type = x.info.get("type", clean_value_json(cs))
    return route_info(cs_type, length=length, length_eff=length_eff, **kwargs)
