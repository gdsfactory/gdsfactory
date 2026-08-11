from __future__ import annotations

__all__ = [
    "via_array_region_raster",
    "via_array_stack_oa_compliant",
    "via_stack",
    "via_stack_corner45",
    "via_stack_corner45_extended",
    "via_stack_heater_m2",
    "via_stack_heater_m3",
    "via_stack_heater_mtop",
    "via_stack_heater_mtop_mini",
    "via_stack_m1_m3",
    "via_stack_m1_mtop",
    "via_stack_m2_m3",
    "via_stack_npp_m1",
    "via_stack_slab_m1",
    "via_stack_slab_m1_horizontal",
    "via_stack_slab_m2",
    "via_stack_slab_m3",
    "via_stack_slab_npp_m3",
]

import warnings
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Literal

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.geometry import Polygon as ShapelyPolygon

import gdsfactory as gf
from gdsfactory._deprecation import deprecate
from gdsfactory.component import Component, ComponentReference, _PolygonPoints
from gdsfactory.typings import ComponentSpec, Floats, Ints, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name(tags=["vias"])
def via_stack(
    bottom_layer: LayerSpec | None = None,
    top_layer: LayerSpec | None = None,
    size: Size = (11.0, 11.0),
    via_between: ComponentSpec | None = None,
    columns: int | None = None,
    rows: int | None = None,
    *,
    layers: LayerSpecs | None = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | tuple[float | tuple[float, float], ...] | None = None,
    vias: Sequence[ComponentSpec | None] | None = ("via1", "via2", None),
    layer_to_port_orientations: dict[LayerSpec, list[int]] | None = None,
    correct_size: bool = False,
    slot_horizontal: bool = False,
    slot_vertical: bool = False,
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    http://www.vlsi-expert.com/2017/12/vias.html

    Args:
        bottom_layer: convenience 2-layer form -- if given (with top_layer),
            equivalent to layers=(bottom_layer, top_layer). Mutually
            exclusive with layers/vias.
        top_layer: see bottom_layer.
        size: of the layers. If the first positional argument is a 2-tuple
            of numbers instead of a layer, it is treated as the legacy
            `size` positional argument and this call falls back to the
            pre-IHP-convenience argument order, with a DeprecationWarning.
        via_between: via ComponentSpec to use between bottom_layer and
            top_layer. Only used together with bottom_layer/top_layer.
        columns: if set, caps the auto-fit via column count to at most
            this many columns (the fit is still auto-computed from size;
            this only ever shrinks it).
        rows: like columns, for rows.
        layers: layers on which to draw rectangles. Ignored if
            bottom_layer/top_layer are given.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size. If a tuple, it is the offset in x and y.
        vias: vias to use to fill the rectangles. Ignored if
            bottom_layer/top_layer are given.
        layer_to_port_orientations: dictionary of layer to port_orientations.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
        slot_horizontal: if True, then vias are horizontal.
        slot_vertical: if True, then vias are vertical.
        port_orientations: list of port_orientations to add. None does not add ports.
    """
    if (
        isinstance(bottom_layer, tuple)
        and len(bottom_layer) == 2
        and any(isinstance(v, float) for v in bottom_layer)
    ):
        # Legacy positional call: via_stack(size, layers, layer_offsets, vias, ...).
        deprecate(
            "via_stack(size, layers, layer_offsets, vias, ...) positional call",
            "via_stack(bottom_layer=, top_layer=, size=, ...) or explicit keywords",
        )
        legacy_positional = (top_layer, size, via_between, columns, rows)
        legacy_names = (
            "layers",
            "layer_offsets",
            "vias",
            "layer_to_port_orientations",
            "correct_size",
        )
        legacy_values = dict(zip(legacy_names, legacy_positional, strict=False))
        size = bottom_layer
        layers = legacy_values.get("layers", layers)  # type: ignore[assignment]
        layer_offsets = legacy_values.get("layer_offsets", layer_offsets)  # type: ignore[assignment]
        vias = legacy_values.get("vias", vias)  # type: ignore[assignment]
        bottom_layer = None
        top_layer = None

    if bottom_layer is not None or top_layer is not None:
        if bottom_layer is None or top_layer is None:
            raise ValueError("Pass both bottom_layer and top_layer, or neither.")
        if layers != ("M1", "M2", "MTOP") or vias != ("via1", "via2", None):
            raise ValueError(
                "Pass either (bottom_layer, top_layer[, via_between]) or "
                "(layers, vias), not both."
            )
        layers = (bottom_layer, top_layer)
        vias = (via_between,)

    layers = layers or []
    vias = vias or []

    width_m, height_m = size
    layer_indices = [gf.get_layer(layer) for layer in layers]
    layer_offsets = layer_offsets or [0] * len(layers)
    layer_to_port_orientations_list = layer_to_port_orientations or {
        gf.get_layer(layers[-1]): list(port_orientations or [])
    }

    elements = {len(layers), len(layer_offsets), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    # Determine required size from all vias BEFORE drawing metal layers
    vias_list = vias or []
    for via, offset in zip(vias_list, layer_offsets, strict=False):
        if via is not None:
            width, height = size
            if isinstance(offset, Iterable):
                offset_x = offset[0]
                offset_y = offset[1]
            else:
                offset_x = offset_y = offset
            width += 2 * offset_x
            height += 2 * offset_y

            _via = gf.get_component(via)
            if "xsize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'xsize' key in info"
                )
            if "ysize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'ysize' key in info"
                )
            if "column_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'column_pitch' key in info"
                )
            if "row_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'row_pitch' key in info"
                )

            w, h = _via.xsize, _via.ysize
            enclosure = _via.info["enclosure"]

            min_width = w + 2 * enclosure
            min_height = h + 2 * enclosure

            # Check and correct size if needed
            if correct_size and (min_width > width or min_height > height):
                corrected_width = max(min_width, width)
                corrected_height = max(min_height, height)
                warnings.warn(
                    f"Changing size from ({width}, {height}) to ({corrected_width}, {corrected_height}) to fit a via!",
                    stacklevel=3,
                )
                # Update the base size (accounting for offsets)
                width_m = max(width_m, corrected_width - 2 * offset_x)
                height_m = max(height_m, corrected_height - 2 * offset_y)
            elif min_width > width or min_height > height:
                raise ValueError(
                    f"Enclosure cannot be satisfied: size ({width}, {height}) is too small "
                    f"to fit a {(w, h)} um via with enclosure={enclosure}. "
                    f"Minimum required size is ({min_width}, {min_height})."
                )

    c = Component()
    c.info["xsize"], c.info["ysize"] = (width_m, height_m)

    # Draw metal layers with corrected size
    for layer_index, offset in zip(layer_indices, layer_offsets, strict=False):
        if isinstance(offset, Iterable):
            offset_x = offset[0]
            offset_y = offset[1]
        else:
            offset_x = offset_y = offset

        size_m = (width_m + 2 * offset_x, height_m + 2 * offset_y)

        if layer_index in layer_to_port_orientations_list:
            ref = c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type="electrical",
                port_orientations=layer_to_port_orientations_list[layer_index],
                auto_rename_ports=False,
            )
            c.add_ports(ref.ports)
        else:
            ref = c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type=None,
                port_orientations=port_orientations,
            )
        # c.absorb(ref)

    # Place vias using the corrected size
    for via, offset in zip(vias_list, layer_offsets, strict=False):
        if via is not None:
            # Use corrected width_m, height_m plus offsets
            if isinstance(offset, Iterable):
                offset_x = offset[0]
                offset_y = offset[1]
            else:
                offset_x = offset_y = offset
            width = width_m + 2 * offset_x
            height = height_m + 2 * offset_y

            _via = gf.get_component(via)
            w, h = _via.xsize, _via.ysize
            enclosure = _via.info["enclosure"]
            pitch_y = _via.info["row_pitch"]
            pitch_x = _via.info["column_pitch"]

            if slot_horizontal:
                # Check that size allows for enclosure in horizontal slot mode
                slot_via_width = width - 2 * enclosure
                if slot_via_width <= 0:
                    raise ValueError(
                        f"Enclosure cannot be satisfied in slot_horizontal mode: "
                        f"width={width}, enclosure={enclosure}. "
                        f"Need width > 2*enclosure, got {width} <= {2 * enclosure}"
                    )
                via = gf.get_component(via, size=(slot_via_width, h))
                nb_vias_x = 1
                nb_vias_y = max(1, (height - 2 * enclosure - h) / pitch_y + 1)
                # Use slot_via_width for via sizing, but keep width for positioning
                w = slot_via_width

            elif slot_vertical:
                # Check that size allows for enclosure in vertical slot mode
                slot_via_height = height - 2 * enclosure
                if slot_via_height <= 0:
                    raise ValueError(
                        f"Enclosure cannot be satisfied in slot_vertical mode: "
                        f"height={height}, enclosure={enclosure}. "
                        f"Need height > 2*enclosure, got {height} <= {2 * enclosure}"
                    )
                via = gf.get_component(via, size=(w, slot_via_height))
                nb_vias_x = max(0, (width - w - 2 * enclosure) / pitch_x + 1)
                nb_vias_y = 1
                # Use slot_via_height for via sizing, but keep height for positioning
                h = slot_via_height
            else:
                via = _via
                nb_vias_x = max(0, (width - w - 2 * enclosure) / pitch_x + 1)
                nb_vias_y = max(0, (height - h - 2 * enclosure) / pitch_y + 1)

            nb_vias_x = int(np.floor(nb_vias_x)) or 1
            nb_vias_y = int(np.floor(nb_vias_y)) or 1
            if columns is not None:
                nb_vias_x = max(1, min(nb_vias_x, columns))
            if rows is not None:
                nb_vias_y = max(1, min(nb_vias_y, rows))
            ref = c.add_ref(
                via,
                columns=nb_vias_x,
                rows=nb_vias_y,
                column_pitch=pitch_x,
                row_pitch=pitch_y,
            )

            a = width / 2
            b = height / 2
            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2

            # Verify that enclosure is respected (with small tolerance for floating point precision)
            tolerance = 1e-9
            if cw < enclosure - tolerance or ch < enclosure - tolerance:
                raise ValueError(
                    f"Enclosure violation: calculated margins (cw={cw:.3f}, ch={ch:.3f}) "
                    f"are less than required enclosure={enclosure}. "
                    f"Size ({width:.3f}, {height:.3f}) is too small for {nb_vias_x}x{nb_vias_y} "
                    f"vias of size ({w}, {h}) with pitch ({pitch_x}, {pitch_y})."
                )

            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))
    elec = [p for p in c.ports if p.port_type == "electrical"]
    if elec:
        c.create_pin(ports=elec, name="pad")
    return c


@gf.cell_with_module_name(tags=["vias"])
def via_stack_corner45(
    width: float = 10,
    layers: Sequence[LayerSpec | None] = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_port: LayerSpec | None = None,
    correct_size: bool = False,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        width: of the corner45.
        layers: layers on which to draw rectangles.
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
        correct_size: if True, if the specified dimensions are too small it increases
            them to the minimum possible to fit a via.
    """
    height = width
    layers_list = layers or []
    layer_offsets_list = layer_offsets or [0] * len(layers_list)

    elements = {len(layers_list), len(layer_offsets_list), len(vias)}
    if len(elements) > 1:
        warnings.warn(
            f"Got {len(layers_list)} layers, {len(layer_offsets_list)} layer_offsets, {len(vias)} vias",
            stacklevel=3,
        )

    if layers_list:
        layer_port = layer_port or layers_list[-1]

    c = Component()
    if layer_port:
        c.info["layer"] = layer_port

    ref: ComponentReference | None = None
    for layer, offset in zip(layers_list, layer_offsets_list, strict=False):
        if layer and layer == layer_port:
            ref = c << gf.c.wire_corner45(
                width=width + 2 * offset, layer=layer, with_corner90_ports=False
            )
            c.add_ports(ref.ports)
        elif layer is not None:
            ref = c << gf.c.wire_corner45(
                width=width + 2 * offset, layer=layer, with_corner90_ports=False
            )
    assert ref is not None

    width_corner = width
    width = ref.xsize
    height = ref.ysize
    xmin = ref.xmin
    ymin = ref.ymin

    vias_list = vias or []
    for via, offset in zip(vias_list, layer_offsets_list, strict=False):
        if via is not None:
            width45 = (
                2 * (width_corner + 2 * offset) * np.cos(np.deg2rad(45))
            )  # Width in the x direction
            _via = gf.get_component(via)
            if "xsize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'xsize' key in info"
                )
            if "ysize" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'ysize' key in info"
                )

            if "column_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'column_pitch' key in info"
                )
            if "row_pitch" not in _via.info:
                raise ValueError(
                    f"Component {_via.name!r} does not have a 'row_pitch' key in info"
                )

            w, h = _via.info["xsize"], _via.info["ysize"]
            enclosure = _via.info["enclosure"]
            pitch_x = _via.info["column_pitch"]
            pitch_y = _via.info["row_pitch"]

            via = _via

            min_width = w + 2 * enclosure
            min_height = h + 2 * enclosure

            if (min_width > width45 and correct_size) or (
                min_width <= width45 and min_height > height and correct_size
            ):
                warnings.warn(
                    f"Changing size from ({width}, {height}) to ({min_width}, {min_height}) to fit a via!",
                    stacklevel=3,
                )
                width45 = max(min_width, width45)
                height = max(min_height, height)
            elif min_width > width45 or min_height > height:
                raise ValueError(
                    f"{min_width=} > {width=} or {min_height=} > {height=}"
                )

            # Keep placing rows until we cover the whole height
            y_covered = enclosure

            while y_covered + enclosure < height:
                y = ymin + y_covered + h / 2  # Position of the via

                # x offset from the edge of the metal to make sure enclosure is fulfilled
                xoff_enc = 2 * enclosure * np.cos(np.deg2rad(45))
                xoff = (y_covered + h) * np.tan(np.deg2rad(45)) + xoff_enc

                xpos0 = xmin + xoff

                # Calculate the number of vias that fit in a given width
                if (y_covered + h) < (height - width45):
                    # The x width is width45
                    xwidth = width45
                else:
                    # The x width is decreasing
                    xwidth = (height - (y_covered + h)) * np.tan(np.deg2rad(45))

                if min_width <= xwidth:
                    vias_per_row = (
                        xwidth - 2 * xoff_enc - 2 * h * np.tan(np.deg2rad(45))
                    ) / (pitch_x) + 1
                    # Place the vias at the given x, y
                    for i in range(int(vias_per_row)):
                        ref = c << via
                        ref.center = (xpos0 + pitch_x * i + w / 2, y)

                y_covered = y_covered + h + pitch_y

    c.flatten()
    return c


@gf.cell_with_module_name(tags=["vias"])
def via_stack_corner45_extended(
    corner: ComponentSpec = "via_stack_corner45",
    via_stack: ComponentSpec = "via_stack",
    width: float = 3,
    length: float = 10,
) -> Component:
    """Rectangular via array stack at a 45 degree angle.

    Args:
        corner: corner component.
        via_stack: for the via stack.
        width: of the corner45.
        length: of the straight.
    """
    c = gf.Component()
    corner_component = c << gf.get_component(corner, width=width / np.sqrt(2))
    s = gf.get_component(via_stack, size=(length, width))
    sr = c << s
    sl = c << s
    sr.connect("e1", corner_component.ports["e1"])
    sl.connect("e1", corner_component.ports["e2"])
    return c


via_stack_m1_mtop = via_stack_m1_m3 = partial(
    via_stack,
    layers=("M1", "M2", "MTOP"),
    vias=("via1", "via2", None),
)
via_stack_m2_m3 = partial(
    via_stack,
    layers=("M2", "MTOP"),
    vias=("via2", None),
)
via_stack_slab_m1 = partial(
    via_stack,
    layers=("SLAB90", "M1"),
    vias=("viac", "via1"),
)
via_stack_slab_m2 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2"),
    vias=("viac", "via1", None),
)

via_stack_slab_m3 = partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "MTOP"),
    vias=("viac", "via1", "via2", None),
)
via_stack_npp_m1 = partial(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_slab_npp_m3 = partial(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, "viac"),
)
via_stack_heater_mtop = via_stack_heater_m3 = partial(
    via_stack, layers=("HEATER", "M2", "MTOP"), vias=(None, "via1", "via2")
)
via_stack_heater_mtop_mini = partial(via_stack_heater_mtop, size=(4, 4))

via_stack_heater_m2 = partial(via_stack, layers=("HEATER", "M2"), vias=(None, "via1"))

via_stack_slab_m1_horizontal = partial(via_stack_slab_m1, slot_horizontal=True)


def _region_to_shapely(region: _PolygonPoints) -> ShapelyPolygon:
    """Convert a _PolygonPoints region to a Shapely Polygon."""
    from kfactory import kdb

    if isinstance(region, np.ndarray):
        coords = region.tolist()
    elif isinstance(region, (kdb.DPolygon, kdb.DSimplePolygon)):
        coords = [(p.x, p.y) for p in region.each_point()]
    elif isinstance(region, kdb.Polygon):
        coords = [(p.x * 1e-3, p.y * 1e-3) for p in region.each_point()]
    elif isinstance(region, kdb.Region):
        merged = region.merged()
        poly = next(merged.each())
        coords = [(p.x * 1e-3, p.y * 1e-3) for p in poly.each_point()]
    else:
        coords = [(float(x), float(y)) for x, y in region]
    return ShapelyPolygon(coords)


@gf.cell_with_module_name(tags=["vias"])
def via_array_region_raster(
    region: _PolygonPoints = ((-5, -5), (5, -5), (5, 5), (-5, 5)),
    bottom_layer: LayerSpec = "M1",
    via_layer: LayerSpec = "VIA1",
    top_layer: LayerSpec = "M2",
    via_type: Literal["square", "rectangle"] = "square",
    via_x_spacing: float = 0.3,
    via_y_spacing: float = 0.3,
    via_x_minimum_cut_size: float = 0.3,
    via_y_minimum_cut_size: float = 0.3,
    via_x_minimum_enclosure: float = 0.06,
    via_y_minimum_enclosure: float = 0.06,
    via_x_minimum_spacing: float = 0.3,
    via_y_minimum_spacing: float = 0.3,
) -> Component:
    """Via array that fills an arbitrary polygon region with vias on a regular grid.

    Builds a meshgrid of via center positions, filters to those inside the
    region (eroded by enclosure), and places via cuts plus top/bottom metal.

    Args:
        region: polygon defining the area to fill with vias.
        bottom_layer: metal layer below the vias.
        via_layer: via cut layer.
        top_layer: metal layer above the vias.
        via_type: "square" uses via_x_minimum_cut_size for both axes.
        via_x_spacing: via spacing in x (um), must be >= via_x_minimum_spacing.
        via_y_spacing: via spacing in y (um), must be >= via_y_minimum_spacing.
        via_x_minimum_cut_size: via cut width in x (um).
        via_y_minimum_cut_size: via cut height in y (um).
        via_x_minimum_enclosure: min enclosure of via by metal in x (um).
        via_y_minimum_enclosure: min enclosure of via by metal in y (um).
        via_x_minimum_spacing: min spacing between via cuts in x (um).
        via_y_minimum_spacing: min spacing between via cuts in y (um).
    """
    assert via_x_spacing >= via_x_minimum_spacing, (
        "Bound to fail spacing DRC on x direction"
    )
    assert via_y_spacing >= via_y_minimum_spacing, (
        "Bound to fail spacing DRC on y direction"
    )
    c = Component()

    cut_w = via_x_minimum_cut_size
    cut_h = via_y_minimum_cut_size
    if via_type == "rectangle":
        minx, miny, maxx, maxy = _region_to_shapely(region).bounds
        dx = maxx - minx
        dy = maxy - miny
        if dy >= dx:
            cut_h = 2 * via_y_minimum_cut_size
        else:
            cut_w = 2 * via_x_minimum_cut_size

    enc_x = via_x_minimum_enclosure
    enc_y = via_y_minimum_enclosure

    pitch_x = cut_w + via_x_spacing
    pitch_y = cut_h + via_y_spacing

    poly = _region_to_shapely(region)

    erosion = max(enc_x + pitch_x / 2, enc_y + pitch_y / 2)
    via_region = poly.buffer(-erosion, join_style="mitre")

    if via_region.is_empty:
        c.add_polygon(list(poly.exterior.coords), layer=bottom_layer)
        c.add_polygon(list(poly.exterior.coords), layer=top_layer)
        return c

    minx, miny, maxx, maxy = via_region.bounds

    xs = np.arange(minx, maxx + pitch_x, pitch_x)
    ys = np.arange(miny, maxy + pitch_y, pitch_y)
    # xs = np.arange(minx, maxx, pitch_x)
    # ys = np.arange(miny, maxy, pitch_y)
    gx, gy = np.meshgrid(xs, ys)
    centers = np.column_stack([gx.ravel(), gy.ravel()])

    points = MultiPoint([Point(x, y) for x, y in centers])
    valid_points = points.intersection(via_region)

    if valid_points.is_empty:
        c.add_polygon(list(poly.exterior.coords), layer=bottom_layer)
        c.add_polygon(list(poly.exterior.coords), layer=top_layer)
        return c

    valid_centers = (
        np.array([[p.x, p.y] for p in valid_points.geoms])
        if hasattr(valid_points, "geoms")
        else np.array([[valid_points.x, valid_points.y]])
    )

    hw = cut_w / 2
    hh = cut_h / 2
    for cx, cy in valid_centers:
        c.add_polygon(
            [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ],
            layer=via_layer,
        )

    c.add_polygon(list(poly.exterior.coords), layer=bottom_layer)
    c.add_polygon(list(poly.exterior.coords), layer=top_layer)

    # Open Access metadata
    c.info["num_vias"] = len(valid_centers)
    c.info["via_x_spacing"] = via_x_spacing
    c.info["via_y_spacing"] = via_y_spacing
    c.info["via_x_cut_size"] = cut_w
    c.info["via_y_cut_size"] = cut_h
    c.info["via_x_enclosure"] = via_x_minimum_enclosure
    c.info["via_y_enclosure"] = via_y_minimum_enclosure
    c.info["enclosing_region"] = list(poly.exterior.coords)
    c.info["top_layer"] = top_layer
    c.info["bottom_layer"] = bottom_layer

    return c


@gf.cell_with_module_name(tags=["vias"])
def via_array_stack_oa_compliant(
    bottom_layer: LayerSpec = "M1",
    top_layer: LayerSpec = "M3",
    region: _PolygonPoints | None = None,
    size: tuple[float, float] | None = (10, 10),
    grid_size: tuple[int, int] | None = None,
    via_type: Literal["square", "rectangle"] = "square",
    via_x_minimum_cut_size_rules: dict[LayerSpec, float] | None = None,
    via_y_minimum_cut_size_rules: dict[LayerSpec, float] | None = None,
    via_x_minimum_enclosure_rules: dict[LayerSpec, float] | None = None,
    via_y_minimum_enclosure_rules: dict[LayerSpec, float] | None = None,
    via_x_minimum_spacing_rules: dict[LayerSpec, float] | None = None,
    via_y_minimum_spacing_rules: dict[LayerSpec, float] | None = None,
    layer_connectivity_sequence: LayerSpecs = (
        "M1",
        "VIA1",
        "M2",
        "VIA2",
        "M3",
    ),
) -> Component:
    """OpenAccess-compliant via stack between bottom_layer and top_layer.

    Iterates through the layer_connectivity_sequence, placing a
    via_array_region_raster for each via layer between bottom_layer and
    top_layer. Each via layer uses its own DRC rules from the per-layer
    rule dictionaries.

    The region, size, and grid_size parameters are mutually exclusive.
    Priority order: region > size > grid_size. If multiple are provided,
    only the highest-priority one is used.

    - region: arbitrary polygon coordinates defining the via area.
    - size: (width, height) in um, centered at origin.
    - grid_size: (columns, rows) of vias. The region is computed from
      the grid dimensions using the DRC rules of the first via layer.

    Args:
        bottom_layer: lowest metal layer in the stack.
        top_layer: highest metal layer in the stack.
        region: arbitrary polygon region for via placement.
        size: (width, height) rectangle centered at origin (um).
        grid_size: (columns, rows) number of vias per axis.
        via_type: "square" or "rectangle" via cuts.
        via_x_minimum_cut_size_rules: per-via-layer minimum cut width in x.
        via_y_minimum_cut_size_rules: per-via-layer minimum cut height in y.
        via_x_minimum_enclosure_rules: per-via-layer minimum metal enclosure in x.
        via_y_minimum_enclosure_rules: per-via-layer minimum metal enclosure in y.
        via_x_minimum_spacing_rules: per-via-layer minimum spacing in x.
        via_y_minimum_spacing_rules: per-via-layer minimum spacing in y.
        layer_connectivity_sequence: ordered tuple of alternating
            drawing and via layers (e.g. M1, VIA, M2, VIA1, M3, ...).
    """
    _default_cut = {"VIA": 0.3, "VIA1": 0.3, "VIA2": 0.4, "VIA3": 0.5}
    _default_enc = {"VIA": 0.06, "VIA1": 0.06, "VIA2": 0.06, "VIA3": 0.06}
    _default_spc = {"VIA": 0.3, "VIA1": 0.3, "VIA2": 0.4, "VIA3": 0.5}

    cut_x_rules = via_x_minimum_cut_size_rules or _default_cut
    cut_y_rules = via_y_minimum_cut_size_rules or _default_cut
    enc_x_rules = via_x_minimum_enclosure_rules or _default_enc
    enc_y_rules = via_y_minimum_enclosure_rules or _default_enc
    spc_x_rules = via_x_minimum_spacing_rules or _default_spc
    spc_y_rules = via_y_minimum_spacing_rules or _default_spc

    seq = list(layer_connectivity_sequence)
    drawing_layers: list[LayerSpec] = seq[0::2]
    via_layers: list[LayerSpec] = seq[1::2]

    if bottom_layer not in drawing_layers:
        raise ValueError(
            f"{bottom_layer=} not found in drawing layers {drawing_layers}"
        )
    if top_layer not in drawing_layers:
        raise ValueError(f"{top_layer=} not found in drawing layers {drawing_layers}")

    bot_idx = drawing_layers.index(bottom_layer)
    top_idx = drawing_layers.index(top_layer)
    if top_idx <= bot_idx:
        raise ValueError(
            f"top_layer {top_layer} must be above bottom_layer {bottom_layer} "
            f"in the connectivity sequence."
        )

    # Resolve mutually exclusive inputs: region > size > grid_size
    if region is not None:
        resolved_region = region
    elif size is not None:
        w, h = size
        resolved_region = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
    elif grid_size is not None:
        cols, rows = grid_size
        first_via = via_layers[bot_idx]
        cut_x = cut_x_rules.get(first_via, 0.3)
        cut_y = cut_y_rules.get(first_via, 0.3)
        enc_x = enc_x_rules.get(first_via, 0.06)
        enc_y = enc_y_rules.get(first_via, 0.06)
        spc_x = spc_x_rules.get(first_via, 0.3)
        spc_y = spc_y_rules.get(first_via, 0.3)
        w = cols * cut_x + (cols - 1) * spc_x + 2 * enc_x
        h = rows * cut_y + (rows - 1) * spc_y + 2 * enc_y
        resolved_region = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
    else:
        raise ValueError("One of region, size, or grid_size must be provided.")

    c = Component()
    total_vias = 0
    layer_info: list[dict] = []

    for i in range(bot_idx, top_idx):
        metal_below = drawing_layers[i]
        via_lyr = via_layers[i]
        metal_above = drawing_layers[i + 1]

        via_comp = via_array_region_raster(
            region=resolved_region,
            bottom_layer=metal_below,
            via_layer=via_lyr,
            top_layer=metal_above,
            via_type=via_type,
            via_x_spacing=spc_x_rules.get(via_lyr, 0.3),
            via_y_spacing=spc_y_rules.get(via_lyr, 0.3),
            via_x_minimum_cut_size=cut_x_rules.get(via_lyr, 0.3),
            via_y_minimum_cut_size=cut_y_rules.get(via_lyr, 0.3),
            via_x_minimum_enclosure=enc_x_rules.get(via_lyr, 0.06),
            via_y_minimum_enclosure=enc_y_rules.get(via_lyr, 0.06),
            via_x_minimum_spacing=spc_x_rules.get(via_lyr, 0.3),
            via_y_minimum_spacing=spc_y_rules.get(via_lyr, 0.3),
        )
        c.add_ref(via_comp)

        num = via_comp.info.get("num_vias", 0)
        total_vias += num
        layer_info.append(
            {
                "via_layer": via_lyr,
                "bottom_layer": metal_below,
                "top_layer": metal_above,
                "num_vias": num,
                "via_x_cut_size": via_comp.info.get("via_x_cut_size", 0),
                "via_y_cut_size": via_comp.info.get("via_y_cut_size", 0),
                "via_x_spacing": via_comp.info.get("via_x_spacing", 0),
                "via_y_spacing": via_comp.info.get("via_y_spacing", 0),
                "via_x_enclosure": via_comp.info.get("via_x_enclosure", 0),
                "via_y_enclosure": via_comp.info.get("via_y_enclosure", 0),
            }
        )

    # OpenAccess-compliant metadata
    c.info["bottom_layer"] = bottom_layer
    c.info["top_layer"] = top_layer
    c.info["total_num_vias"] = total_vias
    c.info["num_via_layers"] = top_idx - bot_idx
    c.info["via_type"] = via_type
    c.info["layer_connectivity_sequence"] = list(layer_connectivity_sequence)
    c.info["per_layer_info"] = layer_info
    resolved_shapely = _region_to_shapely(resolved_region)
    c.info["enclosing_region"] = list(resolved_shapely.exterior.coords)

    return c


if __name__ == "__main__":
    c = via_stack_heater_mtop_mini(size=(1, 1), correct_size=True)
    c.show()
