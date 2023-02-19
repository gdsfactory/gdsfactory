from typing import Optional

import numpy as np

from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack
import gdsfactory as gf
import shapely
from shapely.affinity import translate
from shapely.ops import unary_union


class Parameter:
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        nominal_value: Optional[float] = None,
        step: Optional[float] = None,
    ) -> None:
        """Generic parameter class for training Component models.

        Arguments:
            min_value: minimum value of the parameter.
            max_value: maximum value of the parameter.
            nominal_value: nominal value of the parameter.
            step: size of the step going from min_value to max_value when generating data.
        """

    def sample(self, rand_val: float = None):
        """Return a random value within a parameter allowable values.

        User can provide their own random number between 0 and 1 (mapping to a value between min and max), or default to uniform sampling.

        Arguments:
            rand_val: random value between 0 and 1, where 0 maps to min_value and 1 to max_value.

        """
        rand_val = rand_val or np.random.rand(1)[0]
        return self.min_value + (self.max_value - self.min_value) * rand_val

    def count(self):
        """Given min, max, and step, returns number of grid points."""
        return np.ceil(np.abs(self.max_value - self.min_value) / self.step)

    def arange(self):
        """Given min, max, and step, return array of values between min and max (inclusive)."""
        return np.arange(self.min_value, self.max_value + self.step / 2, self.step)

    def corners(self):
        """Returns an array of min, nominal, and max values of the parameter."""
        return np.array([self.min_value, self.nominal_value, self.max_value])


class LayerStackThickness(Parameter):
    def __init__(
        self,
        layerstack: Optional[LayerStack] = None,
        layername: Optional[str] = "core",
        **kwargs,
    ) -> None:
        """Layerstack thickness parameter.

        Arguments:
            layerstack: LayerStack
            layername: Name of the layer in the layerstack
            min_value: minimum value of the parameter. Default to layer thickness minus tolerance.
            max_value: maximum value of the parameter. Default to layer thickness plus tolerance.
            nominal_value: nominal value of the parameter. Default to layer thickness.
            step: size of the step going from min_value to max_value when generating data. Default to 3 steps between min and max.
        """
        super().__init__(**kwargs)
        self.layerstack = layerstack or get_layer_stack()
        self.layername = layername
        self.min_value = (
            kwargs["min_value"]
            or self.layerstack.layers[self.layername].thickness
            - self.layerstack.layers[self.layername].thickness_tolerance
        )
        self.max_value = (
            kwargs["max_value"]
            or self.layerstack.layers[self.layername].thickness
            + self.layerstack.layers[self.layername].thickness_tolerance
        )
        self.nominal_value = self.layerstack.layers[self.layername].thickness
        self.step = kwargs["step"] or np.abs(self.max_value - self.min_value) / 3
        self.current_value = None
        return None


class NamedParameter(Parameter):
    def __init__(self, **kwargs) -> None:
        """Parameter associated with the Component function signature or simulation (e.g. some physical dimension, wavelength)."""
        super().__init__(**kwargs)
        self.min_value = kwargs.get("min_value")
        self.max_value = kwargs.get("max_value")
        self.nominal_value = kwargs.get("nominal_value")
        self.step = kwargs.get("step")
        return None


class LithoParameter(Parameter):
    def __init__(
        self,
        type: str = "layer_dilation_erosion",
        layerstack: Optional[LayerStack] = None,
        layername: Optional[str] = "core",
        **kwargs,
    ) -> None:
        """Parameter associated with a morphological transformation of the Component.

        Currently implemented transformations:
            * Erosion and dilation (type = "layer_dilation_erosion")
            * Layer translation offset (type = "layer_x_offset" and "layer_y_offset")
            * Corner rounding (type = "layer_round_corners")
        """
        self.min_value = kwargs.get("min_value")
        self.max_value = kwargs.get("max_value")
        self.nominal_value = kwargs.get("nominal_value")
        self.step = kwargs.get("step")
        layerstack = layerstack or get_layer_stack()

        self.layer = layerstack[layername].layer

        if type == "layer_dilation_erosion":
            self.transformation = self.layer_dilation_erosion
        elif type == "layer_x_offset":
            self.transformation = self.layer_x_offset
        elif type == "layer_y_offset":
            self.transformation = self.layer_y_offset
        elif type == "layer_round_corners":
            self.transformation = self.layer_round_corners

        return None

    def layer_dilation_erosion(self, component, dilation_value):
        temp_component = gf.Component()
        for layer, layer_polygons in component.get_polygons(by_spec=True).items():
            if layer == self.layer:
                # Make sure all layer polygons are fused properly
                shapely_polygons = [
                    shapely.geometry.Polygon(polygon) for polygon in layer_polygons
                ]
                shapely_polygons = unary_union(shapely_polygons)
                for shapely_polygon in (
                    shapely_polygons.geoms
                    if hasattr(shapely_polygons, "geoms")
                    else [shapely_polygons]
                ):
                    buffered_polygon = shapely_polygon.buffer(dilation_value)
                    temp_component.add_polygon(
                        buffered_polygon.exterior.coords, layer=layer
                    )
            else:
                for layer_polygon in layer_polygons:
                    temp_component.add_polygon(layer_polygon, layer=layer)
        temp_component.add_ports(ports=component.get_ports())
        return temp_component

    def layer_x_offset(self, component, offset_value):
        temp_component = gf.Component()
        for layer, layer_polygons in component.get_polygons(by_spec=True).items():
            for layer_polygon in layer_polygons:
                if layer == self.layer:
                    shapely_polygon = shapely.geometry.Polygon(layer_polygon)
                    translated_polygon = translate(
                        shapely_polygon, xoff=offset_value, yoff=0.0
                    )
                    temp_component.add_polygon(
                        translated_polygon.exterior.coords, layer=layer
                    )
                else:
                    temp_component.add_polygon(layer_polygon, layer=layer)
        temp_component.add_ports(ports=component.get_ports())
        return temp_component

    def layer_y_offset(self, component, offset_value):
        temp_component = gf.Component()
        for layer, layer_polygons in component.get_polygons(by_spec=True).items():
            for layer_polygon in layer_polygons:
                if layer == self.layer:
                    shapely_polygon = shapely.geometry.Polygon(layer_polygon)
                    translated_polygon = translate(
                        shapely_polygon, xoff=0.0, yoff=offset_value
                    )
                    temp_component.add_polygon(
                        translated_polygon.exterior.coords, layer=layer
                    )
                else:
                    temp_component.add_polygon(layer_polygon, layer=layer)
        temp_component.add_ports(ports=component.get_ports())
        return temp_component

    def layer_round_corners(self, component, round_value):
        temp_component = gf.Component()
        for layer, layer_polygons in component.get_polygons(by_spec=True).items():
            for layer_polygon in layer_polygons:
                if layer == self.layer:
                    shapely_polygon = shapely.geometry.Polygon(layer_polygon)
                    buffered_polygon = (
                        shapely_polygon.buffer(round_value, join_style=1)
                        .buffer(-2 * round_value, join_style=1)
                        .buffer(round_value, join_style=1)
                    )
                    temp_component.add_polygon(
                        buffered_polygon.exterior.coords, layer=layer
                    )
                else:
                    temp_component.add_polygon(layer_polygon, layer=layer)
        temp_component.add_ports(component.ports)
        return temp_component


if __name__ == "__main__":
    c = gf.Component("myComponent")
    poly1 = c.add_polygon(
        [
            [4.0, -2.5],
            [3.0, -2.5],
            [2.0, -2.5],
            [1.5, -3.0],
            [2.0, -3.5],
            [2.5, -4.0],
            [2.0, -4.5],
            [1.0, -4.5],
            [0.5, -4.0],
            [0.5, -3.0],
            [0.5, -2.0],
            [1.0, -1.5],
            [1.5, -1.0],
            [1.5, -0.0],
        ],
        layer=1,
    )
    poly2 = c.add_polygon(
        [
            [0.5, -4.5],
            [4, -4.5],
            [4, 0],
            [0.5, 0],
        ],
        layer=2,
    )
    c.show()

    # param = LithoParameter(layername="core")
    # eroded_c = param.layer_dilation_erosion(c, 0.2)
    # eroded_c.show()

    # param = LithoParameter(layername=(1,0))
    # eroded_c = param.layer_dilation_erosion(c, -0.2)
    # eroded_c.show()

    # param = LithoParameter(layername=(1,0))
    # eroded_c = param.layer_x_offset(c, 0.5)
    # eroded_c.show()

    # param = LithoParameter(layername=(1,0))
    # eroded_c = param.layer_y_offset(c, 0.5)
    # eroded_c.show()

    # param = LithoParameter(layername=(1,0))
    # eroded_c = param.layer_round_corners(c, 0.2)
    # eroded_c.show()
