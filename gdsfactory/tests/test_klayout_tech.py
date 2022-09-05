from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.klayout_tech import (
    KLayoutGroupProperty,
    KLayoutLayerProperties,
    KLayoutLayerProperty,
)
from gdsfactory.layers import LAYER_COLORS, LayerColors


def test_klayout_tech_create(
    data_regression: DataRegressionFixture, check: bool = True
) -> KLayoutLayerProperties:

    lc: LayerColors = LAYER_COLORS

    lyp = KLayoutLayerProperties(
        layers={
            layer.name: KLayoutLayerProperty(
                layer=(layer.gds_layer, layer.gds_datatype),
                name=layer.name,
                fill_color=layer.color,
                frame_color=layer.color,
                dither_pattern=layer.dither,
            )
            for layer in lc.layers.values()
        },
        groups={
            "Doping": KLayoutGroupProperty(
                members=["N", "NP", "NPP", "P", "PP", "PPP", "PDPP", "GENPP", "GEPPP"]
            ),
            "Simulation": KLayoutGroupProperty(
                members=["SIM_REGION", "MONITOR", "SOURCE"]
            ),
        },
    )

    # lyp.to_lyp("test_lyp")
    if check:
        data_regression.check(lyp.dict())

    return lyp


if __name__ == "__main__":
    lyp = test_klayout_tech_create(None, check=False)
    d = lyp.dict()
