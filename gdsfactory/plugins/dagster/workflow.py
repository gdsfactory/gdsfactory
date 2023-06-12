from typing import Dict

from dagster import ConfigurableResource, Definitions, asset


class ReaderResource(ConfigurableResource):
    value: str


@asset
def design(reader: ReaderResource) -> Dict:
    # read_based_on_config()
    return {"design:": reader.value}


@asset
def verification():
    return {"verification:": 1}


@asset
def manufacturing():
    return {"manufacturing:": 1}


@asset
def validation():
    return {"validation:": 1}


@asset
def structure_layer(
    context,
    design,
    verification,
    manufacturing,
    validation,
):
    combined = {
        **design,
        **verification,
        **manufacturing,
        **validation,
    }
    context.log.info(f"Combined data {combined}")
    return combined


defs = Definitions(
    assets=[
        design,
        verification,
        manufacturing,
        validation,
        structure_layer,
    ],
    resources={"reader": ReaderResource(value="configured-value")},
)
