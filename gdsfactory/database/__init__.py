"""Database definitions and setup."""

from __future__ import annotations

import json
from functools import partial

from sqlalchemy import create_engine

from gdsfactory.database.sql_base import metadata
from gdsfactory.database.serialization import (
    GdsfactoryJSONEncoder,
    GdsfactoryJSONDecoder,
)

## Import all possible models here, so all tables will be initialised with `create_engine`
from gdsfactory.database.models import (
    Process,
    Unit,
    Wafer,
    ComputedResult,
    Result,
    Reticle,
    ComputedResultSelfRelation,
    Die,
    ResultComputedResultRelation,
    ResultInfo,
    ResultProcessRelation,
    ResultSelfRelation,
    Component,
    Port,
    ComponentInfo,
    ResultComponentRelation,
    RelationInfo,
)
from gdsfactory.database.simulation_models import SParameterResults

try:
    pass
except ModuleNotFoundError:
    pass


create_engine = partial(
    create_engine,
    json_serializer=partial(json.dumps, cls=GdsfactoryJSONEncoder),
    json_deserializer=partial(json.loads, cls=GdsfactoryJSONDecoder),
)

__all__ = [
    "metadata",
    "create_engine",
    "Process",
    "Unit",
    "Wafer",
    "ComputedResult",
    "Result",
    "Reticle",
    "ComputedResultSelfRelation",
    "Die",
    "ResultComputedResultRelation",
    "ResultInfo",
    "ResultProcessRelation",
    "ResultSelfRelation",
    "Component",
    "Port",
    "ComponentInfo",
    "ResultComponentRelation",
    "RelationInfo",
    "SParameterResults",
    "GdsfactoryJSONEncoder",
    "GdsfactoryJSONDecoder",
]


if __name__ == "__main__":
    print(__all__)
