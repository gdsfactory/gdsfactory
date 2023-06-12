""" Upload a component / simulation result to the database """

import hashlib
import os
import tempfile
from functools import lru_cache
from typing import List, Optional

import boto3
import boto3.session
import numpy as np
import pandas as pd
from sqlmodel import Field
from sqlmodel import Session as _Session
from sqlmodel import SQLModel, create_engine

import gdsfactory as gf


class Session(_Session):
    def safe_add(self, model: SQLModel):
        """adds a model to the database, but ignores it if it's already in there."""
        return self.execute(model.__class__.__table__.insert().prefix_with("IGNORE").values([model.dict()]))  # type: ignore

    def safe_add_all(self, models: List[SQLModel]):
        """adds a model to the database, but ignores it if it's already in there."""
        cls = models[0].__class__
        assert all(model.__class__ is cls for model in models)
        return self.execute(cls.__table__.insert().prefix_with("IGNORE").values([model.dict() for model in models]))  # type: ignore


class Component(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    function_name: str = Field(min_length=1, max_length=20)
    module: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=40)
    hash: str = Field(min_length=32, max_length=32, unique=True)


class Simulation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    function_name: str = Field(min_length=1, max_length=20)
    hash: str = Field(min_length=32, max_length=32, unique=True)
    component_hash: str = Field(min_length=32, max_length=32, unique=True)
    wavelength: float
    port_in: str = Field(min_length=1, max_length=10)
    port_out: str = Field(min_length=1, max_length=10)
    abs: float
    angle: float


@lru_cache(maxsize=None)
def get_database_engine():
    host = os.getenv("PS_HOST", "")
    database = os.getenv("PS_DATABASE", "")
    username = os.getenv("PS_USERNAME", "")
    password = os.getenv("PS_PASSWORD", "")
    ssl_ca = os.getenv("PS_SSL_CERT", "")
    connection_string = f"mysql+pymysql://{username}:{password}@{host}/{database}"
    return create_engine(
        connection_string, echo=True, connect_args={"ssl": {"ca": ssl_ca}}
    )


@lru_cache(maxsize=None)
def s3_client():
    return boto3.client("s3")


def get_component_hash(component: gf.Component) -> str:
    with tempfile.NamedTemporaryFile() as file:
        path = os.path.abspath(file.name)
        component.write_gds(path)
        return hashlib.md5(file.read()).hexdigest()


def get_s3_key_from_hash(prefix: str, hash: str, ext: str = "gds") -> str:
    return os.path.join(f"{prefix}/{ext}/{hash}.{ext}")


def convert_to_db_format(sp: dict) -> pd.DataFrame:
    df = pd.DataFrame(sp)
    wls = df.pop("wavelengths").values
    dfs = []
    for c in df.columns:
        port_in, port_out = (p.strip() for p in c.split(","))
        cdf = pd.DataFrame(
            {
                "wavelength": wls,
                "port_in": port_in,
                "port_out": port_out,
                "abs": np.abs(df[c].values),
                "angle": np.angle(df[c].values),
            }
        )
        dfs.append(cdf)

    return pd.concat(dfs, axis=0)


if __name__ == "__main__":
    import gdsfactory.simulation.gmeep as gm

    component = gf.components.taper(length=100)
    component_yaml = component.to_yaml()

    component_model = Component(
        function_name=component.metadata["function_name"],
        module=component.metadata["module"],
        name=component.name,
        hash=get_component_hash(component),
    )
    engine = get_database_engine()
    with Session(engine) as session:
        session.safe_add(component_model)
        session.commit()

    s3 = s3_client()
    with tempfile.TemporaryDirectory() as tempdir:
        component_gds_path = os.path.join(tempdir, f"{component_model.hash}.gds")
        component_yaml_path = os.path.join(tempdir, f"{component_model.hash}.yml")
        component.write_gds(component_gds_path)
        with open(component_yaml_path, "w") as file:
            file.write(component_yaml)

        component_key = get_s3_key_from_hash("component", component_model.hash, "gds")
        component_yaml_key = get_s3_key_from_hash(
            "component", component_model.hash, "yml"
        )
        s3.upload_file(component_gds_path, "gdslib", component_key)
        s3.upload_file(component_yaml_path, "gdslib", component_yaml_key)

    # with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = "/tmp"
    tmppath = gm.write_sparameters_meep_1x1(
        component, is_3d=False, dirpath=tmpdir, only_return_filepath_sim_settings=True
    )  # TODO: split simulation yaml file generation and actual simulation...
    simulation_hash = str(tmppath)[-36:-4]
    sp = gm.write_sparameters_meep_1x1(component, is_3d=False, dirpath=tmpdir)
    yaml_filename = next(fn for fn in os.listdir(tmpdir) if fn.endswith(".yml"))
    yaml_path = os.path.join(tmpdir, yaml_filename)
    df = convert_to_db_format(sp)
    df["component_hash"] = component_model.hash
    df["hash"] = simulation_hash
    df["function_name"] = "write_sparameters_meep_1x1"
    df = df[
        [
            "function_name",
            "hash",
            "component_hash",
            "wavelength",
            "port_in",
            "port_out",
            "abs",
            "angle",
        ]
    ]
    simulation_models = []
    for (
        function_name,
        hash,
        component_hash,
        wavelength,
        port_in,
        port_out,
        abs,
        angle,
    ) in df.values:
        simulation_model = Simulation(
            function_name=function_name,
            hash=hash,
            component_hash=component_hash,
            wavelength=wavelength,
            port_in=port_in,
            port_out=port_out,
            abs=abs,
            angle=angle,
        )
        simulation_models.append(simulation_model)
    with Session(engine) as session:
        session.safe_add_all(simulation_models)
        session.commit()

    s3 = s3_client()
    yaml_key = get_s3_key_from_hash("simulation", simulation_hash, "yml")
    s3.upload_file(yaml_path, "gdslib", yaml_key)
