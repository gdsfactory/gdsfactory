""" Upload a component / simulation result to the database """

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NamedTuple

import boto3
import boto3.session
import numpy as np
import pandas as pd
import pymysql

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm

# Models (1 to 1 match with components in db)


class ComponentModel(NamedTuple):
    function_name: str
    module: str
    name: str
    hash: str


# Other


def temporary_path(filename: str = "") -> str:
    with tempfile.NamedTemporaryFile() as file:
        path = os.path.abspath(file.name)
    if not filename:
        return path
    dirpath, _ = os.path.split(path)
    path = os.path.join(dirpath, filename)
    return path


def file_hash(path: str | Path):
    with open(path, "rb") as file:
        hex = hashlib.md5(file.read()).hexdigest()
    return hex


@contextmanager
def comp_and_yaml_temp_files(basename: str = ""):
    temppath = temporary_path(basename)
    comppath = Path(f"{temppath}.gds")
    yamlpath = Path(f"{temppath}.yml")
    try:
        yield comppath, yamlpath
    finally:
        comppath.touch()
        yamlpath.touch()
        os.remove(comppath)
        os.remove(yamlpath)


@contextmanager
def database_cursor():
    conn = pymysql.connect(
        host=os.getenv("PS_HOST", ""),
        database=os.getenv("PS_DATABASE", ""),
        user=os.getenv("PS_USERNAME", ""),
        password=os.getenv("PS_PASSWORD", ""),
        ssl_ca=os.getenv("PS_SSL_CERT", ""),
    )
    cursor = conn.cursor()
    yield cursor
    conn.commit()
    cursor.close()
    conn.close()


def select_rows(table: str, num_rows: int):
    query = f"SELECT * FROM {table} LIMIT {num_rows:.0f}"
    print(query)
    with database_cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    for row in rows:
        print(row)


def component_model(component: gf.Component):
    with comp_and_yaml_temp_files() as (comppath, yamlpath):
        component.write_gds(comppath)
        with open(yamlpath, "w") as file:
            file.write(component.to_yaml())
        function_name = component.metadata["function_name"]
        module = component.metadata["module"]
        hash = file_hash(comppath)
        name = component.name
    return ComponentModel(function_name, module, name, hash)


def upload_component(component: gf.Component, db_cursor, s3_client):
    my_component_model = component_model(component)
    fields = ", ".join(my_component_model._fields)
    values = ", ".join(f"'{c}'" for c in my_component_model)
    query = f"INSERT INTO Component ({fields}) VALUES ({values})"
    print(query)
    db_cursor.execute(query)
    with comp_and_yaml_temp_files(basename=my_component_model.hash) as (
        comppath,
        yamlpath,
    ):
        component.write_gds(comppath)
        with open(yamlpath, "w") as file:
            file.write(component.to_yaml())
        s3_client.upload_file(
            comppath, "gdslib", f"Component/{os.path.basename(comppath)}"
        )
        s3_client.upload_file(
            yamlpath, "gdslib", f"Component/{os.path.basename(yamlpath)}"
        )

def convert_to_db_format(sp: dict) -> pd.DataFrame:
    df = pd.DataFrame(sp)
    wls = df.pop("wavelengths").values
    dfs = []
    for c in df.columns:
        port_in, port_out = (p.strip() for p in c.split(","))
        values = df[c].values
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
    components = [
        gf.components.mzi(delta_length=3),
        gf.components.taper(length=100),
    ]
    s3_client = boto3.client("s3")
    with database_cursor() as cursor:
        for component in components:
            try:
                upload_component(component, cursor, s3_client)
                print(f"successfully added {component.name}.")
            except pymysql.err.IntegrityError:
                print(
                    f"unable to add {component.name}. Maybe it exists already in the database?"
                )

    comp = gf.components.taper()
    with tempfile.TemporaryDirectory() as tmpdir:
        sp = gm.write_sparameters_meep_1x1(comp, is_3d=False, dirpath=tmpdir)
        ymlpath = next(fn for fn in os.listdir(tmpdir) if fn.endswith(".yml"))
        print(ymlpath)

    df = convert_to_db_format(sp)
    print(df)
