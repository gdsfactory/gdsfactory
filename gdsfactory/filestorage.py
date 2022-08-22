"""Store files."""

from abc import ABC
from io import BytesIO

import numpy as np
from pydantic import BaseModel
from typing_extensions import Literal

from gdsfactory.config import CONFIG
from gdsfactory.types import Optional, PathType

FileTypes = Literal["sparameters", "modes", "gds"]


class FileStorage(ABC):
    dirpath: Optional[PathType] = CONFIG["gdslib"]
    filetype: FileTypes

    def write(self, filename: str, data):
        pass

    def read(self, filename: str) -> np.ndarray:
        pass

    def _write_local_cache(self, filename: str, data):
        if not self.dirpath:
            return
        filepath = self.dirpath / f"{self.filetype}/{filename}.npz"

        with open(filepath, "wb") as file:
            np.savez_compressed(file, **data)

    def _read_local_cache(self, filename: str, data) -> Optional[np.ndarray]:
        if not self.dirpath:
            return
        filepath = self.dirpath / f"{self.filetype}/{filename}.npz"
        if filepath.exists():
            return np.load(filepath)


class FileStorageDvc(BaseModel, FileStorage):
    """DVC data versioning control.

    The main issue is that it requires all the repo to be local.
    """

    bucket_name: str

    def write(self, filename: str, data):
        self._write_local_cache(filename=filename, data=data)

    def read(self, filename: str) -> np.ndarray:
        data = self._read_local_cache(filename=filename)
        if data:
            return data

        import dvc.api

        filepath = f"{self.filetype}/{filename}.npz"

        with dvc.api.open(filepath, "rb") as file:
            return np.load(file.read_bytes())


class FileStorageGoogleCloud(BaseModel, FileStorage):
    bucket_name: str

    def write(self, filename: str, data) -> None:
        from google.cloud import storage

        filepath = f"{self.filetype}/{filename}.npz"
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(filepath)

        with blob.open("wb", ignore_flush=True) as file:
            np.savez_compressed(file, **data)

    def read(self, filename: str) -> np.ndarray:
        data = self._read_local_cache(filename=filename)
        if data:
            return data

        from google.cloud import storage

        filepath = f"{self.filetype}/{filename}.npz"
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(filepath)
        b = blob.download_as_bytes()
        return np.load(BytesIO(b))


if __name__ == "__main__":
    # s["o1@0", "o2@0"] = np.zeros_like(w)

    w = np.linspace(1.5, 1.6, 3)
    s = dict(wavelengths=w)
    s["o1@0,o2@0"] = np.zeros_like(w)

    f = FileStorageDvc(bucket_name="gdsfactory", filetype="modes")
    # f = FileStorageGoogleCloud(bucket_name="gdsfactory", filetype="sparameters")
    f.write("demo", s)

    # s2 = f.read("demo")
    # print(s2["o1@0,o2@0"])
