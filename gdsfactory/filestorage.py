"""Store files."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from pydantic import BaseModel
from typing_extensions import Literal

from gdsfactory.config import PATH
from gdsfactory.typings import Optional, PathType

FileTypes = Literal["sparameters", "modes", "gds", "measurements"]


class FileStorage(BaseModel):
    dirpath: Optional[PathType] = PATH.gdslib
    filetype: FileTypes

    def write(self, filename: str, data):
        raise NotImplementedError("need to implement")

    def read(self, filename: str) -> np.ndarray:
        raise NotImplementedError("need to implement")

    def _write_local_cache(self, filename: str, data):
        if not self.dirpath:
            return
        filepath = self.dirpath / f"{self.filetype}/{filename}.npz"

        with open(filepath, "wb") as file:
            np.savez_compressed(file, **data)

    def _read_local_cache(self, filename: str) -> Optional[np.ndarray]:
        if not self.dirpath:
            return
        filepath = self.dirpath / f"{self.filetype}/{filename}.npz"
        if filepath.exists():
            return np.load(filepath)


class FileStorageGoogleCloud(FileStorage):
    bucket_name: str

    def write(self, filename: str, data) -> None:
        try:
            from google.cloud import storage
        except ImportError as e:
            raise ImportError("pip install google-cloud-storage") from e

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

        try:
            from google.cloud import storage
        except ImportError as e:
            raise ImportError("pip install google-cloud-storage") from e

        filepath = f"{self.filetype}/{filename}.npz"
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(filepath)
        b = blob.download_as_bytes()
        return np.load(BytesIO(b))


# def test_google_cloud() -> None:
#     w = np.linspace(1.5, 1.6, 3)
#     s = dict(wavelengths=w)
#     s["o1@0,o2@0"] = np.zeros_like(w)

#     f = FileStorageGoogleCloud(bucket_name="gdsfactory", filetype="sparameters")
#     # f.write("demo", s)

#     field = "o1@0,o2@0"
#     s2 = f.read("demo")
#     np.isclose(s[field], s2[field])


if __name__ == "__main__":
    # test_google_cloud()

    w = np.linspace(1.5, 1.6, 3)
    field = "o1@0,o2@0"
    s = {"wavelengths": w, "o1@0,o2@0": np.zeros_like(w)}
    f = FileStorageGoogleCloud(bucket_name="gdsfactory", filetype="sparameters")
    # f.write("demo", s)

    s2 = f.read("demo")
    print(s2["o1@0,o2@0"])
