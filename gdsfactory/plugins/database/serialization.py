from json import JSONEncoder, JSONDecoder

import numpy as np


class GdsfactoryJSONEncoder(JSONEncoder):
    """Encoder extending base :class:`.JSONEncoder` capabilities by serialising.

    * NumPy arrays
    * (WIP) GDS data
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif np.iscomplex(o):
            return {"__complex__": True, "real": np.real(o), "imag": np.imag(o)}
        elif isinstance(o, np.generic):
            return o.item()
        else:
            return JSONEncoder.default(self, o)
        # match obj:
        #     case np.ndarray:
        #         return obj.tolist()
        #     case np.generic:
        #         return obj.item()
        #     case _:
        #         return JSONEncoder.default(self, obj)


class GdsfactoryJSONDecoder(JSONDecoder):
    """TODO."""

    def __init__(self, **kwargs):
        JSONDecoder.__init__(self, object_hook=self.object_hook, **kwargs)

    def object_hook(self, dct):
        if "__complex__" in dct:
            return complex(dct["real"], dct["imag"])
        return dct
