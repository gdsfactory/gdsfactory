""" """

from functools import lru_cache, partial

cache = partial(lru_cache, maxsize=None)
