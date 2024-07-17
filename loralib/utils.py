from pathlib import Path
import logging
import json
from collections import defaultdict
from functools import update_wrapper


class JsonCache(dict):
    __slot__ = ("fp", "factory")

    def __init__(self, fp, factory=lambda x, *args: defaultdict(dict, *args)):
        super().__init__
        self.fp = Path(fp)
        self.factory = factory
        self.load()

    def __missing__(self, key):
        self[key] = value = self.factory(key)
        return value

    def load(self):
        if self.fp.exists():
            logging.info("Loading %s", self.fp)
            with open(self.fp, "rt") as fd:
                data = json.load(fd)
            for k, v in data.items():
                self[k] = self.factory(k, v)

    def save(self, discard=False):
        if not len(self):
            return
        with open(self.fp, "wt") as fd:
            json.dump(self, fd)
        if discard:  # avoid double save
            self.clear()

    def __del__(self):
        self.save(discard=True)


def cached(cache_name):
    """
    Decorator to cache method results in self._cache.

    This decorator caches the results of a method in `self._cache` under a
    specified cache name. Only the first argument of the method is used as the
    cache key.

    Args:
        cache_name (str): The name of the cache within `self.cache`.

    Example:
        class MyClass:
            def __init__(self):
                self._cache = {}

            @cached("my_cache")
            def expensive_method(self, key, *args, **kwargs):
                # Perform expensive computation
                return result
    """

    def decorator(fun):
        def wrapper(self, key, *args, **kwargs):
            root_cache = self._cache
            cache = root_cache[cache_name]
            if cache is None:
                cache = root_cache["name"] = {}
            cached = cache.get(key)
            if cached is None:
                cached = cache[key] = fun(self, key, *args, **kwargs)
            return cached

        return update_wrapper(wrapper, fun)

    return decorator
