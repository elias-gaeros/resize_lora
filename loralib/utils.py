from pathlib import Path
import logging
import json
from functools import update_wrapper


class JsonCache:
    def __init__(self, fp):
        self.fp = Path(fp)
        self.cache = {}
        self.load()

    def load(self):
        if self.fp.exists():
            logging.info("Loading %s", self.fp)
            with open(self.fp, "rt") as fd:
                self.cache = json.load(fd)

    def save(self, discard=False):
        if not self.cache:
            return
        with open(self.fp, "wt") as fd:
            json.dump(self.cache, fd)
        if discard:  # avoid double save
            self.cache = False

    def get(self, model_path):
        model_path = Path(model_path).resolve()
        return self.cache.setdefault(str(model_path), {})

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
            cache = root_cache.get(cache_name)
            if cache is None:
                cache = root_cache["name"] = {}
            cached = cache.get(key)
            if cached is None:
                cached = cache[key] = fun(self, key, *args, **kwargs)
            return cached

        return update_wrapper(wrapper, fun)

    return decorator
