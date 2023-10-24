import zarr._storage.store
import numpy as np


class ZranStore(zarr._storage.store.Store):

    def __init__(self, mutablemapping):
        self._writeable = False
        self._mutable_mapping = mutablemapping
        self.supports_efficient_get_partial_values = True

    def __getitem__(self, key):
        return self._mutable_mapping[key]

    def get_partial_values(self, inputs):
        out = b''.join(
            [self._mutable_mapping.fs.get_partial_values(key, start, nitems) for key, (start, nitems) in inputs])
        return [bytes(out)]

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            self._mutable_mapping[key] = value.tobytes()
        else:
            self._mutable_mapping[key] = value

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._mutable_mapping)

    def __len__(self):
        return len(self._mutable_mapping)

    def __repr__(self):
        return f"<{self.__class__.__name__} object at {hex(id(self))}>"
