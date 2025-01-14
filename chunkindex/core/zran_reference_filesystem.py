#Copyright 2025 Centre National d'Etudes Spatiales
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import io
import os
from typing import Generator, Iterable, Any
import numpy as np
import json
import xarray.backends
from chunkindex.core import zran_xarray
from fsspec.implementations.reference import ReferenceFileSystem, _protocol_groups


class ZranReferenceFileSystem(ReferenceFileSystem):
    """
    The class ZranReferenceFileSystem herits of the class fsspec.implementations.reference.ReferenceFileSystem.

    It adds the capabilities to read partial chunks using a zran index.
    """

    def __init__(self, index: str | os.PathLike[Any] | io.BufferedIOBase | xarray.backends.AbstractDataStore, **kwargs):
        """
        Create a ZranReferenceFileSystem object.

        :param index: a file path or file-like object or a xarray DataStore that contains the index data
        as a zran_xarray.Index object and that can be opened using xarray.open_dataset() method.
        :param kwargs: see fsspec.ReferenceFileSystem documentation.
        """
        self.index_path = index
        super().__init__(**kwargs)

    def get_metadata(self, path: str, attrs: Iterable[str]) -> Generator:
        """
        Returns metadata read from the .zarray file associated to the input path of a zarr-like variable.

        :param path: path to a zarr-like variable
        :param attrs: list of attributes to read from the .zarray file
        :return: a list that contains the values of the attributes
        """
        # path example: 'x/0.0' where 'x' is the variable
        var = path.split('/')[0]
        return (json.loads(self.references[var + '/.zarray'])[a] for a in attrs)

    def get_partial_values(self, path: str, offset: int, length: int) -> bytes:
        """
        Returns the data values read from the chunk defined by its path, but only from the offset and length.

        This methods returns decompressed data.

        :param path: path of the chunk in the dataset e.g.: 'x/0.0' where 'x' is the variable
        :param offset: offset from which to read the data in the chunk (in bytes)
        :param length: length of the uncompressed data chunk to retrieve (in bytes)
        :return: the data read as a byte array.
        """

        # Get the datatype of the variable read and the list of filters applied on this data
        dtype, filters = self.get_metadata(path, ('dtype', 'filters'))
        # Compute the number of bytes in on sample of data
        itemsize = np.dtype(dtype).itemsize

        # We only implement functions for the shuffle filter
        shuffle = False
        if filters:
            for f in filters:
                if f['id'] == 'shuffle':
                    shuffle = True
                    assert (f['elementsize'] == itemsize)
                elif f['id'] == 'zlib':
                    pass
                else:
                    raise NotImplementedError(f"{type(self)} only support the shuffle filter")

        out = {}
        proto_dict = _protocol_groups(path, self.references)
        for proto, paths in proto_dict.items():
            fs = self.fss[proto]
            urls, starts = [], []
            for chunk_path in paths:
                dataset_url, chunk_start, _ = self._cat_common(chunk_path)
                urls.append(dataset_url)
                starts.append(chunk_start)

            for dataset_url, chunk_start, chunk_path in zip(urls, starts, paths):
                if isinstance(dataset_url, bytes):
                    out[chunk_path] = dataset_url
                else:
                    # Open the zran index used for the partial decompression of the data
                    with zran_xarray.open_index(self.index_path, group=chunk_path) as index:
                        # Open the dataset file
                        with fs.open(dataset_url) as f:
                            # Move to the beginning of the chunk
                            f.seek(chunk_start)
                            # Decompress and read length bytes from the offset
                            bytes_out = index.decompress(f, offset, length, shuffle=shuffle, bps=itemsize)
                    out[chunk_path] = bytes_out

        if len(out) == 1:
            out = list(out.values())[0]

        return out
