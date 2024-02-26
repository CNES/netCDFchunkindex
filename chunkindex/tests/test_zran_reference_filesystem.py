import unittest
import chunkindex
import numpy as np
import zarr
import os
import contextlib
from chunkindex.tests.create_datasets import create_netcdf_dataset_test, create_kerchunk_index


class TestZranReferenceFileSystem(unittest.TestCase):

    def setUp(self) -> None:
        # Create a test dataset
        self.dataset = create_netcdf_dataset_test()
        # Create a kerchunk index
        self.kerchunk = create_kerchunk_index(self.dataset)

        # Define the zran index path
        self.index = self.dataset.parent.joinpath(str(self.dataset.stem) + '_index.nc')
        # Remove the index file if it already exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.index)
        # Create the index as a netcdf file
        chunkindex.create_index(self.index, self.dataset)

        # Create the file system with the kerchunk and zran index
        self.fs = chunkindex.ZranReferenceFileSystem(index=self.index, fo=str(self.kerchunk))

    def test_ZranReferenceFileSystem(self):
        # Check the keys
        expected_keys = ['.zgroup',
                         'group_1/.zgroup',
                         'group_1/x/.zarray',
                         'group_1/x/.zattrs',
                         'group_1/x/0.0',
                         'group_1/x/0.1',
                         'group_1/x/1.0',
                         'group_1/x/1.1',
                         'group_1/y/.zarray',
                         'group_1/y/.zattrs',
                         'group_1/y/0.0',
                         'group_1/y/0.1',
                         'group_1/y/1.0',
                         'group_1/y/1.1',
                         'x/.zarray',
                         'x/.zattrs',
                         'x/0.0',
                         'x/0.1',
                         'x/1.0',
                         'x/1.1',
                         'y/.zarray',
                         'y/.zattrs',
                         'y/0.0',
                         'y/0.1',
                         'y/1.0',
                         'y/1.1'
                         ]
        self.assertTrue(all(k == e for k, e in zip(self.fs.get_mapper().keys(), expected_keys)))

    def test_ZranReferenceFileSystem_get_partial_values(self):

        # Open the dataset using Zarr
        ds = zarr.open(self.fs.get_mapper())

        # Access partial chunks data at the beginning of the chunk
        itemsize = 4
        start = 0
        n = 7
        offset = start*itemsize
        length = n*itemsize
        decompressed_data = np.frombuffer(
            ds.x.store.fs.get_partial_values('x/0.0', offset, length),
            dtype=np.uint32)
        self.assertTrue(np.array_equal(decompressed_data, ds.x[0, start:start+n]))

        # We access partial chunks at the end of the chunk
        n = 10
        decompressed_data = np.frombuffer(
            ds.x.store.fs.get_partial_values('x/1.1', (300 * 300 - n) * itemsize, n * itemsize),
            dtype=np.uint32)
        # print(decompressed_data)
        # print(ds.x[-1, -n:])
        self.assertTrue(np.array_equal(decompressed_data, ds.x[-1, -n:]))


if __name__ == '__main__':
    unittest.main()
