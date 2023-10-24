import unittest
import chunkindex
import xarray as xr
from chunkindex.core import zran_xarray
from chunkindex.core.hdf import WINDOW_LENGTH
from chunkindex.util.multi_dimensional_slice import MultiDimensionalSlice
import os
import numpy as np
import contextlib
from chunkindex.tests.create_datasets import create_netcdf_dataset_test


class TestHdf(unittest.TestCase):

    def setUp(self) -> None:
        # Create a test dataset
        self.dataset = create_netcdf_dataset_test()
        # Define the index path
        self.index = self.dataset.parent.joinpath(str(self.dataset.stem) + '_index.nc')

        # Remove the index file if it already exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.index)

        # Create the index as a netcdf file
        chunkindex.create_index(self.index, self.dataset)

    def test_hdf_create_index(self):
        # Try to open the index file
        chunk_path = "x/0.0"
        with zran_xarray.open_index(self.index, group=chunk_path) as index:
            self.assertEqual(len(index.win), WINDOW_LENGTH)  # add assertion here

    def test_hdf_read_slice1(self):

        with open(self.dataset, 'rb') as ds:
            with open(self.index, mode='rb') as index:

                def check_read_slice(nd_slice):
                    decompressed_data = chunkindex.read_slice(ds, index, 'x', nd_slice)
                    self.assertTrue(np.array_equal(decompressed_data, xr.open_dataset(self.dataset).x[nd_slice]))

                check_read_slice(MultiDimensionalSlice((slice(0, 1), slice(0, 10))))
                check_read_slice(MultiDimensionalSlice((slice(1, 2), slice(0, 10))))
                check_read_slice(MultiDimensionalSlice((slice(300, 305), slice(300, 305))))


if __name__ == '__main__':
    unittest.main()
