import unittest
import numpy as np
import zlib
import io
from pathlib import Path
from chunkindex.core import zran_h5py
from chunkindex.core import hdf
import chunkindex.core.zran_index


class TestZranXarray(unittest.TestCase):

    def setUp(self) -> None:
        # Create some compressed data
        self.data = np.arange(1e6, dtype='float64')
        self.compressed_data = zlib.compress(self.data.tobytes())
        dataset_dir = Path('data')
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = dataset_dir / 'data_index_test.nc'


        # Define the encoding options for the index windows
        WINDOW_LENGTH = chunkindex.core.zran_index.WINDOW_LENGTH
        self.encoding = {
          'window': {
            'dtype': 'int8',
            'zlib': True,
            'complevel': 1,
            'shuffle': False,
            'chunksizes': [1, WINDOW_LENGTH]
          }
        }


    def test_ZranXarray_decompress(self):
        # Create a zran index
        index = zran_h5py.create_index(self.compressed_data, span=100 * 1024)
        chunk_id = 'compressed_data/0.0'
        # Write the zran_xarray index to the netcdf file
        index.to_netcdf(self.index_path, group=chunk_id, mode="a", encoding=self.encoding)

        # Open the zran_h5py index to the netcdf file
        with open(self.index_path, mode='rb') as index:
            zindex = zran_h5py.open_index(index, group=chunk_id)
            # Use the index read to decompress some data
            offset = 100000
            length = 10
            # Decompress the data
            decompressed_data = zindex.decompress(self.compressed_data, offset=offset * 8, length=length * 8)
            decompressed_data = np.frombuffer(decompressed_data, dtype='float64')

        # Check the decompressed data
        self.assertTrue(np.array_equal(decompressed_data, self.data[offset:offset + length]))

    def test_ZranXarray_partial_decompress(self):
        # Create a zran index
        index = zran_h5py.create_index(self.compressed_data, span=100 * 1024)
        chunk_id = 'compressed_data/0.0' 
        # Write the zran_xarray index to the netcdf file
        index.to_netcdf(self.index_path, group=chunk_id, mode="a", encoding=self.encoding)

        # Open the zran_h5py index to the netcdf file
        with open(self.index_path, mode='rb') as index:
            zindex = zran_h5py.open_index(index, group=chunk_id)

            # Partial decompression of the data
            with io.BytesIO(self.compressed_data) as f:
                offset = int(zindex.outloc[-1] / 8) + 1  # Set the offset after the location of the index point
                length = 10
                decompressed_data = zindex.decompress(f, offset=offset * 8, length=length * 8)

                # Check the decompressed data
                decompressed_data = np.frombuffer(decompressed_data, dtype='float64')
                self.assertTrue(np.array_equal(decompressed_data, self.data[offset:offset + length]))


if __name__ == '__main__':
    unittest.main()
