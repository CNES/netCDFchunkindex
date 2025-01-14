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
import unittest
import numpy as np
import zlib
import io
from chunkindex.core import zran_xarray


class TestZranXarray(unittest.TestCase):

    def setUp(self) -> None:
        # Create some compressed data
        self.data = np.arange(1e6, dtype='float64')
        self.compressed_data = zlib.compress(self.data.tobytes())

    def test_ZranXarray_decompress(self):
        # Create a zran index
        index = zran_xarray.create_index(self.compressed_data, span=100 * 1024)
        print(index)

        # Use the index read to decompress some data
        offset = 100000
        length = 10
        decompressed_data = index.decompress(self.compressed_data, offset=offset * 8, length=length * 8)
        decompressed_data = np.frombuffer(decompressed_data, dtype='float64')

        # Check the decompressed data
        self.assertTrue(np.array_equal(decompressed_data, self.data[offset:offset + length]))

    def test_ZranXarray_partial_decompress(self):
        # Create a zran index
        index = zran_xarray.create_index(self.compressed_data, span=100 * 1024)

        # Partial decompression of the data
        with io.BytesIO(self.compressed_data) as f:
            offset = int(index.outloc[-1] / 8) + 1  # Set the offset after the location of the index point
            length = 10
            decompressed_data = index.decompress(f, offset=offset * 8, length=length * 8)

            # Check the decompressed data
            decompressed_data = np.frombuffer(decompressed_data, dtype='float64')
            self.assertTrue(np.array_equal(decompressed_data, self.data[offset:offset + length]))


if __name__ == '__main__':
    unittest.main()
