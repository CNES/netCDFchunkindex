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
from chunkindex.util.multi_dimensional_slice import MultiDimensionalSlice


class TestMultiDimensionalSlice(unittest.TestCase):

    def test_MultiDimensionalSlice_typeError(self):
        # Check that the constructor returns a TypeError
        self.assertRaises(TypeError, MultiDimensionalSlice, [""])

    def test_MultiDimensionalSlice_intersection(self):
        # Check the overlap
        s1 = MultiDimensionalSlice((slice(1, 5), slice(1, 5)))
        s2 = MultiDimensionalSlice([slice(4, 8), slice(3, 4)])
        intersection = s1.intersection_with(s2)
        self.assertEqual(intersection, MultiDimensionalSlice((slice(4, 5), slice(3, 4))))

    def test_MultiDimensionalSlice_samples(self):
        # Check the number of samples in a multidimensional slice
        s = MultiDimensionalSlice((slice(1, 5), slice(1, 5)))
        self.assertEqual(s.samples, 16)
        # Check the number of samples in a multidimensional with a step greater than 1
        s = MultiDimensionalSlice((slice(1, 5, 2), slice(1, 5, 1)))
        self.assertEqual(s.samples, 8)
        # Check the number of samples in an empty multidimensional slice
        s = MultiDimensionalSlice((slice(1, 5), slice(5, 5)))
        self.assertEqual(s.samples, 0)


if __name__ == '__main__':
    unittest.main()
