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
