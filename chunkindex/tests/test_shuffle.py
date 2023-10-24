import unittest
from chunkindex.util.shuffle import shuffle, unshuffle


class TestShuffle(unittest.TestCase):

    def test_shuffle(self):
        self.assertEqual(
            shuffle(b'112233445566', 2),
            b'123456123456')

    def test_unshuffle(self):
        self.assertEqual(
            unshuffle(b'123456123456', 2),
            b'112233445566')


if __name__ == '__main__':
    unittest.main()
