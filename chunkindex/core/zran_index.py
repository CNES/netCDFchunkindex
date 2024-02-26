import numpy as np
import zran
import bisect
from chunkindex.util.shuffle import unshuffle
from functools import lru_cache, cached_property
from typing import BinaryIO

MODE_ZLIB = 15
WINDOW_LENGTH = 32768


def read_offset(f: BinaryIO, offset: int, length: int, whence: 0 | 1 | 2 = 0) -> bytes:
    """
    Read data from file starting from an offset.

    :param f: input file object
    :param offset: the offset in bytes
    :param length: the length of data to read in bytes
    :param whence: (Optional) Whether the `offset` is relative to the beginning of the input file `f` (default value 0),
    or relative to the current position (1) or relative to the file’s end (2).
    :return: the length bytes of data read.
    """

    # Record the pointer location
    last_pos = f.tell()

    # Move the file pointer to the offset
    f.seek(offset, whence)
    # Read length bytes
    data = f.read(length)

    # Restore the pointer location
    f.seek(last_pos)

    return data


class Index(zran.Index):
    """
    The zran_index.Index class is an interface to the zran.Index class.

    It adds shuffle and partial decompression features.
    """

    class Point(zran.Point):
        """
        Inner class of the ZranIndex, interface to the zran.Point class.
        """
        pass

    def __init__(self, compressed_size: int, uncompressed_size: int, points: list[zran.Point], span: int):
        """
        Create a ZranIndex index object.

        An index point at the very beginning of the compressed data is inserted if not provided in the list of points.

        :param points: a list of zran.Point objects
        :param uncompressed_size: the uncompressed size of the data to which this index applies
        :param compressed_size: the compressed size of the data to which this index applies
        :return: a ZranIndex object
        """

        # Create the first index point if it is not the one that is currently read
        if all(p.outloc != 0 for p in points):
            points.insert(0, Index.Point(outloc=0, inloc=2, bits=0, window=np.zeros(WINDOW_LENGTH, dtype='B')))

        # Create the zran.Index object
        super().__init__(mode=MODE_ZLIB, compressed_size=compressed_size, uncompressed_size=uncompressed_size,
                         have=len(points), points=points)

        # Get the location of the index points in the uncompressed stream (out) and compressed stream (in)
        self.outloc = [p.outloc for p in self.points]
        self.inloc = [p.inloc for p in self.points]
        self.span = span

    def get_point(self, loc: int) -> zran.Point:
        """
        Return the closest zran index point before loc.

        :param loc: location (in bytes) in the decompressed data
        :return: the closest zran index point before loc
        """
        # Find the index points the closest before the offset
        lo = bisect.bisect(self.outloc, loc) - 1
        return self.points[lo]

    # Read and decompress only the required amount of data
    def decompress(self, f: BinaryIO | bytes, offset: int, length: int, whence: int = 1,
                   shuffle: bool = False, bps: int = None) -> bytes:
        """
        Partially decompress a binary stream using a zran index.

        This function reads a chunk of compressed data in the input file object `f` and perform the decompression
        to retrieve `length` bytes of the uncompress data from the offset `offset`.

        :param f: input file object containing the data compressed with deflate
        :param offset: offset from which to retrieve the uncompressed data (in bytes)
        :param length: length of the uncompressed data chunk to retrieve (in bytes)
        :param whence: (Optional) Whether the `offset` is relative to
         - the beginning of the input file `f` (0),
         - the current position (default=1)
         - the file’s end (2).
        :param shuffle: whether the shuffle filter has been applied before the data compression
        :param bps: (only required if shuffle=True) number of bytes per sample in the data
        :return: the uncompressed byte array.
        """

        # If shuffle is True, make call to the decompress_shuffle() method
        if shuffle:
            if not bps:
                raise ValueError('bps is required when shuffle = True')
            return self.decompress_shuffle(f, int(offset / bps), int(length / bps), bps)

        # Get the starting index point
        starting_point = self.get_point(offset)

        # Compute the offsets corresponding to the starting access point
        # in the decompressed (out) and compressed (in) data
        offset_out = starting_point.outloc
        # Keep one more byte if some bits are required from the previous byte
        offset_in = starting_point.inloc - int(starting_point.bits > 0)

        # Create a new index point with modified offset
        new_index_point = Index.Point(inloc=int(starting_point.bits > 0), outloc=0,
                                      bits=starting_point.bits, window=starting_point.window)

        # Create an index object
        index = Index(points=[new_index_point],
                      compressed_size=int(self.compressed_size - offset_in),
                      uncompressed_size=int(self.uncompressed_size - offset_out), span=self.span)

        # Find the location in the compressed data of the index point after the data we want to retrieve,
        # i.e. after offset + length in the uncompressed data
        hi = bisect.bisect(self.outloc, offset + length)
        # Compute the compressed data length we need to read between the two access points
        # Note: we will read up to the end of the compressed file if we go beyond the last index point
        end_in = self.inloc[hi] if hi < len(self.inloc) else self.compressed_size
        length_in = end_in - offset_in

        # Read compressed data from the input file
        if isinstance(f, bytes):
            compressed_data = f[offset_in:offset_in+length_in]
        else:
            compressed_data = read_offset(f, offset_in, length_in, whence=whence)

        # Decompress the data read using the index
        return zran.decompress(compressed_data, index, offset - offset_out, int(length))

    def decompress_shuffle(self, f: BinaryIO, offset: int, length: int, bps: int) -> bytes:
        """
        Decompress a chunk of data applying un-shuffling if required.

        :param f:          open file object pointer set at the beginning of the chunk
        :param offset:     offset of the area we want to decompress in that chunk (in bytes)
        :param length:     length of the area we want to decompress in that chunk (in bytes)
        :param bps:        number of bytes per sample in the data
        :return: the decompressed binary array
        """

        # Compute the number of samples in the chunk
        n_samples = int(self.uncompressed_size / bps)

        # Create the data stream
        binary_stream = b''
        # Decompress each portion of the data stream, one portion per byte
        for i in range(bps):
            # Decompress the data
            binary_stream += self.decompress(f, offset, length, whence=1)
            offset += n_samples

        # Do unshuffling
        return unshuffle(binary_stream, bps)


def create_index(*args, **kwargs):
    """
    Overloads zran.Index.create_index() method.
    """
    index = zran.Index.create_index(*args, **kwargs)
    return Index(compressed_size=index.compressed_size,
                 uncompressed_size=index.uncompressed_size,
                 points=index.points, span=kwargs['span'])
