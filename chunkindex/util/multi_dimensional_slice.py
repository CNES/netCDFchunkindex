import math
from typing import Iterable


def slice_intersection(s: slice, o: slice) -> slice:
    """
    Return the overlap of two slices.

    Warning: supports only slices with a step equal to 1

    :param s: slice
    :param o: other slice
    :return: True if the slices intersect.
    """
    if (s.step and s.step != 1) or (o.step and o.step != 1):
        raise NotImplementedError(f"{__name__} function only supports slices with a step of 1")
    if (s.start < o.stop and s.stop > o.start) \
            or (o.start < s.stop and o.stop > s.start):
        return slice(max(s.start, o.start), min(s.stop, o.stop), s.step)
    else:
        return slice(0, 0)


class MultiDimensionalSlice(tuple):
    """
    Multidimensional slice class.
    """

    def __new__(cls, arg: Iterable[slice]):
        """
        Implements a multidimensional slice.

        It is a tuple of slices, one slice per dimension.

        WARNING: Currently supports only slices with a step of 1.
        """

        # Check the type of the input argument
        if not isinstance(arg, Iterable):
            raise TypeError(f"{cls.__name__} expects 'Iterable', '{type(arg).__name__}' received")

        # Check the type of the items
        slices = []
        for item in arg:
            if isinstance(item, slice):
                slices.append(item)
            elif isinstance(item, tuple) and len(item) <= 3:
                slices.append(slice(*item))
            else:
                raise TypeError(f"{cls.__name__} expects an iterable of 'slice', '{type(item).__name__}' received")

        # Create a tuple of slices
        return super().__new__(cls, slices)

    @property
    def samples(self) -> int:
        """
        Return the number of samples in the multidimensional slice.

        :return: the number of samples in the multidimensional slice.
        """
        return math.prod([(s.stop - s.start) // (s.step or 1) for s in self])

    def intersection_with(self, other):
        """
        Return a multidimensional slice describing the overlap of this slice with another multidimensional slice.

        :param other: the other multidimensional slice
        :return: a multidimensional slice describing the overlap of this slice with another multidimensional slice.
        """

        # Check type
        if not isinstance(other, MultiDimensionalSlice):
            other = MultiDimensionalSlice(other)
        # Check dimensions
        if len(other) != len(self):
            raise ValueError('Dimension mismatch')

        # Return the slice overlap
        return MultiDimensionalSlice([slice_intersection(s, o) for s, o in zip(self, other)])
