import xarray
from chunkindex.core import zran_index
import numpy as np
from functools import cached_property


class Index(zran_index.Index):
    """
    The zran_xarray.Index handles a zran index stored in a xarray dataset.
    """

    def __init__(self, index: zran_index.Index | xarray.Dataset):
        """
        Create a xarray dataset that contains the zran index data.

        The xarray dataset is formatted is as follows:

        Dataset:
        --------
        dimensions:
            win = 32768
            points = points
        variables:
            bytes windows(points, win)
            int inloc(points)
            int bits(points)
        coords:
            int outloc(points)

        // global attributes:
                :uncompressed_size = uncompressed_size
                :compressed_size = compressed_size
        """

        if isinstance(index, zran_index.Index):
            # Get the outloc, inloc and bits
            outloc = [p.outloc for p in index.points]
            inloc = [p.inloc for p in index.points]
            bits = [p.bits for p in index.points]
            # Get the windows and create a 2D numpy array
            windows = np.vstack([np.frombuffer(p.window, dtype='b') for p in index.points])

            # Create the xarray dataset
            self.ds = xarray.Dataset(
                data_vars={
                    'window': (['outloc', 'win'], windows),
                    'inloc': (['outloc'], inloc),
                    'bits': (['outloc'], bits),
                },
                coords={
                    'outloc': (['outloc'], outloc),

                },
                attrs={
                    'uncompressed_size': index.uncompressed_size,
                    'compressed_size': index.compressed_size,
                    'mode': index.mode,
                    'span': index.span
                }
            )
        elif isinstance(index, xarray.Dataset):
            # TODO: check the index variables and attributes
            self.ds = index
        else:
            raise TypeError("A ZranIndex or a xarray.Dataset is required to build a ZranXarrayDataset")

    @cached_property
    def outloc(self):
        return self.ds.outloc.values

    @cached_property
    def inloc(self):
        return self.ds.inloc.values

    @cached_property
    def bits(self):
        return self.ds.bits.values

    @cached_property
    def window(self):
        return self.ds.window.values

    @cached_property
    def uncompressed_size(self):
        return self.ds.uncompressed_size

    @cached_property
    def compressed_size(self):
        return self.ds.compressed_size

    @cached_property
    def mode(self):
        return self.ds.mode

    @cached_property
    def span(self):
        return self.ds.span

    @cached_property
    def win(self):
        return self.ds.win

    def get_point(self, loc) -> zran_index.Index.Point:
        """
        Return the closest zran index point before loc from the data store in the xarray dataset.

        :param loc: location (in bytes) in the decompressed data
        :return: the closest zran index point before outloc as a ZranIndex.Point object
        """
        # Get the location of the closest index point before loc
        outloc = self.ds.outloc.sel(outloc=loc, method='ffill').values
        # Create the zran index point
        return zran_index.Index.Point(outloc=outloc,
                                      inloc=self.ds.inloc.sel(outloc=outloc).values,
                                      bits=self.ds.bits.sel(outloc=outloc).values,
                                      window=self.ds.window.sel(outloc=outloc).values.tobytes())

    def to_index(self) -> zran_index.Index:
        """
        Convert a ZranIndexDataset object to a ZranIndex object.

        :return: a ZranIndex object
        """
        # Build the index points
        points = [self.get_point(outloc) for outloc in self.ds.outloc]
        # Build the index
        return zran_index.Index(compressed_size=self.ds.compressed_size,
                                uncompressed_size=self.ds.uncompressed_size,
                                points=points)

    def to_netcdf(self, *args, **kwargs) -> bytes:
        """
        Write the zran index into a netcdf file.

        This methods is a wrapper of the xarray.Dataset.to_netcdf() method.

        :param args: see xarray.Dataset.to_netcdf()
        :param kwargs: see xarray.Dataset.to_netcdf()
        :return:
        """
        return self.ds.to_netcdf(*args, **kwargs)

    # Define __enter__() and __exit()__ methods to allow the context manager
    # i.e. allow using the with statement as follow:
    #    with zran_xarray.open_dataset(index_file) as index:
    def __enter__(self):
        return Index(self.ds.__enter__())

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.ds.__exit__(exc_type, exc_val, exc_tb)


def create_index(*args, **kwargs):
    """
    Overloads zran.Index.create_index() method.
    """
    index = zran_index.create_index(*args, **kwargs)
    return Index(index)


def open_index(*args, **kwargs):
    """
    Overloads xarray.open_dataset() method.
    """
    ds = xarray.open_dataset(*args, **kwargs)
    return Index(ds)
