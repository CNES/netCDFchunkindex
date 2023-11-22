import xarray
import h5py as h5
from chunkindex.core import zran_index
import numpy as np
from functools import cached_property


class Index(zran_index.Index):
    """
    The zran_h5py.Index handles a zran index stored in a h5py dataset.
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
                    'mode': index.mode
                }
            )
        elif isinstance(index, h5._hl.group.Group):
            # TODO: check the index variables and attributes
            self.ds = index
        else:
            raise TypeError("A ZranIndex or a xarray.Dataset is required to build a ZranXarrayDataset")

    @cached_property
    def outloc(self):
        return self.ds['outloc'][:]

    @cached_property
    def inloc(self):
        return self.ds['inloc'][:]

    @cached_property
    def bits(self):
        return self.ds['bits'][:]

    @cached_property
    def window(self):
        return self.ds['window'][:]

    @cached_property
    def uncompressed_size(self):
        return self.ds.attrs['uncompressed_size'][0]

    @cached_property
    def compressed_size(self):
        return self.ds.attrs['compressed_size'][0]

    @cached_property
    def mode(self):
        return self.ds.attrs['mode'][0]

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
        outloc = np.searchsorted(self.ds['outloc'][:], loc, side='right') - 1
        # Create the zran index point
        return zran_index.Index.Point(outloc=self.ds['outloc'][outloc],
                                      inloc=self.ds['inloc'][outloc],
                                      bits=self.ds['bits'][outloc],
                                      window=self.ds['window'][outloc].tobytes())

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
    # Remove __enter__ method because not available on h5py

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
    Overloads h5.File() method.
    """
    ds = h5.File(*args)[kwargs['group']]
    return Index(ds)
