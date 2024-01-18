import os
from typing import Iterable, BinaryIO

import h5py
import numpy as np
import chunkindex.core.zran_xarray
import chunkindex.core.zran_h5py
import chunkindex.core.zran_index
from chunkindex.util.multi_dimensional_slice import MultiDimensionalSlice

SPAN = 102400  # 100kB
WINDOW_LENGTH = chunkindex.core.zran_index.WINDOW_LENGTH
MASKANDSCALE = True
METHOD = 'h5py'


def chunkid_str(chunk_offset: tuple[int], chunk_size: tuple[int]) -> str:
    return '.'.join([str(int(o / s)) for o, s in zip(chunk_offset, chunk_size)])


def create_index(index_path: str | os.PathLike, dataset: str | os.PathLike, span=SPAN) -> None:
    """
    Build the indexes for each chunk of each variable of the input dataset and write the index generated into a netcdf
    file.

    :param index_path: a file path or file-like object in which the index will be written as a netcdf file
    :param dataset: a file path or file-like object of the NetCDF-4/HDF5 dataset.
    :param span: number of uncompressed bytes between the index points. Default: 100kB
    """

    if os.path.exists(index_path):
        raise FileExistsError(f"{index_path} already exists")

    # Define the encoding options for the index windows
    encoding = {
        'window': {
            'dtype': 'int8',
            'zlib': True,
            'complevel': 1,
            'shuffle': False,
            'chunksizes': [1, WINDOW_LENGTH]
        }
    }

    def _index_variable(name, item):

        # Do not process groups
        if isinstance(item, h5py.Group):
            return

        # Do not process variables that are not dimensions
        if item.is_scale:
            return

        #  Only process regular variables

        # Get the chunk size for that variable
        chunk_size = item.chunks
        # Get the variable id to access low-level HDF5 API and iterate over its chunks
        dsid = item.id

        def gen_index(chunk):
            # Read the compressed data from the chunk
            compressed_data = dsid.read_direct_chunk(chunk.chunk_offset)[1]
            # Create the zran index for that chunk with one index point every SPAN uncompressed bytes
            index = chunkindex.core.zran_xarray.create_index(compressed_data, span=span)
            # Define the name of the chunk: e.g. var/0.1
            chunk_id = name + '/' + chunkid_str(chunk.chunk_offset, chunk_size)
            # Write the zran_xarray index to the netcdf file
            index.to_netcdf(index_path, group=chunk_id, mode="a", encoding=encoding)

        # Loop over all the chunks in that variable
        dsid.chunk_iter(gen_index)

    # Open the netCDF dataset
    with h5py.File(dataset) as ds:
        # Create an index for each variable in the dataset
        ds.visititems(_index_variable)


def read_slice(dataset: BinaryIO, index: BinaryIO, var: str,
               nd_slice: MultiDimensionalSlice | Iterable[slice] | Iterable[tuple],
               maskandscale=MASKANDSCALE, method: str=METHOD):
    """
    Read a slice of data from within a variable in a HDF5 dataset.

    This function makes use of zran index to read and uncompress partial chunks of data.

    Usage example:

        with open("dataset.nc", mode='rb') as ds:
            with open("dataset_index.nc", mode='rb') as index:
                chunkindex.read_slice(ds, index, "my_group/my_nc_variable", ((300, 305), (300, 305)))

    :param dataset: the opened file object of the NetCDF-4/HDF5 dataset.
    :param index: an opened file object that contains the index data in netCDF-4 format.
    :param var: the name of the dataset variable we want to access to.
    :param nd_slice: slice or multidimensional slice corresponding to the data to access in the variable `var`.
    :param maskandscale: turn on or off automatic conversion of data (apply scale_factor and add_offset) and masked Fillvalue
    :param method: select which lib to use h5py or xarray
    :return: the slice of data read.
    """

    if not isinstance(nd_slice, MultiDimensionalSlice):
        nd_slice = MultiDimensionalSlice(nd_slice)

    # Open the HDF5 dataset
    with h5py.File(dataset) as ds:

        # Get the dataset variable var
        dsvar = ds[var]

        # If maskandscale is activated get attribute
        if maskandscale:
            # Get the list of attribute of variable var
            liste_att = dsvar.attrs.keys()
            # get _FillValue attribute if exists
            if '_FillValue' in liste_att:
                fillvalue = dsvar.attrs['_FillValue'][0]
            else:
                fillvalue = False
            # get scale_factor attribute if exists
            if 'scale_factor' in liste_att:
                scale_factor = dsvar.attrs['scale_factor'][0]
            else:
                scale_factor = 1
            # get offset attribute if exists
            if 'add_offset' in liste_att:
                offset = dsvar.attrs['add_offset'][0]
            else:
                offset = 0

        # Get the chunks shape
        chunk_size = dsvar.chunks

        # Get the number of bytes in one sample (bytes-per-sample)
        bps = np.dtype(dsvar.dtype).itemsize

        # Get the chunks locations in the dataset as slices
        chunks_slices = list(dsvar.iter_chunks())

        # Compute the intersection of the chunks with the ndslice we want to read
        chunks_slice_intersections = [MultiDimensionalSlice(s).intersection_with(nd_slice) for s in chunks_slices]
        # Get the id of the chunks for which the intersection is not empty
        chunks_to_read = np.where([inter.samples > 0 for inter in chunks_slice_intersections])[0]

        # Get the variable id to access low-level HDF5 API and iterate over its chunks
        dsid = dsvar.id

        # Create an array that fake that variable
        fake_var = np.empty(dsvar.shape, dsvar.dtype)

        # Loop over all the chunks of that variable
        for chunk_id in chunks_to_read:

            # Create an array that fake a chunk in that variable
            fake_chunk = np.empty(chunk_size, dsvar.dtype)

            # Get the chunk offset
            chunk_info = dsid.get_chunk_info(chunk_id)
            chunk_offset = chunk_info.chunk_offset

            # Identify the slice to read in that chunk
            chunk_slice_intersection = chunks_slice_intersections[chunk_id]

            # Transpose this intersection relative to the origin of the chunk
            # to get the location of the slice in the chunk
            chunk_slice = chunks_slices[chunk_id]
            chunk_origin = [s.start for s in chunk_slice]
            slice_in_chunk = MultiDimensionalSlice(
                [slice(s.start - o, s.stop - o, s.step) for s, o in zip(chunk_slice_intersection, chunk_origin)])

            # Compute the index of the slice in the chunk
            slice_indices = np.arange(0, np.prod(chunk_size)).reshape(chunk_size)[slice_in_chunk].flatten()
            offset_in_chunk = slice_indices[0]
            length_in_chunk = slice_indices[-1] - offset_in_chunk + 1

            # ---------------------------------------------------------------------------------------------------------
            # NOTE:
            # By computing an offset and a length rather than utilizing only the indices identified in the list
            # slice_indices, we make the choice to read all the sample from the first indice up to the last indice
            # even if there are some sample that are not need (i.e. samples that are not identified in the list
            # slice_indices). We make this choice to avoid multiple calls to the function decompress_shuffle().
            # Indeed, each call to this function may imply an HTTP request with some latency. To minimize the latency
            # we prefer minimizing the number of call to this function even if this implies downloading a few more data
            # than strictly necessary.
            # This behavior may change in the future.
            # ---------------------------------------------------------------------------------------------------------

            # Move the file pointer to the beginning of the chunk
            dataset.seek(chunk_info.byte_offset)

            # Define the name of the group in the index: e.g. var/0.1
            index_group = var + '/' + chunkid_str(chunk_offset, chunk_size)
            # Open the index
            if method == 'h5py':
                zindex = chunkindex.core.zran_h5py.open_index(index, group=index_group)
            else:
                zindex = chunkindex.core.zran_xarray.open_index(index, group=index_group)
            # Decompress the data
            decompressed_byte_array = zindex.decompress(dataset, offset_in_chunk*bps, length_in_chunk*bps,
                                                            shuffle=dsvar.shuffle, bps=bps, whence=1)

            # Fill the fake chunk with the decompressed data
            if 'decompressed_byte_array' in locals():
                fake_chunk = fake_chunk.flatten()
                fake_chunk[range(offset_in_chunk, offset_in_chunk + length_in_chunk)] = np.frombuffer(
                    decompressed_byte_array, dtype=dsvar.dtype)
                fake_chunk = fake_chunk.reshape(chunk_size)
                fake_var[chunk_slice_intersection] = fake_chunk[slice_in_chunk]

    # Apply scale_factor, offset and mask Fillvalue data
    if maskandscale:
        if fillvalue:
            # Apply mask and scaling
            fake_var = np.ma.masked_where(fake_var==fillvalue, fake_var)*scale_factor + offset
        else:
            # No fillvalue, apply only scaling
            fake_var = fake_var*scale_factor + offset


    return fake_var[nd_slice]
