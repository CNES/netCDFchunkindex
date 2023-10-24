import io
from pathlib import Path
import xarray as xr
import numpy as np
import json
import kerchunk.hdf


def create_netcdf_dataset_test() -> Path:
    # Define the dataset path
    dataset_dir = Path('data')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'ramp.nc'

    # Define the datatype
    dtype = 'int32'

    # Define the number of samples in the dataset
    shape = (600, 600)
    n = np.prod(shape)
    chunk_size = tuple(int(s / 2) for s in shape)

    # Create the dataset : a ramp with n samples
    x = np.arange(n)
    x = x.reshape(shape).astype(dtype)

    # Create a data array with xarray
    x_xr = xr.DataArray(x)
    # Create a dataset
    ds = xr.Dataset({'x': x_xr})

    # Define the encoding options
    encoding = {
        'x': {
            'dtype': dtype,
            'zlib': True,
            'complevel': 1,
            'shuffle': False,
            'chunksizes': chunk_size
        }
    }
    # Write the dataset to a netcdf file
    ds.to_netcdf(dataset_path, encoding=encoding)

    return dataset_path


def create_kerchunk_index(dataset_path: Path) -> Path:

    # Define the file path of the json file in which the informations
    # about the chunks and dataset structure will be stored.
    json_path = dataset_path.with_suffix('.json')

    # Create the data structure
    h5chunks = kerchunk.hdf.SingleHdf5ToZarr(str(dataset_path)).translate()

    # Write it to a json file
    with open(json_path, 'w') as f_out:
        f_out.write(json.dumps(h5chunks))

    return json_path

