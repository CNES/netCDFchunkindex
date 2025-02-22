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
from pathlib import Path
import xarray as xr
import numpy as np
import json
import kerchunk.hdf
import os
import contextlib


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
    y = x.copy()
    y[0:10,2] = -999
    
    # Create a data array with xarray
    x_xr = xr.DataArray(x)
    y_xr = xr.DataArray(y)
    # Create a dataset
    ds = xr.Dataset({'x': x_xr, 'y': y_xr,})

    # Add attribute to variable y
    ds['y'].attrs = {'_FillValue': -999, 'scale_factor': 2, 'add_offset': 100}

    # Define the encoding options
    encoding = {
        'x': {
            'dtype': dtype,
            'zlib': True,
            'complevel': 1,
            'shuffle': False,
            'chunksizes': chunk_size
        },
        'y': {
            'dtype': dtype,
            'zlib': True,
            'complevel': 1,
            'shuffle': False,
            'chunksizes': chunk_size
        }
    }

    # Remove it if it already exists
    with contextlib.suppress(FileNotFoundError):
        os.remove(dataset_path)

    # Write the dataset to a netcdf file
    ds.to_netcdf(dataset_path, encoding=encoding)

    # Write the same dataset but in group "group_1"
    ds.to_netcdf(dataset_path, group="group_1", mode="a", encoding=encoding)

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
