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
import chunkindex
import argparse
import datetime
from pathlib import Path
import urllib
import xarray
import time
from netCDF4 import Dataset
import netCDF4
import numpy
import fsspec
import h5py as h5
import contextlib
import os
import zarr
import s3fs
import aiohttp



class ReadData():
    """
        ReadData class
    """
    def __init__(self, config, result_file) -> None:
        """
            Init method for ReadData class

            :param config: config file structure read by yaml
            :type config: yaml
            :param result_file: result file name
            :type result_file: string
        """

        self.config = config
        # How many times repeat the read
        self.iteration_max = config.get('NUM_ITER')

        # Define the url for dataset
        self.base_url = config.get('LIGHTTPD_CONFIG')
        # Get variable and slice to read
        self.slice_data = eval(config.get('SLICE_DATA'))
        self.variable = config.get('VARIABLE')

        # define result file
        self.fichier_resultat = result_file

        # define S3 configuration
        if config.get('KEY') is not None:
            self.fs_s3 = s3fs.S3FileSystem(anon=False, key=config.get('KEY'), secret=config.get('SECRET'), endpoint_url=config.get('ENDPOINT_URL'))
        else:
            self.fs_s3 = s3fs.S3FileSystem(anon=False, endpoint_url=config.get('ENDPOINT_URL'))







    def multiple_launch(self, fonction):
        """
            multiple_launch method
            Launch x time the function given in parameter

            :param fonction: function to repeat 
            :type fonction: python method
            :return: time statistics
            :rtype: numpy
        """
        # Init time array
        temps = numpy.zeros(self.iteration_max)
        for i in range(0, self.iteration_max):
            redo = True
            compteur = 0
            while(redo):
                try:
                    start = time.time()
                    fonction()
                    end = time.time()
                    temps[i] = (end - start)*1000
                    print(temps[i])
                    redo = False
                except aiohttp.ClientConnectionError:
                    # something went wrong with the exception, decide on what to do next
                    print("Oops, the connection was dropped before we finished")
                    compteur += 1
                    if compteur > 5:
                        raise
                    else:
                        redo = True
                except aiohttp.ClientError:
                    # something went wrong in general. Not a connection error, that was handled above.
                    print("Oops, something else went wrong with the request")
                    compteur += 1
                    if compteur > 5:
                        raise
                    else:
                        redo = True

        # return mean, std, min, quantile 10, quantile 90 and max
        return [numpy.mean(temps), numpy.std(temps), numpy.min(temps), numpy.quantile(temps, 0.1), numpy.quantile(temps, 0.9), numpy.max(temps)]

    def enregistrer_resultat(self, message, tmean, tstd, tmin, tq10, tq90, tmax):
        """
            save timing stat

            :param message: type of test
            :type message: string
            :param tmean: mean
            :type tmean: float
            :param tstd: standard deviation
            :type tstd: float
            :param tmin: minimum
            :type tmin: float
            :param tq10: 10 quantile
            :type tq10: float
            :param tq90: 90 quantile
            :type tq90: float
            :param tmax: max
            :type tmax: float
        """

        message = message + ': %.2fms %.2fms %.2fms %.2fms %.2fms %.2fms %d\n'
        self.fichier_resultat.write(message % (tmean, tstd, tmin, tq10, tq90, tmax, self.iteration_max))
        print(message % (tmean, tstd, tmin, tq10, tq90, tmax, self.iteration_max))

    def set_ref_data(self, ref_data):
        """
            setter for ref_data
            :param ref_data: reference data
            :type ref_data: numpy array
        """
        self.ref_data = ref_data


class ReadDataNetCDF(ReadData):
    """
        ReadDataNetCDF class
        Child class of ReadData
    """
    def __init__(self, config, result_file) -> None:
        """
            Init method for ReadDataNetCDF class

            :param config: config file structure read by yaml
            :type config: yaml
            :param result_file: result file name
            :type result_file: string
        """
        super().__init__(config, result_file)

        # Create the local test dataset
        self.dataset_path = Path(self.config.get('LOCAL_DATA_DIR') + self.config.get('NETCDF_FILE'))
        netcdf_basename = self.dataset_path.stem

        # Define the local index path
        index_file = str(netcdf_basename) + '_index.nc'
        self.index = self.dataset_path.parent.joinpath(index_file)

        # Create s3 url dataset
        self.s3_url_dataset_nc = 's3://' + self.config.get('BUCKET_NAME') + '/' + self.config.get('NETCDF_FILE')
        print('Test performance on file: ' + str(self.s3_url_dataset_nc))

        # Define the index s3 url
        self.s3_url_index = 's3://' + self.config.get('BUCKET_NAME') + '/' + index_file

        # If index file does not exist create it
        # Warning if index file does not exist on S3 bucket there will have an error below and the direct creation on S3 are not implemented yet
        if not os.path.exists(self.index) and (config.get('TEST_chunkindex')):
            self.create_index()

    def create_index(self):
        # Create the index as a netcdf file
        chunkindex.create_index(str(self.index), str(self.dataset_path))

    def read_direct_Dataset(self):
        with Dataset(self.dataset_path, 'r') as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            self.set_ref_data(data)

    def read_s3_Dataset(self):
        with netCDF4.Dataset(self.s3_url_dataset_nc + "#mode=bytes") as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        with self.fs_s3.open(self.s3_url_dataset_nc, mode='rb') as f:
            with xarray.open_dataset(f, engine="h5netcdf", decode_times=False, **args) as dataset:
                data = dataset[variable][self.slice_data]
                print(data.max().values)
                assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def read_s3_h5py(self):
        with self.fs_s3.open(self.s3_url_dataset_nc, mode='rb') as f:
            with h5.File(f) as ds:
                data = ds[self.variable][self.slice_data]
                liste_att = ds[self.variable].attrs.keys()
                if '_FillValue' in liste_att:
                    fillvalue = ds[self.variable].attrs['_FillValue'][0]
                else:
                    fillvalue = False
                if 'scale_factor' in liste_att:
                    scale_factor = ds[self.variable].attrs['scale_factor'][0]
                else:
                    scale_factor = 1
                if 'offset' in liste_att:
                    offset = ds[self.variable].attrs['offset'][0]
                else:
                    offset = 0
                if fillvalue:
                    data = numpy.where(data==fillvalue, numpy.nan, data)*scale_factor + offset
                else:
                    data = data*scale_factor + offset
                print(numpy.nanmax(data))

                #self.ref_data = data
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_chunkindex_h5py_local(self):
        with self.fs_s3.open(self.s3_url_dataset_nc, mode='rb') as dataset:
            with fsspec.open(self.index, mode='rb') as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data)
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_chunkindex_h5py_remote(self):
        with self.fs_s3.open(self.s3_url_dataset_nc, mode='rb') as dataset:
            with self.fs_s3.open(self.s3_url_index, mode='rb') as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data)
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def launch_test(self):
        # Define reference dataset
        # Test below always run to fill the self.ref_data variable
        # Direct read of the file with Dataset NetCDF4
        resultats = super().multiple_launch(self.read_direct_Dataset)
        message = 'NetCDF4_Dataset'
        super().enregistrer_resultat(message, *resultats)

        if (self.config.get('NETCDF_TEST')):
            # Test with NetCDF Dataset
            # Direct S3 acces is not working with netcdf.Dataset library
            resultats = super().multiple_launch(self.read_s3_Dataset)
            message = 'NetCDF4_Dataset'
            super().enregistrer_resultat(message, *resultats)

        if (self.config.get('H5PY_TEST')):
            # Test with h5py 
            resultats = super().multiple_launch(self.read_s3_h5py)
            message = 'NetCDF4_h5py'
            super().enregistrer_resultat(message, *resultats)

        if (self.config.get('XARRAY_TEST')):
            # Test with xarray
            resultats = super().multiple_launch(self.read_s3_xarray)
            message = 'NetCDF4_xarray'
            super().enregistrer_resultat(message, *resultats)
       
        if (self.config.get('CHUNKINDEX_TEST')):
            # Test with chunkindex h5py index in local
            resultats = super().multiple_launch(self.read_s3_chunkindex_h5py_local)
            message = 'chunkindex_local'
            super().enregistrer_resultat(message, *resultats)
            # Test with chunkindex h5py index in remote
            resultats = super().multiple_launch(self.read_s3_chunkindex_h5py_remote)
            message = 'chunkindex_remote'
            super().enregistrer_resultat(message, *resultats)



class ReadDataZarr(ReadData):
    """
        ReadDataNetCDF class
        Child class of ReadData
    """

    def __init__(self, config, result_file) -> None:
        """
            Init method for ReadDataZarr class

            :param config: config file structure read by yaml
            :type config: yaml
            :param result_file: result file name
            :type result_file: string
        """
        super().__init__(config, result_file)

        # Create s3 url dataset
        self.s3_url_dataset_zarr = 's3://' + self.config.get('BUCKET_NAME') + '/' + self.config.get('ZARR_FILE')
        print('Test performance on file: ' + str(self.s3_url_dataset_zarr))

    def read_s3_zarr(self):
        mapper = self.fs_s3.get_mapper(self.s3_url_dataset_zarr)
        with zarr.open(mapper) as zarr_ds:
            data = zarr_ds[self.variable][self.slice_data]
            liste_att = zarr_ds[self.variable].attrs.keys()
            if '_FillValue' in liste_att:
                fillvalue = zarr_ds[self.variable].attrs['_FillValue']
            else:
                fillvalue = False
            if 'missing_value' in liste_att:
                missing_value = zarr_ds[self.variable].attrs['missing_value']
                if not fillvalue:
                    fillvalue = missing_value
            if 'scale_factor' in liste_att:
                scale_factor = zarr_ds[self.variable].attrs['scale_factor']
            else:
                scale_factor = 1
            if 'offset' in liste_att:
                offset = zarr_ds[self.variable].attrs['offset']
            else:
                offset = 0
            # How missing_value and _FillValue are managed in zarr ?
            # Topic : https://github.com/pydata/xarray/issues/5475
            if fillvalue:
                data = numpy.where(data==fillvalue, numpy.nan, data)*scale_factor + offset
            else:
                data = data*scale_factor + offset
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_zarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        mapper = self.fs_s3.get_mapper(self.s3_url_dataset_zarr)
        with xarray.open_zarr(store=mapper, consolidated=True, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def launch_test(self):

        if (self.config.get('ZARR_TEST')):

            # ZARR
            # Test lecture du fichier zarr avec lighttpd et zarr
            resultats = super().multiple_launch(self.read_s3_zarr)
            message = 'zarr'
            super().enregistrer_resultat(message, *resultats)

            # Test lecture du fichier zarr avec lighttpd et xarray
            resultats = super().multiple_launch(self.read_s3_zarr_xarray)
            message = 'zarr_xarray'
            super().enregistrer_resultat(message, *resultats)


class ReadDataNcZarr(ReadData):
    """
        ReadDataNetCDF class
        Child class of ReadData
    """

    def __init__(self, config, result_file) -> None:
        """
            Init method for ReadDataNcZarr class

            :param config: config file structure read by yaml
            :type config: yaml
            :param result_file: result file name
            :type result_file: string
        """
        super().__init__(config, result_file)

        # Create the test dataset
        self.dataset_path = self.config.get('LOCAL_DATA_DIR') + self.config.get('NCZARR_FILE')

        # Create s3 url dataset
        self.s3_url_dataset_nczarr = 's3://' + self.config.get('BUCKET_NAME') + '/' + self.config.get('NCZARR_FILE')
        print('Test performance on file: ' + str(self.s3_url_dataset_nczarr))

    def read_direct_nczarr_Dataset(self):
        with Dataset("file://" + self.dataset_path + "#mode=nczarr") as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_nczarr_Dataset(self):
        with Dataset(self.s3_url_dataset_nczarr + "#mode=nczarr") as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_nczarr_zarr(self):
        mapper = self.fs_s3.get_mapper(self.s3_url_dataset_nczarr)
        with zarr.open(mapper) as zarr_ds:
            data = zarr_ds[self.variable][self.slice_data]
            liste_att = zarr_ds[self.variable].attrs.keys()
            if '_FillValue' in liste_att:
                fillvalue = zarr_ds[self.variable].attrs['_FillValue']
            else:
                fillvalue = False
            if 'scale_factor' in liste_att:
                scale_factor = zarr_ds[self.variable].attrs['scale_factor']
            else:
                scale_factor = 1
            if 'offset' in liste_att:
                offset = zarr_ds[self.variable].attrs['offset']
            else:
                offset = 0
            if fillvalue:
                data = numpy.where(data==fillvalue, numpy.nan, data)*scale_factor + offset
            else:
                data = data*scale_factor + offset
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_s3_nczarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        mapper = self.fs_s3.get_mapper(self.s3_url_dataset_nczarr)
        # consolidated=False have to be set maybe because of metadata pb in nczarr
        with xarray.open_zarr(store=mapper, consolidated=False, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(data.max().values)
            assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))


    def launch_test(self):

        if (self.config.get('NCZARR_TEST')):

            # NCZARR
            # Reading test of nczarr with Dataset 
            # Direct S3 acces is not working with netcdf.Dataset library
            #resultats = super().multiple_launch(self.read_s3_nczarr_Dataset)
            #message = 'nczarr_Dataset'
            #super().enregistrer_resultat(message, *resultats)

            # Reading test of nczarr with lighttpd and zarr
            resultats = super().multiple_launch(self.read_s3_nczarr_zarr)
            message = 'nczarr_zarr'
            super().enregistrer_resultat(message, *resultats)

            # Reading test of nczarr with lighttpd and xarray
            resultats = super().multiple_launch(self.read_s3_nczarr_xarray)
            message = 'nczarr_xarray'
            super().enregistrer_resultat(message, *resultats)



