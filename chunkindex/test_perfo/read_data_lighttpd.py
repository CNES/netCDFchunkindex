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

        # Create the test dataset
        self.dataset_path = Path(self.config.get('LOCAL_DATA_DIR') + self.config.get('NETCDF_FILE'))
        netcdf_basename = self.dataset_path.stem

        # Define the index path
        index_file = str(self.dataset_path.stem) + '_index.nc'
        self.index = self.dataset_path.parent.joinpath(index_file)

        # Define the url for dataset
        self.dataset_url = urllib.parse.urljoin(self.base_url, netcdf_basename + '.nc')
        print('Test performance on file: ' + str(self.dataset_url))

        # Define the url for index
        self.index_url = urllib.parse.urljoin(self.base_url, index_file)

        # If index file does not exist create it
        if not os.path.exists(self.index) and (config.get('TEST_chunkindex')):
            self.create_index()

    def create_index(self):
        # Create the index as a netcdf file
        chunkindex.create_index(str(self.index), str(self.dataset_path))

    def read_direct_chunkindex_h5py(self):
        with open(self.dataset_path, 'rb') as dataset:
            with open(self.index, mode='rb') as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data)
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_direct_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        #with xarray.open_dataset(self.dataset_path, engine="netcdf4", decode_times=False, **args) as dataset:
        with xarray.open_dataset(self.dataset_path, engine="h5netcdf", decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(data.max().values)
            assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def read_direct_Dataset(self):
        with Dataset(self.dataset_path, 'r') as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            self.set_ref_data(data)

    def read_direct_h5py(self):
        with h5.File(self.dataset_path) as ds:
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
            #print(data)
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        with xarray.open_dataset(self.dataset_url, engine="h5netcdf", decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(data.max().values)
            assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def read_lighttpd_Dataset(self):
        with Dataset(self.dataset_url + "#mode=bytes") as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_fsspec_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        with fsspec.open(self.dataset_url, 'rb') as f:
            with xarray.open_dataset(f, engine="h5netcdf", decode_times=False, **args) as dataset:
                data = dataset[variable][self.slice_data]
                print(data.max().values)
                assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def read_lighttpd_fsspec_h5py(self):

        with fsspec.open(self.dataset_url) as f:
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

                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_chunkindex_h5py_local(self):
        with fsspec.open(self.dataset_url) as dataset:
            with open(self.index, mode='rb') as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data)
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_chunkindex_xarray_local(self):
        with fsspec.open(self.dataset_url) as dataset:
            with open(self.index, mode='rb') as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data, method='xarray')
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_chunkindex_h5py_remote(self):
        with fsspec.open(self.dataset_url) as dataset:
            with fsspec.open(self.index_url) as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data)
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_chunkindex_xarray_remote(self):
        with fsspec.open(self.dataset_url) as dataset:
            with fsspec.open(self.index_url) as index:
                data = chunkindex.read_slice(dataset, index, self.variable, self.slice_data, method='xarray')
                print(numpy.nanmax(data))
                assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def launch_test(self):
        # Define reference dataset
        # Test below always run to fill the self.ref_data variable
        # Direct read of the file with Dataset NetCDF4
        resultats = super().multiple_launch(self.read_direct_Dataset)
        message = 'NetCDF4_Dataset'
        super().enregistrer_resultat(message, *resultats)

        if (self.config.get('H5PY_TEST')):
            # Test lecture du fichier de manière directe avec h5py
            resultats = super().multiple_launch(self.read_direct_h5py)
            message = 'NetCDF4_h5py'
            super().enregistrer_resultat(message, *resultats)
            if (self.config.get('TEST_LIGHTTPD')):
                # Test lecture du fichier avec lighttpd fsspec et h5py
                resultats = super().multiple_launch(self.read_lighttpd_fsspec_h5py)
                message = 'NetCDF4_fsspec_h5py'
                super().enregistrer_resultat(message, *resultats)


        if (self.config.get('XARRAY_TEST')):
            # Test lecture du fichier avec xarray en direct
            resultats = super().multiple_launch(self.read_direct_xarray)
            message = 'NetCDF4_xarray'
            super().enregistrer_resultat(message, *resultats)
            if (self.config.get('TEST_LIGHTTPD')):
                # Test lecture du fichier avec lighttpd fsspec et xarray
                resultats = super().multiple_launch(self.read_lighttpd_fsspec_xarray)
                message = 'NetCDF4_fsspec_xarray'
                super().enregistrer_resultat(message, *resultats)
       
        if (self.config.get('CHUNKINDEX_TEST')):
            if (self.config.get('TEST_LIGHTTPD')):
                # Test lecture du fichier avec lighttpd chunkindex h5py local
                resultats = super().multiple_launch(self.read_lighttpd_chunkindex_h5py_local)
                message = 'chunkindex_fsspec_local'
                super().enregistrer_resultat(message, *resultats)
                # Test lecture du fichier avec lighttpd chunkindex h5py remote
                resultats = super().multiple_launch(self.read_lighttpd_chunkindex_h5py_remote)
                message = 'chunkindex_fsspec_remote'
                super().enregistrer_resultat(message, *resultats)
            # Test lecture du fichier en direct avec chunkindex h5py local
            resultats = super().multiple_launch(self.read_direct_chunkindex_h5py)
            message = 'chunkindex'
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

        # Create the test dataset
        self.dataset_path = Path(self.config.get('LOCAL_DATA_DIR') + self.config.get('ZARR_FILE'))

        # Define the url for dataset
        self.url_dataset_zarr = urllib.parse.urljoin(self.base_url, self.config.get('ZARR_FILE'))
        print('Test performance on file: ' + str(self.url_dataset_zarr))

    def read_direct_zarr(self):
        with zarr.open(self.dataset_path) as zarr_ds:
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


    def read_direct_zarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        with xarray.open_zarr(self.dataset_path, consolidated=True, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))


    def read_lighttpd_zarr(self):
        mapper = fsspec.get_mapper(self.url_dataset_zarr)
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


    def read_lighttpd_zarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        mapper = fsspec.get_mapper(self.url_dataset_zarr)
        with xarray.open_zarr(store=mapper, consolidated=True, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))



    def launch_test(self):

        if (self.config.get('ZARR_TEST')):

            # ZARR
            if (self.config.get('TEST_LIGHTTPD')):
                # Test lecture du fichier zarr avec lighttpd et zarr
                resultats = super().multiple_launch(self.read_lighttpd_zarr)
                message = 'zarr_fsspec'
                super().enregistrer_resultat(message, *resultats)

                # Test lecture du fichier zarr avec lighttpd et xarray
                resultats = super().multiple_launch(self.read_lighttpd_zarr_xarray)
                message = 'zarr_fsspec_xarray'
                super().enregistrer_resultat(message, *resultats)

            # Test lecture du fichier zarr en direct
            resultats = super().multiple_launch(self.read_direct_zarr)
            message = 'zarr'
            super().enregistrer_resultat(message, *resultats)

            # Test lecture du fichier zarr en direct avec xarray
            resultats = super().multiple_launch(self.read_direct_zarr_xarray)
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

        # Define the url for dataset
        self.url_dataset_nczarr = urllib.parse.urljoin(self.base_url, self.config.get('NCZARR_FILE'))
        print('Test performance on file: ' + str(self.url_dataset_nczarr))


    def read_direct_nczarr_Dataset(self):
        with Dataset("file://" + self.dataset_path + "#mode=nczarr") as dataset:
            data = dataset[self.variable][self.slice_data]
            print(numpy.nanmax(data))
            assert(numpy.allclose(data, self.ref_data, equal_nan=True))

    def read_lighttpd_nczarr_zarr(self):
        mapper = fsspec.get_mapper(self.url_dataset_nczarr)
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

    def read_lighttpd_nczarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        mapper = fsspec.get_mapper(self.url_dataset_nczarr)
        # consolidated=False have to be set maybe because of metadata pb in nczarr
        with xarray.open_zarr(store=mapper, consolidated=False, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(data.max().values)
            assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def read_direct_nczarr_zarr(self):
        with zarr.open(self.dataset_path) as zarr_ds:
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

    def read_direct_nczarr_xarray(self):
        if (len(self.variable.split('/')) > 1):
            args={'group': self.variable.rsplit('/', 1)[0]}
            variable = self.variable.split('/')[-1]
        else:
            args = dict()
            variable = self.variable
        # consolidated=False have to be set maybe because of metadata pb in nczarr
        with xarray.open_zarr(self.dataset_path, consolidated=False, decode_times=False, **args) as dataset:
            data = dataset[variable][self.slice_data]
            print(data.max().values)
            assert(numpy.allclose(data.values, self.ref_data, equal_nan=True))

    def launch_test(self):

        if (self.config.get('NCZARR_TEST')):

            # NCZARR
            # Reading test of nczarr with Dataset (direct access without lighttpd)
            # Apparently dataset does not support http request
            resultats = super().multiple_launch(self.read_direct_nczarr_Dataset)
            message = 'nczarr_Dataset'
            super().enregistrer_resultat(message, *resultats)

            if (self.config.get('TEST_LIGHTTPD')):
                # Reading test of nczarr with lighttpd and zarr
                resultats = super().multiple_launch(self.read_lighttpd_nczarr_zarr)
                message = 'nczarr_fsspec_zarr'
                super().enregistrer_resultat(message, *resultats)

                # Reading test of nczarr with lighttpd and xarray
                resultats = super().multiple_launch(self.read_lighttpd_nczarr_xarray)
                message = 'nczarr_fsspec_xarray'
                super().enregistrer_resultat(message, *resultats)

            # Reading test of nczarr with zarr
            resultats = super().multiple_launch(self.read_direct_nczarr_zarr)
            message = 'nczarr_zarr'
            super().enregistrer_resultat(message, *resultats)

            # Reading test of nczarr with xarray
            resultats = super().multiple_launch(self.read_direct_nczarr_xarray)
            message = 'nczarr_xarray'
            super().enregistrer_resultat(message, *resultats)


