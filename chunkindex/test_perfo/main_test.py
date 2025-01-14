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
# ======================================================
#
# Project : NetCDF Streaming
# Thales Services Numeriques
#
# ======================================================
# HISTORIQUE
# FIN-HISTORIQUE
# ======================================================


import yaml
import argparse

def main(config_file):

    # init return code
    retour = 0

    with open(f'{config_file}', 'r') as file:
        config = yaml.safe_load(file)
        target = config.get('TECHNO')
        netcdf_test = config.get('NETCDF_TEST')
        zarr_test = config.get('ZARR_TEST')
        nczarr_test = config.get('NCZARR_TEST')
        out_file = config.get('RESULT_FILE')

    if target == 'S3':
        print('Launch test on S3')
        import read_data_s3 as read_data
    elif target == 'lighttpd':
        print('Launch test with lighttpd')
        import read_data_lighttpd as read_data
    else:
        print('No type of technology specified in config file')
        print('Expected:')
        print('TECHNO: S3 or TECHNO: lighttpd')
        exit()

    # define result file
    result_file = open(out_file, 'w')

    try:
        # Instanciate TEST
        test_netcdf = read_data.ReadDataNetCDF(config, result_file)
        test_netcdf.launch_test()

        if (zarr_test):
            test_zarr = read_data.ReadDataZarr(config, result_file)
            test_zarr.set_ref_data(test_netcdf.ref_data)
            test_zarr.launch_test()

        if (nczarr_test):
            test_nczarr = read_data.ReadDataNcZarr(config, result_file)
            test_nczarr.set_ref_data(test_netcdf.ref_data)
            test_nczarr.launch_test()

    except:
        retour = 1
        raise

    finally:
        result_file.close()

    exit(retour)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch chunkindex and NetCDF benchmark.')
    parser.add_argument('filename', metavar='PARAM_filename', nargs=1, help='parameters filename (.yaml)')
    args = parser.parse_args()

    file_path = args.filename[0]

    main(file_path)



