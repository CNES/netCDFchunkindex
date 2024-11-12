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
import plot_results

def main(config_file):

    # init return code
    retour = 0

    with open(f'{config_file}', 'r') as file:
        config = yaml.safe_load(file)
        target = config.get('TECHNO')

    try:
        # Instanciate TEST
        plot_netcdf = plot_results.PlotResults(config)
        plot_netcdf.run_plot()

    except:
        retour = 1
        raise

    exit(retour)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch chunkindex and NetCDF benchmark plot.')
    parser.add_argument('filename', metavar='PARAM_filename', nargs=1, help='parameters filename (.yaml)')
    args = parser.parse_args()

    file_path = args.filename[0]

    main(file_path)



