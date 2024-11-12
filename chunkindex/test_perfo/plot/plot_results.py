# ======================================================
#
# Project : NetCDF Streaming
# Thales Services Numeriques
#
# ======================================================
# HISTORIQUE
# FIN-HISTORIQUE
# ======================================================

# We have a directory where there are some results file
# We want generate figure from this files


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join

class PlotResults():
    """
        PlotResults class
    """
    def __init__(self, config) -> None:
        """
            Init method for PlotResults class

            :param config: config file structure read by yaml
            :type config: yaml
        """

        self.config = config

        # Define directory where figure are going to be write
        self.repertoire_figure = config.get('OUTPUT_DIR')
        # Define result dir where data are store
        self.result_dir = config.get('RESULT_DIR')

    def run_plot(self):
        """
            Main method
            Read data and prepare dict for the plot
        """

        fichier1 = self.result_dir + self.config.get('DATA_SHUFFLE_ON')
        fichier2 = self.result_dir + self.config.get('DATA_SHUFFLE_OFF')
        print(fichier1)

        with open(fichier1, mode = 'r') as f:
            resultat1 = f.readlines()
        with open(fichier2, mode = 'r') as f:
            resultat2 = f.readlines()

        # method to init dict for plot with available and wanted data
        def init_dict(value, moyenne, ecart_type):
            if self.config.get(value.upper()):
                name = self.config.get(value.upper() + '_NAME')
                # No shuffle in zarr and nczarr
                if ('zarr' in value):
                    moyenne[name] = list()
                    ecart_type[name] = list()
                # separate shuffle test
                else:
                    method_1 = name + '_s_on'
                    moyenne[method_1] = list()
                    ecart_type[method_1] = list()
                    method_2 = name + '_s_off'
                    moyenne[method_2] = list()
                    ecart_type[method_2] = list()

        # method to add to dict value for plot
        def add_to_dict(value, moyenne, ecart_type, shuffle, add_zarr):
            if self.config.get(value.upper()):
                name = self.config.get(value.upper() + '_NAME')
                if ('zarr' in value) and add_zarr:
                    moyenne[name].append(float(i.split(' ')[1].replace('ms','')))
                    ecart_type[name].append(float(i.split(' ')[2].replace('ms','')))
                elif not 'zarr' in value:
                    if shuffle:
                        name = name + '_s_on'
                        moyenne[name].append(float(i.split(' ')[1].replace('ms','')))
                        ecart_type[name].append(float(i.split(' ')[2].replace('ms','')))
                    elif not shuffle:
                        name = name + '_s_off'
                        moyenne[name].append(float(i.split(' ')[1].replace('ms','')))
                        ecart_type[name].append(float(i.split(' ')[2].replace('ms','')))

        # Init dict
        moyenne = dict()
        ecart_type = dict()
        # get list of result inside the file
        for i in resultat1:
            method = i.split(':')[0]
            init_dict(method, moyenne, ecart_type)

        # Fill dict for first file (shuffle on)
        for i in resultat1:
            method = i.split(':')[0]
            add_to_dict(method, moyenne, ecart_type, True, True)
        # Fill dict for first file (shuffle off)
        for i in resultat2:
            method = i.split(':')[0]
            add_to_dict(method, moyenne, ecart_type, False, False)

        figure_filename = self.config.get('OUTPUT_DIR') + '/' + self.config.get('FIGURE_NAME')

        # call the create_figure method
        self.create_figure(moyenne, ecart_type, figure_filename)


    def create_figure(self, moyenne, ecart_type, figure_filename):
        """
            create and save figure from dict given in arguments

            :param moyenne: mean of results
            :type moyenne: dict
            :param ecart_type: standart deviation of results
            :type ecart_type: dict
            :param figure_filename: figure filename
            :type figure_filename: string
        """

        fig, ax = plt.subplots()

        labels = []
        measurements = []
        error = []
        for attribute, measurement in moyenne.items():
            labels.append(attribute) 
            measurements.append(measurement[0])
            error.append(ecart_type[attribute][0])

        # Color the best time in green
        rects = ax.barh(labels, measurements, xerr=error)
        indice = np.argsort(measurements)
        minimum = min(measurements)
        maximum = max(measurements)
        delta = (maximum - minimum)/100
        rects[indice[0]].set_color('green')
        if (measurements[indice[1]] - delta) < measurements[indice[0]]:
            rects[indice[1]].set_color('green')
        if (measurements[indice[2]] - delta) < measurements[indice[0]]:
            rects[indice[2]].set_color('green')

        ax.bar_label(rects, padding=3, fontsize=10, rotation=0)
        ax.set_title(self.config.get('FIGURE_TITLE'))
        ax.set_xlabel('Time (ms)')
        axe = ax.get_xlim()
        ax.set_xlim(0, axe[1]*1.1)
        #ax.xaxis.set_tick_params(pad = 5)
        #ax.yaxis.set_tick_params(pad = 10)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none') 
        # Add x, y gridlines
        ax.grid(visible = True, color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.2)
        # Save figure
        plt.savefig(figure_filename, bbox_inches='tight')
        # show figure
        plt.show()






