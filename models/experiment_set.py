# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:11:29 2016

This is a wrapper to load and analyze social behavior data.

Data structure:
Experiment_set
    experiment[n]                       # number of independent recorded avi files
        experimentMetaInformation       # experiment params, video params
        pair[m]                         # number of animals x episodes (aka animal-pair-episode)
            pairStatistics              # shoaling index, interAnimalDistance
            animal[0]
            animal[1]
                animalTimeSeriesCollection
                    # data time series: position, speed, neighbor maps



Data files to analyze are specified in csv file, e.g. generated from a jupyter notebook

Columns parsed from csv file (info):
info['aviPath'] = aviPath    # path to one or more avi files
info['txtPath'] = posPath    # path to trajectory data files corresponding to avi files
info['pairList'] = PLPath[0] # path to pairing matrix

info['epiDur'] = 5      # duration of individual episodes (default: 5 minutes)
info['episodes'] = -1   # number of episodes to process: -1 to load all episodes (default: -1)
info['inDish'] = np.arange(len(posPath))*120     # time in dish before experiments started (default: 10)
info['arenaDiameter_mm'] = 100 # arena diameter (default: 100 mm)
info['minShift'] = 60 # minimum number of seconds to shift for control IAD
info['episodePLcode'] = 1 # flag if first two characters of episode name encode animal pair matrix (default: 0)
info['recomputeAnimalSize'] = 0 # flag to compute animals size from avi file (takes time, default: 1)
info['SaveNeighborhoodMaps'] = 0 # flag to save neighborhood maps for subsequent analysis (takes time, default: 1)
info['computeLeadership'] = 0 # flag to compute leadership index (takes time, default: 1)
info['ComputeBouts'] = 0 # flag to compute swim bout frequency (takes time, default: 1)
info['set'] = np.arange(len(posPath))   # experiment set: can label groups of experiments (default: 0)


OUTPUTS:

[trajectoryFn]siSummary_[episodeLength].csv
    summary table of animal-stimulus statistics for each episode
        -animalID
        -partnerID
        -social index
        -avgSpeed
        -size
        -swim bout rate
        -...

[trajectoryFn]MapData.npy
    Neighborhood Maps (dump of numpy array)

[trajectoryFn]anSize.csv
    Animal size for all animals in 2000 random frames

anSizeAll.csv
    Animal size estimated from [trajectoryFn]anSize.csv


@author: jlarsch
"""

from tkinter import filedialog as tkFileDialog
from models.experiment import experiment
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime


class experiment_set(object):
    def __init__(self, csvFile=[]):
        
        self.csvFile = csvFile
        if not self.csvFile:
            self.csvFile = tkFileDialog.askopenfilename()

        self.df = None
        self.experiments = None
        self.pdf = None

        self.process_csv_experiment_list()

    def process_csv_experiment_list(self):

        self.df = pd.read_csv(self.csvFile, sep=',')
        self.experiments = []

        numExpts = self.df.shape[0]

        for index, row in self.df.iterrows():

            print('processing: ', index + 1, ' out of ', numExpts, 'experiments')
            self.experiments.append(experiment(row))

    def saveExperimentOverviewPDF(self, experiment, label):
        # save pdf summary in same foldera as csv
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            currentTime = datetime.now()
            experiment.PdfFile = experiment.csvFile[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
            try:
                self.pdf = PdfPages(self.PdfFile)
                experiment.plotOverview(label)
                experiment.pdf.savefig()  # saves the current figure as pdf page
                plt.close()
            except:
                print('PDF could not be created')
                pass
