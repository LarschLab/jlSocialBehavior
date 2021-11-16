# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:57:38 2016

@author: jlarsch
"""

import copy
import numpy as np
from models.geometry import Trajectory
from models.animal import Animal
import warnings

class Pair(object):
    # Class defining one pair of animals during one experiment episode

    def __init__(self, shift=[0,0], animalIDs=[0, 1], rng=[], epiNr=None,fullAn=False):

        self.epiNr = epiNr          # Episode number to use in this instance
        self.shift = shift          # tags pair for calculation of control shifted data
        self.animalIDs = animalIDs  # Animal numbers using experiment numbering
        self.animals = []           # Place holder for list of animal instances
        self.experiment = None      # Place holder for parent experiment reference
        self.rng = None             # Place holder for range of frames belonging to this pair-episode
        self.fullAn = fullAn        # tags pair as full experiment data. Append to pair_f in experiment.
        self.rng = rng

    def addAnimal(self, animal):
        self.animals.append(animal)
        return self.animals[animal.ID]

    def joinExperiment(self, experiment):
        self.experiment = experiment
        experiment.addPair(self)

        for i in range(2):
            Animal(ID=i).joinPair(self)

        for i in range(2):
            self.animals[i].wakeUp()

    def medBoutDur(self):
        tmp = [np.nanmedian(np.diff(an.ts.boutStart())) for an in self.animals]
        x = np.array(tmp) / float(self.experiment.expInfo.fps)
        return x

    def LeadershipIndex(self):
        x = np.array([a.ts.FrontnessIndex() for a in self.animals])
        return x

    def thigmoIndex(self):
        x = np.array([np.nanmean(an.ts.positionPol().y()) for an in self.animals])
        return x

    def IAD(self):
        self.shift = [0,0]
        dist = Trajectory()
        dist.xy = self.animals[0].ts.position().xy-self.animals[1].ts.position().xy
        x = np.sqrt(dist.x()**2 + dist.y()**2)
        self.shift = [0,0]
        return x

    def IADs(self):
        x = []
        for i in range(self.experiment.expInfo.nShiftRuns):
            self.shift = self.experiment.shiftList[:, i]
            #print('shift:',self.shift)
            dist = Trajectory()
            dist.xy = self.animals[0].ts.position().xy-self.animals[1].ts.position().xy
            sq = dist.x()**2 + dist.y()**2
            x.append(np.sqrt(sq))
            #print(np.where(~np.isfinite(sq))[0])
            self.shift = [0,0]
        return np.array(x).astype('float')

    def IADhist(self):
        histBins = np.arange(100)
        x = np.histogram(self.IAD()[np.isfinite(self.IAD())], bins=histBins, normed=1)[0]
        return x

    def spIAD_meanTrace(self):
        a= self.IADs()
        #print(a.shape)
        if a.dtype == np.object_:
            print('this is not a float!!!!!')
            print(a)
        x = np.nanmean(a, axis=0)
        return x

    def spIAD_m(self):
        a=self.spIAD_meanTrace()

        x = np.nanmean(a)
        return x

    def IAD_m(self):
        x = np.nanmean(self.IAD())
        return x

    def spIAD_std(self):
        x = np.nanstd(self.IADs())
        return x

    def ShoalIndex(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                spIAD = self.spIAD_m()
                IAD = self.IAD_m()
                x = (spIAD - IAD) / spIAD
            except Warning:
                print('shoalIndex warning: NaN found. Will not compute Attraction in this episode.')
                #raise
                x = np.nan
        return x

    def avgSpeed(self):
        a1 = np.nanmean(self.animals[0].ts.speed())
        a2 = np.nanmean(self.animals[1].ts.speed())
        return np.array([a1, a2])

    def a1SpeedTrace_smooth(self):
        return np.array(self.animals[0].ts.speed_smooth())

    def a1SpeedTrace(self):
        return np.array(self.animals[0].ts.speed())

    def avgSpeed_smooth(self):
        a1 = np.nanmean(self.animals[0].ts.speed_smooth())
        a2 = np.nanmean(self.animals[1].ts.speed_smooth())
        return np.array([a1, a2])
        
    def max_out_venture(self):
        mov = []
        mov.append([x.ts.positionPol().y() for x in self.animals])
        return np.nanmax(mov)

    def crossCorrStimAn(self):

        an = self.animals[0].ts.speed_smooth()
        an[np.isnan(an)]=0
        an = an-np.nanmean(an)
        st = self.animals[1].ts.speed_smooth()
        st[np.isnan(st)] = 0
        st = st-np.nanmean(st)
        cc = np.correlate(st, an[:-20], mode='valid')
        return np.nanmin(cc), np.nanargmin(cc), np.nanmax(cc), np.nanargmax(cc)

    def get_var_from_all_animals(self, var):
        #this function returns a specified variable from all animals as a matrix.
        #animal number will be second dimension
        
        tmp = self.animals
        for i in range(len(var)):
            tmp = [getattr(x, var[i]) for x in tmp]
        return np.stack(tmp, axis=1)
