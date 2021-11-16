import numpy as np
import os
import datetime
import glob

import functions.getVideoProperties
import functions.plotFunctions_joh as johPlt
import functions.randomDotsOnCircle as randSpacing
import functions.video_functions as vf
from models.pair import Pair
from models.animal import Animal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from functions.getAnimalSizeFromVideo import getAnimalSizeFromVideo
import pandas as pd
from datetime import timedelta
from datetime import datetime
import random

#
# experiment class stores raw animal trajectory data and meta information for one video recording
# experiments contain 1 or more animal-pair-episodes (ape)
# 1 ape represents a period of animal-stimulus pairing used to compute aggregate stats such  as inter animal distance

class ExperimentMeta(object):
    # ExperimentMeta class collects file paths, arena and video parameters for one experiment
    def __init__(self, expinfo):
        self.readLim = None

        if isinstance(expinfo, str):
            print('Experiment called with string input. Using all default parameters.')
            self.trajectoryPath = expinfo
        else:

            trajectoryPath = expinfo['txtPath']
            self.trajectoryPathAll = np.array(trajectoryPath.split())
            self.trajectoryFileNum = self.trajectoryPathAll.shape[0]
            if self.trajectoryFileNum > 1:
                self.trajectoryPath = self.trajectoryPathAll[0]
            else:
                self.trajectoryPath = trajectoryPath

            try:
                num_lines = sum(1 for line in open(self.trajectoryPath))
                self.readLim = np.min([expinfo['readLim'], num_lines])
                print('readLim: ' + str(self.readLim))
            except KeyError:
                print('readLim not specified. Using all data (default).')
                self.readLim = None
            except:
                print('Could not determine readLim at this point. Using all data (default).')
                self.readLim = None


        self.aviPath = None
        self.aspPath = None
        self.anSizeFile = None  #   path to animal size file
        self.roiPath = None     #   path to Bonsai_getCircles ROI definition file
        self.pairListFn = None  #   path to pair list file
        self.processingDir = None
        self.outputDir = None

        self.animalSet = None
        self.inDishTime = None
        self.minShift = None
        self.episodeDur = None
        self.episodes = None
        self.episodePLcode = None
        self.pairListAll = None
        self.nShiftRuns = None
        self.anIDAll = None
        self.birthDayAll = None # used to specify on loading until 2018, now adding age info from notebook
        self.expTime = None # used to specify on loading until 2018, now adding expTime info from notebook
        self.SaveNeighborhoodMaps = None
        self.recomputeAnimalSize = None
        self.ComputeBouts = None
        self.computeLeadership = None
        self.ComputeSync = None
        self.stimulusProtocol = None
        self.allowEpisodeSwitch = None

        self.numPairs = None
        self.txtTrajectories = None
        self.rois = None
        self.videoDims = None
        self.numFrames = None
        self.fps = None
        self.pxPmm = None

        self.filteredMaps = None



        #Parse arguments from expinfo or use defaults with a notification.

    def fillCompleteMeta(self, expinfo, rawTra):

        #   some basic checks on validity of inputs, defaults for missing inputs.
        try:
            self.episodeDur = float(expinfo['epiDur'])
        except KeyError:
            print('episode duration required')
            raise

        try:
            if expinfo['aviPath']=='default':
                print('searching for default avi...')
                head, tail = os.path.split(self.trajectoryPath)
                self.aviPath = glob.glob(head+'\\*.avi')[0]
                print('found: ',self.aviPath)
            else:
                self.aviPath = expinfo['aviPath']
            tmp = open(self.aviPath, 'r')
            tmp.close()
        except KeyError as error:
            print('No video file specified. Will resort to defaults.', error)
            self.aviPath = None

        except TypeError as error:
            print('No video file specified. Will resort to defaults.', error)
            self.aviPath = None

        except IndexError as error:
            print('No video file specified. Will resort to defaults.', error)
            self.aviPath = None

        except FileNotFoundError as error:
            print('Specified video file not found.', error)
            self.aviPath = None
            raise

        try:
            self.pairListFn = expinfo['pairList']
            self.pairListAll = np.loadtxt(self.pairListFn, dtype='int')
            print('Pairlist found. Size: ', self.pairListAll.shape)
        except KeyError:
            print('Pair list not specified. Using default...')
            self.pairListAll = np.diag((-1, -1)) + 1
        except ValueError:
            print('Pair list not found. Using default...')
            self.pairListAll = np.diag((-1, -1)) + 1
        except IOError:
            print('Pair list file not found. Using default...')
            self.pairListAll = np.diag((-1, -1)) + 1

        try:
            self.episodePLcode = expinfo['episodePLcode']
        except KeyError:
            print('Use of PLcode not specified. Not using (default).')
            self.episodePLcode = 0

        if self.episodePLcode:
            self.numPairs = self.pairListAll.shape[1]-1
        else:
            self.numPairs = self.pairListAll.sum()

        try:
            self.arenaDiameter_mm = expinfo['arenaDiameter_mm']
        except KeyError:
            print('arenaDiameter_mm not specified. Using 100 mm (default).')
            self.arenaDiameter_mm = 100


        try:
            self.pxPmm, self.roiPath, self.rois = self.PxPmmFromRois()
        except NameError:

            try:

                self.pxPmm = float(expinfo['pxPmm'])
                print('using predefined pxPmm scaling.', self.pxPmm)
            except KeyError:

                # try reading from video
                print('pxPmm not yet computed. Trying to re compute.')
                self.pxPmm = vf.get_pixel_scaling(path, forceCorrectPixelScaling=1, forceInput=0)


        # If a video file name is passed, collect video parameters
        if self.aviPath:
            # get video meta data
            vp = functions.getVideoProperties.getVideoProperties(self.aviPath)  # video properties
            # self.ffmpeginfo = vp
            self.videoDims = [int(vp['width']), int(vp['height'])]
            self.numFrames = int(vp['nb_frames'])
            self.fps = int(vp['fps'])
            # self.videoDims = np.array([2048, 1280])
            # self.numFrames = 558135
            # self.fps = 30
        else:  # otherwise, use defaults
            self.videoDims = np.array([2048, 1280])
            self.numFrames = rawTra.shape[0]
            self.fps = 30
            print('Attention! Using default video parameters: ', self.videoDims, self.numFrames, self.fps)

        try:
            self.camHeight = expinfo['camHeight']

        except KeyError:
            print('camHeight not specified. Using 79 cm (default).')
            self.camHeight = 79

        try:
            self.minShift = expinfo['minShift'] * self.fps

        except KeyError:
            print('minShift not specified. Using 60 seconds (default).')
            self.minShift = 60 * self.fps

        try:
            self.nShiftRuns = expinfo['nShiftRuns']
        except KeyError:
            print('nShiftRuns not specified. Using default: 10')
            self.nShiftRuns = 10


        try:
            self.anSizeFile = expinfo['AnSizeFile']
            tmp = open(self.anSizeFile, 'r')
            tmp.close()
        except KeyError:
            search = os.path.split(self.trajectoryPath)[0]
            tmp = glob.glob(search + '\\*_anSIze*')
            if tmp:
                self.anSizeFile = tmp[0]
                print('AnSizeFile not specified. Found default.',self.anSizeFile)
            else:

                self.anSizeFile = -1
                print('No AnSizeFile specified and no default found. recompute on demand')

        except FileNotFoundError:
            print('AnSizeFile specified but not found.')
            self.anSizeFile = -1

        try:
            self.episodes = expinfo['episodes']
            maxEpisodes= int(np.floor(((self.numFrames / self.fps) / 60) / self.episodeDur))
            if maxEpisodes<self.episodes:
                print('Warning: more episodes specified than what is available. setting to ', maxEpisodes)
                self.episodes=maxEpisodes
            if self.episodes == -1:
                self.episodes = maxEpisodes
        except KeyError:
            print('episode number not specified. Using default: all')
            self.episodes = int(np.floor(((self.numFrames / self.fps) / 60) / self.episodeDur))

        print('Using: ', self.episodes, ' episodes. Episode duration (minutes): ', self.episodeDur)


        try:
            self.animalSet = expinfo['set']
        except KeyError:
            print('animalSet not specified. Using default: 0')
            self.animalSet = 0

        try:
            self.inDishTime = float(expinfo['inDish'])
        except KeyError:
            print('inDishTime not specified. Using default: 0')
            self.inDishTime = 0

        try:
            self.SaveNeighborhoodMaps = expinfo['SaveNeighborhoodMaps']
        except KeyError:
            print('SaveNeighborhoodMaps not specified. Using default: True')
            self.SaveNeighborhoodMaps = True

        try:
            self.filteredMaps = expinfo['filteredMaps']
        except KeyError:
            print('filteredMaps not specified. Using default: False')
            self.filteredMaps = False

        try:
            self.recomputeAnimalSize = expinfo['recomputeAnimalSize']
        except KeyError:
            print('recomputeAnimalSize not specified. Using default: True')
            self.recomputeAnimalSize = True

        try:
            self.ComputeBouts = expinfo['ComputeBouts']
        except KeyError:
            print('ComputeBouts not specified. Using default: True')
            self.ComputeBouts = True

        try:
            self.computeLeadership = expinfo['computeLeadership']
        except KeyError:
            print('computeLeadership not specified. Using default: True')
            self.ComputeBouts = True

        try:
            self.ComputeSync = expinfo['ComputeSync']
        except KeyError:
            print('ComputeSync not specified. Using default: False')
            self.ComputeSync = False



        try:
            self.allowEpisodeSwitch = expinfo['allowEpisodeSwitch']
        except KeyError:
            print('allowEpisodeSwitch not specified. Using default: False')
            self.allowEpisodeSwitch = 0

        try:
            self.stimulusProtocol = expinfo['stimulusProtocol']
        except KeyError:
            print('stimulusProtocol not specified. Using default: 0')
            self.stimulusProtocol = 0

        try:
            tmp = np.array(expinfo['anIDAll'].split())
            self.anIDAll = np.array(tmp)
        except KeyError:
            print('Animal IDs not specified. Using running numbers during data loading.')
            self.anIDAll = np.arange(self.numPairs)

        try:
            tmp = np.array(expinfo['birthDayAll'].split())
            tmp = [datetime.strptime(x, '%Y-%m-%d-%H-%M') for x in tmp]
            self.birthDayAll = np.array(tmp)
        except KeyError:

            try:
                bdg = np.array(expinfo['bdGroup'].split()).astype('int8')
                bdl = np.array(expinfo['bd'].split())
                bdl = [datetime.strptime(x + '-09-00', '%Y-%m-%d-%H-%M') for x in bdl]

                self.birthDayAll = np.array([bdl[x] for x in bdg])

            except KeyError:
                print('BirthDays not specified. Cannot determine animal age during data loading.')
                self.birthDayAll = np.repeat(np.nan, self.numPairs)

            except:
                print('sth went wrong with birthdays.')
                raise

        try:
            self.expTime = expinfo['expTime']
        except KeyError:
            print('expTime not specified. Cannot determine animal age during loading.')
            self.expTime = np.repeat(np.nan, self.numPairs)

        try:
            self.processingDir = expinfo['ProcessingDir']
        except KeyError:
            print('ProcessingDir not specified. Saving with raw data.')
            self.processingDir = os.path.split(self.trajectoryPath)[0]

        try:
            self.outputDir = expinfo['outputDir']
        except KeyError:
            print('outputDir not specified. Saving with raw data.')
            self.outputDir = os.path.split(self.trajectoryPath)[0]

    def PxPmmFromRois(self):
        #
        # calculate pixel scaling based on dish ROI file
        # bonsai VR experiments produce a ROIdef* file
        # idTracker experiment pipeline produces bgMed_scale* file with ROI information

        startDir, tail = os.path.split(self.trajectoryPath)
        ROIpath = glob.glob(startDir + '\\ROIdef*')  # skype experiments
        if len(ROIpath) < 1: # no skype rois, try idTracker pipeline output
            print('No skype roi found.')
            ROIpath = glob.glob(startDir + '\\bgMed_scale*')  # idTracker pipeline output
            if len(ROIpath) < 1:
                ROIpath = glob.glob(startDir + '\\*bgMed.csv')  # medaka pipeline output
                if len(ROIpath) < 1:

                    print('No idTracker pipeline ROI found.')
                    ROIpath = None
                    raise NameError('no ROIs found')
                else:
                    print('medaka pipeline ROI found.')
                    rois = np.loadtxt(ROIpath[0], skiprows=1, delimiter=',')
                    r_px = rois.mean(axis=0)[3]
                    print(r_px)
            else:
                print('idTracker pipeline ROI found.')
                rois = np.loadtxt(ROIpath[0], skiprows=1, delimiter=',')
                r_px = rois.mean(axis=0)[3]

        else:
            print('skype roi found')
            rois = np.loadtxt(ROIpath[0])
            r_px = rois.mean(axis=0)[-1]
        roiPath = ROIpath[0]
        pxPmm = 2 * r_px / self.arenaDiameter_mm
        return pxPmm, roiPath, rois

class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self, expDef):

        self.expInfo = None
        self.rawTra = None
        self.episodeAll = None  # time series of episode name
        self.anSize = None      # np array of animal size
        self.pair = []          # place holder for animal-pair-episodes
        self.pair_f = []
        self.animals = []
        self.shiftList = None   # fix list for 'random' shifts for time shift control data.
        self.pair2animal = []
        self.mapBins = np.linspace(-20, 20, 31)

        # imitate overload behavior
        if type(expDef) == pd.Series:       # typically, run with a pandas df row of arguments

            self.expInfo = ExperimentMeta(expDef)    # initialize meta information
            print('loading data', end="")
            self.rawTra, self.episodeAll = self.loadData()      # load raw data
            print(' ...done. Shape: ', self.rawTra.shape)
            #print(self.rawTra[1000:1004])
            self.expInfo.fillCompleteMeta(expDef,self.rawTra)               # correct and complete meta information
            if self.expInfo.recomputeAnimalSize:
                self.anSize = self.getAnimalSize()              # compute animal size if requested

            # self.load_animalShapeParameters()
            # generate one random shift list, specifying shift for each of the control runs for shifted IAD
            #shiftA=np.array([int(random.uniform(self.expInfo.minShift,
            #                           self.expInfo.episodeDur*self.expInfo.fps*60 - self.expInfo.minShift))
            #        for x in range(self.expInfo.nShiftRuns)])
            #shiftB=shiftA+self.expInfo.minShift

            ## Shifting strategy as of Dec 2019:

            # generate list with pre-defined regularly spaced shift intervals for reproducible results
            # (as opposed to randomized shift on each analysis run)
            #
            # shiftList is (nShifts,2) matrix
            # nShift tuples to shift animals 1(shiftA) and 2(shiftB), respectively
            # shift amounts are determined to cover the following range:
            # for animal 1: from minShift to shiftMid (middle of episode duration)
            # for animal 2: from shiftMid + minshift to episode duration - minShift (reversed order)
            # this should ensure that all shift combinations differ > minShift but with repeating stimuli, there may be rare 'unfortunate' combinations.
            # shiftMid is calculated based on episodeDur(minutes), --> resulting unit = frames
            # expInfo.minShift is specified in seconds as input argument via notebook and transformed to frames in the expInfo class.

            # During calculation of Shoaling Index, all shifts are used to create a mean null expectation.
            # For saved neighborhood maps, only one shift is used by default to save time.


            shiftMid=int(0.5*self.expInfo.episodeDur*self.expInfo.fps*60)
            shiftA= np.linspace(self.expInfo.minShift,
                                shiftMid,
                                self.expInfo.nShiftRuns).astype('int')
            shiftB = np.linspace(shiftMid+self.expInfo.minShift,
                                 2*shiftMid-self.expInfo.minShift,
                                 self.expInfo.nShiftRuns).astype('int')[::-1]


            self.shiftList = np.array([shiftA, shiftB])
            self.linkFullAnimals()
            self.linkFullPairs()
            self.splitToPairs()                                 # Split raw data table into animal-pair-episodes
            self.saveExpData()                                  # compute and collect pair statistics

        elif type(expDef) == str:           # can run with just an avi file as minimal input.

            self.expInfo = ExperimentMeta(expDef)
            print('loading data', end="")
            self.rawTra, self.episodeAll = self.loadData()
            print(' ...done. Shape: ', self.rawTra.shape)


        else:
            print('Wrong experiment definition argument. Provide TxtPath or pd.Series')
            raise FileNotFoundError

    def linkFullAnimals(self):
        for i in range(self.expInfo.numPairs+1):# adding extra 'animal' which is stimulus!
            Animal(ID=i).joinExperiment(self)

    def linkFullPairs(self):#  add full traces as one long episode for easy access

        if self.expInfo.episodePLcode:
            print('PL code used, omitting FullPairs.')
        else:
            pairList = self.expInfo.pairListAll

            for p in range(self.expInfo.numPairs):

                currPartnerAll = np.where(pairList[:, p])[0]

                for mp in range(currPartnerAll.shape[0]):
                    currPartner = currPartnerAll[mp]
                    Pair(shift=[0,0],
                         animalIDs=[p, currPartner],
                         epiNr=0,
                         rng=[0, self.expInfo.numFrames],
                         fullAn=True).joinExperiment(self)

    def splitToPairs(self):
        # Split raw data table into animal-pair-episodes
        # generating one 'pair' instance for each episode and each animal
        # look up pair members from PairList matrix

        numPairs = self.expInfo.numPairs
        episodes = self.expInfo.episodes
        print('Splitting data into ', str(numPairs*episodes), ' animal-pair-episodes:')

        offset = 0
        skip = 0

        fps = self.expInfo.fps
        episodeDur = self.expInfo.episodeDur  # expected in minutes
        episodeFrames = fps * episodeDur * 60
        pair2animal = []

        for i in range(episodes):

            if (i + skip) > episodes:
                break

            rng = np.array([offset + (i * episodeFrames), offset + ((i + 1) * episodeFrames)]).astype('int')

            if self.expInfo.allowEpisodeSwitch == 0:
                blockEpisodes = self.episodeAll[rng[0]:rng[1]]

                if np.unique(blockEpisodes).shape[0] > 1:
                    print(np.unique(self.episodeAll))
                    print(offset,rng)
                    offset = offset + np.where(np.abs(np.diff(blockEpisodes)) > 0)[0][0]+1
                    print('episode transition detected at episode: ', i, '. applying offset: ', offset)
                    rng = np.array([offset + (i * episodeFrames), offset + ((i + 1) * episodeFrames)]).astype('int')
                    skip += 1

                blockEpisodes = self.episodeAll[rng[0]:rng[1]]
                if np.unique(blockEpisodes).shape[0] > 1:
                    print('still wrong offset')
                    raise ValueError
            #else:
            #    print('allowing sub episodes.')

            epCurr = self.episodeAll[rng[0]+10]

            if self.expInfo.episodePLcode:
                pairListNr = int(epCurr[:2])
                pairList = self.expInfo.pairListAll[pairListNr * 16:(pairListNr + 1) * 16, :]
            else:
                pairList = self.expInfo.pairListAll

            for p in range(numPairs):

                currPartnerAll = np.where(pairList[:, p])[0]

                for mp in range(currPartnerAll.shape[0]):
                    currPartner = currPartnerAll[mp]
                    Pair(shift=[0,0], animalIDs=[p, currPartner], epiNr=i, rng=rng).joinExperiment(self)
                    pair2animal.append(p)
        self.pair2animal = np.array(pair2animal)

    def saveExpData(self):

        # collect summary statistics for each animal-pair-episode
        # produce pandas df where each row is one episode
        # ordering is same as self.pair list

        episodeStartFrame = np.array([x.rng[0] for x in self.pair])
        animalSet = np.repeat(self.expInfo.animalSet,episodeStartFrame.shape[0])
        inDishTime = (episodeStartFrame / (30 * 60)) + self.expInfo.inDishTime
        AnimalIndex = np.array([x.animalIDs[0] for x in self.pair])
        cp = np.array([x.animalIDs[1] for x in self.pair])  # current partner
        episodeName = self.episodeAll[episodeStartFrame]
        epiNr = np.array([x.epiNr for x in self.pair])
        bd = self.expInfo.birthDayAll[AnimalIndex]
        anID = self.expInfo.anIDAll[AnimalIndex]
        if np.shape(self.anSize):
            anSize = self.anSize[AnimalIndex]
        else:
            anSize = 0

        print('Computing pair statistics:')
        si = self.computeSocialIndex()

        print('Average speed... ', end='')
        avgSpeed = np.array([x.avgSpeed()[0] for x in self.pair])
        avgSpeed_smooth = np.array([x.avgSpeed_smooth()[0] for x in self.pair])
        print('Thigmotaxis index... ', end='')
        thigmoIndex = np.array([x.thigmoIndex()[0] for x in self.pair])

        if self.expInfo.computeLeadership:
            print('Leadership index... ', end='')
            leadershipIndex = np.array([x.LeadershipIndex()[0] for x in self.pair])
        else:
            leadershipIndex = 0


        if self.expInfo.ComputeBouts == 1:
            print('Bout duration... ', end='')
            boutDur = np.array([x.medBoutDur()[0] for x in self.pair])
        else:
            boutDur = 0

        if self.expInfo.ComputeSync == 1:
            print('Synchronization ... ', end='')
            sync = np.array([x.crossCorrStimAn() for x in self.pair])
            sync_amp=[x[0] for x in sync]
            sync_t=[x[1] for x in sync]
        else:
            sync_amp=0
            sync_t=0

        print(' done.')


        expStart = datetime.strptime(self.expInfo.expTime, '%Y-%m-%d %H:%M:%S')
        tRun = np.array([expStart + timedelta(minutes=x) for x in inDishTime])

        try:
            ageAll = np.array([(expStart - x).days for x in bd])
        except:
            print('Could not compute animal age at experiment date. Using default: 0')
            ageAll = np.zeros(avgSpeed.shape)



        df = pd.DataFrame(
            {'animalSet': animalSet,
             'animalIndex': AnimalIndex,
             'animalID': anID,
             'CurrentPartner': cp,
             'si': si,
             'episode': episodeName.squeeze(),
             'epStart': episodeStartFrame,
             'inDishTime': inDishTime,
             'epiNr': epiNr,
             'time': tRun,
             'birthDay': bd,
             'stimulusProtocol': self.expInfo.stimulusProtocol,
             'age': ageAll})

        df['avgSpeed'] = avgSpeed
        df['avgSpeed_smooth'] = avgSpeed_smooth
        df['anSize'] = anSize
        df['thigmoIndex'] = thigmoIndex
        df['boutDur'] = boutDur
        df['leadershipIndex'] = leadershipIndex
        df['sync_amp'] = sync_amp
        df['sync_t'] = sync_t

        txtFn = os.path.split(self.expInfo.trajectoryPath)[1]
        csvFileOut = os.path.join(self.expInfo.processingDir, txtFn[:-4] + '_siSummary_epi' + str(self.expInfo.episodeDur) + '.csv')
        df.to_csv(csvFileOut, encoding='utf-8')

        if self.expInfo.SaveNeighborhoodMaps:
            numepiAll = self.expInfo.episodes * self.expInfo.numPairs
            nmAll = np.zeros((numepiAll, 3, 2, self.mapBins.shape[0]-1, self.mapBins.shape[0]-1))  # animal,[neighbor,speed,turn, neighbor filt,speed filt,turn filt],[data,shuffle0],[mapDims]
            #print('Computing neighborhood maps... ', end="\r", flush=True)
            print('Computing neighborhood maps... ', end="")
            for i in range(numepiAll):

                if self.expInfo.filteredMaps:

                    # using smoothed post-hoc heading
                    nmAll[i, 0, 0, :, :] = self.pair[i].animals[0].ts.neighborMat_filt(window_len=31)
                    nmAll[i, 1, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_speed_filt(window_len=31)
                    nmAll[i, 2, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_turn_filt(window_len=31)

                    self.pair[i].shift = [self.shiftList[0], 0]
                    nmAll[i, 0, 1, :, :] = self.pair[i].animals[0].ts.neighborMat_filt(window_len=31)
                    nmAll[i, 1, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_speed_filt(window_len=31)
                    nmAll[i, 2, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_turn_filt(window_len=31)

                else:

                    # using tracked heading
                    nmAll[i, 0, 0, :, :] = self.pair[i].animals[0].ts.neighborMat()
                    nmAll[i, 1, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_speed()
                    nmAll[i, 2, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_turn()

                    self.pair[i].shift = [self.shiftList[0], 0]
                    nmAll[i, 0, 1, :, :] = self.pair[i].animals[0].ts.neighborMat()
                    nmAll[i, 1, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_speed()
                    nmAll[i, 2, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_turn()

                self.pair[i].shift = [0, 0]

            #print(' done. Saving maps...', end="\r", flush=True)
            print(' done. Saving maps...', end="")
            npyFileOut = os.path.join(self.expInfo.processingDir, txtFn[:-4] + 'MapData.npy')
            np.save(npyFileOut, nmAll)

    def computeSocialIndex(self):

        si = []
        i = 0
        numPairs = self.expInfo.numPairs
        episodes = self.expInfo.episodes
        print('Computing Shoaling index: ', end='\r')
        for x in self.pair:
            if np.mod(i, 100) == 0:
                print('Computing Shoaling index: ',
                      str(int(100 * (i / (float(numPairs * episodes))))),
                      ' %', end='\r', flush=True)
            si.append(x.ShoalIndex())
            i += 1
        print('Computing Shoaling index: ... done.', end='')
        return np.array(si)

    def loadData(self):
        # read data for current experiment or many-dish-set
        # begin by reading first line to determine format
        #print(' ', self.expInfo.trajectoryPath, end="\r", flush=True)
        print(' ','file: ', self.expInfo.trajectoryPath, end="")
        VRformat = False
        if self.expInfo.trajectoryPath[-3:]=='npy':
            #idtrackerai output
            raw=np.load(self.expInfo.trajectoryPath)
            tmp = np.dstack([raw, np.zeros((raw.shape[:2]))])
            tmp = tmp.reshape((tmp.shape[0], 3 * raw.shape[1]))
            rawData=pd.DataFrame(tmp)
            episodeAll = np.zeros(rawData.shape[0])
        else:
            firstLine = pd.read_csv(self.expInfo.trajectoryPath,
                                    header=None,
                                    nrows=1,
                                    sep=':')

            if firstLine.values[0][0][0] == '(':
                rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                      sep=',|\)|\(',
                                      engine='python',
                                      index_col=None,
                                      header=None,
                                      skipfooter=1,
                                      usecols=[2, 3, 6, 7, 10],
                                      names=np.arange(5))
                episodeAll = rawData[4]
                rawData=rawData.drop(rawData.columns[[4]], axis=1).astype(float)
                rawData.insert(loc=2,column='o1',value=0)
                rawData.insert(loc=5, column='o2', value=0)
                print('old bonsai format detected')
                # rawData= mat.reshape((mat.shape[0],2,2))

            elif firstLine.values[0][0][0] == 'X':
                rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                      header=None,
                                      delim_whitespace=True,
                                      skiprows=1)
                print('found idTracker csv')
                episodeAll = np.zeros(rawData.shape[0])
                #episodeAll = pd.DataFrame(np.zeros(rawData.shape[0]))

            else:  # default for all bonsai VR experiments since mid 2017
                VRformat = True
                rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                      header=None,
                                      delim_whitespace=True,
                                      nrows=self.expInfo.readLim)
                print('VRtrack: '+self.expInfo.trajectoryPath + str(self.expInfo.readLim))
                episodeAll = np.array(rawData.loc[:, rawData.columns[-1]])

            if (str(self.expInfo) is False) and (self.expInfo.trajectoryFileNum > 1) and VRformat:
                print('More than one trajectory file, loading all:', end='\r')
                fnCombinedData = self.expInfo.trajectoryPathAll[-1][:-7]+'all'+'.h5'
                if np.equal(~os.path.isfile(fnCombinedData), -1):
                    self.batchToHdf(fnCombinedData)

                print(' Reading combined data.')
                hdf = pd.HDFStore(fnCombinedData)
                rawData = hdf['rawData']
                rawData = rawData.reset_index().drop('index', axis=1)
                # pd.read_csv(fnCombinedData, header=None, delim_whitespace=True)
                hdf.close()
                episodeAll = np.array(rawData.loc[:, rawData.columns[-1]])

        return rawData, episodeAll

    def batchToHdf(self, fnCombinedData):
        # create (or open) an hdf5 file and opens in append mode
        hdf = pd.HDFStore(fnCombinedData)
        gl = pd.read_csv(self.expInfo.trajectoryPathAll[0],
                         header=None,
                         delim_whitespace=True,
                         skiprows=1000,
                         nrows=1)
        gl_int = gl.select_dtypes(include=['int64'])
        converted_int = gl_int.apply(pd.to_numeric, downcast='signed')
        gl[converted_int.columns] = converted_int

        gl_float = gl.select_dtypes(include=['float64'])
        converted_float = gl_float.apply(pd.to_numeric, downcast='float')
        gl[converted_float.columns] = converted_float

        try:
            i = 0
            for f in self.expInfo.trajectoryPathAll:
                #print(i, ' out of ', self.expInfo.trajectoryFileNum, end='\r', flush=True)
                print(f)
                tmp = pd.read_csv(f, header=None, delim_whitespace=True)
                tmp.fillna(-1, inplace=True)
                tmp = tmp.astype(gl.dtypes)
                # simplify episode to numeric
                tmp.iloc[:, -1] = np.array([x[:2] for x in tmp.iloc[:, -1]]).astype('int8')
                hdf.append('rawData', tmp)
                i += 1
            print(' done. Saving combined Data')
            hdf.close()  # closes the file
            # (csvFileOut, encoding='utf-8')
            # rawData.to_csv(fnCombinedData,index=False,header=False,float_format='%.3f')
        except:
            hdf.close()
            print('problems')
            raise
        return 1

    def getAnimalSize(self):

        if self.expInfo.anSizeFile == -1:

            anSize = getAnimalSizeFromVideo(currAvi=self.expInfo.aviPath,
                                            rawData=self.rawTra,
                                            numPairs=self.expInfo.numPairs,
                                            roiPath=self.expInfo.roiPath,
                                            camHeight=self.expInfo.camHeight)/self.expInfo.pxPmm

            self.expInfo.anSizeFile = self.expInfo.trajectoryPath[:-4]+'_anSize.csv'
            print('saving anSize to', self.expInfo.anSizeFile)

            np.savetxt(self.expInfo.anSizeFile, anSize)
            print('Animal Size saved.')
        else:
            if np.equal(~os.path.isfile(self.expInfo.anSizeFile), -1):
                anSize = np.zeros(15)
            else:
                anSize = np.loadtxt(self.expInfo.anSizeFile)
        return anSize

    def addPair(self, pair):
        if pair.fullAn:
            self.pair_f.append(pair)
            return self.pair_f[-1]
        else:
            self.pair.append(pair)
            return self.pair[-1]

    def addAnimal(self, animal):
        self.animals.append(animal)
        return self.animals[-1]
      
    def load_animalShapeParameters(self):
        
        aspPath=self.expInfo.aspPath
        self.haveASP=np.equal(~os.path.isfile(aspPath),-2)
        
        if self.haveASP:
            npzfile = np.load(aspPath)
            self.tailCurvature_mean=npzfile['tailCurvature_mean']
            self.fish_orientation_elipse=npzfile['ori']
            self.centroidContour=npzfile['centroidContour']
    
    def plotOverview(self,condition='notDefined'):
        
#        for i in range(2):
#            self.pair.animals[i].plot_errorAngle()
#            self.pair.animals[i].plot_d_error_over_d_IAD()
#            
#        for j in range(10):
#            for i in range(2):
#                self.sPair[j].animals[i].plot_errorAngle()
#                self.sPair[j].animals[i].plot_d_error_over_d_IAD()
            
        outer_grid = gridspec.GridSpec(4, 4)        
        plt.figure(figsize=(8, 8))   
        #plt.text(0,0.95,path)
        plt.figtext(0,.01,self.expInfo.trajectoryPath)
        plt.figtext(0,.03,condition)
        plt.figtext(0,.05,self.idQuality)
        
        plt.subplot(4,4,1,rasterized=True)
        plt.cla()
        #plt.plot(self.pair.animals[0].ts.position().x(),self.pair.animals[0].ts.position().y(),'b.',markersize=1,alpha=0.1)
        #plt.plot(self.pair.animals[1].ts.position().x(),self.pair.animals[1].ts.position().y(),'r.',markersize=1,alpha=0.1)
        
        plt.plot(self.pair.animals[0].ts.position().x(),self.pair.animals[0].ts.position().y(),'.',markersize=1,alpha=0.1)
        plt.plot(self.pair.animals[1].ts.position().x(),self.pair.animals[1].ts.position().y(),'.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')  
        
        plt.subplot(4,4,3)
        plt.cla()
        PolhistBins=self.pair.animals[0].ts.PolhistBins()
        plt.step(PolhistBins[:-1], self.pair.animals[0].ts.Pol_n(),lw=2,where='mid')
        plt.step(PolhistBins[:-1], self.pair.animals[1].ts.Pol_n(),lw=2,where='mid')
        plt.axhline(1,ls=':',color='k')

        plt.xlabel('dist from center [mm]')
        plt.ylabel('p')
        plt.title('thigmotaxis') 
        #plt.ylim([0, .1])

        
        
        #plot IAD time series
        IAD=self.pair.IAD()
        x=np.arange(float(np.shape(IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,IAD,'k.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        #IAD histogram for raw and shifted data and simulated random dots
        plt.subplot(4,4,2)
        plt.cla()
        #get rid of nan
        #IAD=self.pair.IAD()
        IAD=IAD[~np.isnan(IAD)]
        histBins=np.arange(100)
        n, bins, patches = plt.hist(IAD, bins=histBins, normed=1, histtype='stepfilled',color='k')
        plt.step(histBins[:-1],self.spIADhist_m.T,lw=1,color=[.5,.5,.5])
        #simulate random uniform spacing
        rad=(self.expInfo.trajectoryDiameterPx/self.expInfo.pxPmm)/2
        num=1000
        dotList,dotND = randSpacing.randomDotsOnCircle(rad,num)
        n, bins, patches = plt.hist(dotND, bins=histBins, normed=1, histtype='step',color='r',linestyle='--')
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        plt.ylim([0, .05])
        
        plt.subplot(4,4,4)
        plt.cla()
        x=[1,2]
        y=[np.nanmean(IAD),self.spIAD_m([])]
        yerr=[0,self.spIAD_std()]
        barlist=plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        barlist[1].set_color([.5,.5,.5])
        lims = plt.ylim()
        plt.ylim([0, lims[1]]) 
        plt.xlim([0.5, 3]) 
        plt.ylabel('[mm]')
        plt.xticks([1.25,2.25],['raw','shift'])
        plt.title('mean IAD')
        
        plt.subplot(4,4,9)
        plt.cla()
        a1=self.pair.animals[0].ts.neighborMat()
        a2=self.pair.animals[1].ts.neighborMat()
        
        meanPosMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        clim=[meanPosMat.min(),meanPosMat.max()]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        def pltArrow(size,color='m'):
            w=size/6.0        
            plt.arrow(
            0,            # x
            -size/2.0,            # y
            0,            # dx
            size,            # dy
            width=w,       # optional - defaults to 1.0
            length_includes_head=True,
            head_width=w*2,
            color=color
            )
            
        #neighborhood matrix for animal0
        #the vertial orientation was confirmed correct in march 2016
        cp=sns.color_palette()
        plt.subplot(4,4,13)
        plt.cla()
        PosMat=a1
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        pltArrow(self.AnSize[0,1],cp[0])
        plt.title('Animal 0 neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,4,14)
        plt.cla()
        PosMat=a2
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        pltArrow(self.AnSize[1,1],cp[1])
        plt.title('Animal 1 neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #LEADERSHIP
        
        plt.subplot(4,4,15)
        plt.cla()
        
        x=[1,2]
        barlist=plt.bar(x,self.LeadershipIndex, width=0.5,color=cp[0])
        barlist[1].set_color(cp[1])
        plt.title('Leadership')
        plt.ylabel('index')
        plt.ylim([-1, 1])
        plt.xlim([0.5, 3])
        
        tmp=[an.at_bottom_of_dish_stack() for an in self.pair.animals]
        #if np.sum(tmp)==0 or self.expInfo.txtTrajectories:
        plt.xticks([1.25,2.25],['same','dish'])

       # else:
       #     xtickList=[int(np.where(self.pair.StackTopAnimal==1)[0])+1.25,int(np.where(self.pair.StackTopAnimal==0)[0])+1.25]
       #     plt.xticks(xtickList,['top','bottom'])            
        
        #BODY SIZE
        plt.subplot(4,4,16)
        plt.cla()
        x=[1,2]
        barlist=plt.bar(x,self.AnSize[:,1], width=0.5,color=cp[0])
        barlist[1].set_color(cp[1])
        plt.xlim([0.5, 3])
        plt.xticks([1.25,2.25],['an0','an1'])
        
        plt.title('Body Size')
        plt.ylabel('length [mm]')
        

        plt.subplot(4,4,10)
        plt.title('accel=f(pos_n)')
        a1=self.pair.animals[0].ts.ForceMat_speedAndTurn()
        a2=self.pair.animals[1].ts.ForceMat_speedAndTurn()
        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[9],31,10)
        
        
        
        
        plt.subplot(4,4,11)
        plt.cla()
        barlist=plt.bar([1,2],self.pair.avgSpeed(),width=0.5)
        barlist[1].set_color(cp[1])
        plt.title('avgSpeed')
        plt.xlim([0.5, 3])
        plt.xticks([1.25,2.25],['an0','an1'])
        try:
            plt.tight_layout()
        except:
            pass
                
        plt.show()
        
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            currentTime=datetime.datetime.now()
            self.pdfPath=self.expInfo.aviPath[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
            with PdfPages(self.pdfPath) as pdf:
                pdf.savefig()
            
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat
#        a2=self.Pair.animals[1].ForceMat
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,10)
#        
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat_speed
#        a2=self.Pair.animals[1].ForceMat_speed
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,.1)
#        
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat_turn
#        a2=self.Pair.animals[1].ForceMat_turn
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,.1)
        
        
        