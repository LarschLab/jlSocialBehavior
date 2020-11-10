import numpy as np
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.experiment_set as es
from functions.peakdet import detect_peaks
from scipy.ndimage import gaussian_filter as gfilt
from collections import defaultdict as ddict
from scipy.stats import binned_statistic_dd
import pickle
import socket
import os
import functions.matrixUtilities_joh as mu
import re
import seaborn as sns

def calc_anglediff(

    unit1,
    unit2,
    theta=np.pi

):

    if unit1 < 0:
        unit1 += 2 * theta

    if unit2 < 0:
        unit2 += 2 * theta

    phi = abs(unit2 - unit1) % (theta * 2)
    sign = 1
    # used to calculate sign
    if not ((unit1 - unit2 >= 0 and unit1 - unit2 <= theta) or (
            unit1 - unit2 <= -theta and unit1 - unit2 >= -2 * theta)):
        sign = -1
    if phi > theta:
        result = 2 * theta - phi
    else:
        result = phi

    return result * sign


def calc_window(

    bout_mean,
    plot=False

):
    '''
    Calculate time window before and after bout for estimating displacement vector per bout
    '''

    peak = bout_mean.argmax()
    w1 = bout_mean[:peak].argmin()
    w2 = bout_mean[peak:].argmin()

    w11 = bout_mean[:w1].argmax()
    w21 = bout_mean[peak + w2:].argmax()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(range(len(bout_mean)), bout_mean)
        ax.axvline(w1)
        ax.axvline(w2 + peak)
        ax.axvline(w11)
        ax.axvline(w2 + w21 + peak)
        plt.show()

    w1 = peak - w1
    w11 = peak - w11

    w21 = w2 + w21

    return w1, w11, w2, w21


def calc_ansess(df):

    n_animals = df.animalIndex.unique().shape[0]

    dates_ids = np.unique([(date.split(' ')[0], anid)
                           for date, anid in zip(df['time'].values, df['animalID'].values)], axis=0)

    n_animals_sess = [dates_ids[np.where(dates_ids[:, 0] == date)[0], 1].astype(int).max() + 1
                      for date in np.unique(dates_ids[:, 0])]

    return n_animals, dates_ids, n_animals_sess


def calc_stats(

    vectors,
    bin_edges,
    dist=None,
    statistic=None,
    statvals=None,
    angles=True

):
    """
    Generates 4dim histogram with either start xy and diff xy of bouts (startx, starty, stopx, stopy)
                                 or with start xy and angles, distances of bouts
    Calculates binned statistic over vectors
    """
    if angles:
        vector_angles = np.arctan2(vectors[:, 3], vectors[:, 2]).reshape(-1, 1)
        vectors = np.concatenate([
            vectors[:, 0:1],
            vectors[:, 1:2],
            vector_angles,
            dist.reshape(-1, 1)], axis=1
        )

    if statistic:

        stats_dim = list()
        for sv in statvals:
            stats, edges, bno = binned_statistic_dd(
                vectors[:, :2],
                sv.reshape(1, -1),
                bins=bin_edges[:2],
                statistic=statistic
            )
            stats_dim.append(stats)

    else:

        stats_dim = None

    hist, edges = np.histogramdd(vectors, bins=bin_edges)
    return hist, stats_dim


def calc_stats_alt(

    vectors,
    bin_edges,
    dist,
    rel_stim_hd

):
    """
    Generates 5dim histogram with start xy and angles, distances of bouts and relative stimulus heading
    """

    vector_angles = np.arctan2(vectors[:, 3], vectors[:, 2]).reshape(-1, 1)
    print(rel_stim_hd.shape, vectors.shape, vector_angles.shape)
    vectors = np.concatenate([
        vectors[:, 0:1],
        vectors[:, 1:2],
        vector_angles,
        dist.reshape(-1, 1),
        rel_stim_hd.reshape(-1, 1)
    ], axis=1
    )

    hist, edges = np.histogramdd(vectors, bins=bin_edges)

    return hist


def calc_nmap_laterality(

        nmap,
        thresh_factor,
        plot=False

):
    nmap_ri = np.concatenate([
        nmap.ravel().reshape(-1, 1),
        np.arange(0, nmap.ravel().shape[0], 1).reshape(-1, 1)
    ], axis=1)

    nmap_ris = nmap_ri[nmap_ri[:, 0].argsort()]
    nmap_cs = np.cumsum(nmap_ris[:, 0])

    hs = nmap.shape[0] / 2

    thresh_idx = np.where(nmap_cs > nmap_cs[-1] * thresh_factor)[0].min()
    top_idx = [int(i) for i in nmap_ris[thresh_idx:, 1]]
    top_idx = np.unravel_index(top_idx, (nmap.shape[0], nmap.shape[1]))

    #     plt.plot(nmap_cs)
    #     plt.axvline(thresh_idx)
    #     plt.show()

    idx_set = set()

    attraction_diffs = []
    for i, j in zip(top_idx[0], top_idx[1]):

        if i < hs:

            diff = 2 * hs - 1 - i
            idx_set.add((int(round(diff)), j))
            idx_set.add((i, j))
            attraction_diff = nmap[i, j] - nmap[int(round(diff)), j]

        else:

            diff = hs - (i - hs) - 1
            idx_set.add((int(round(diff)), j))
            idx_set.add((i, j))
            attraction_diff = nmap[int(round(diff)), j] - nmap[i, j]

        attraction_diffs.append(attraction_diff)

    if plot:
        plt.imshow(nmap.T, origin="lower")
        plt.show()

        idx_set = np.array([i for i in idx_set])
        nmap_c = nmap.copy()
        nmap_c[(idx_set[:, 0], idx_set[:, 1])] = -1
        nmap_c[top_idx] = -2

        nmap_c[0, 10] = 0
        plt.imshow(nmap_c.T, origin='lower')
        plt.show()

    return np.mean(attraction_diffs)

def position_relative_to_neighbor_rot_alt_swapped(

        ts,
        frames_ep,
        **kwargs
):
    neighborpos = ts.animal.neighbor.ts.position_smooth().xy
    n_episodes = int(neighborpos.shape[0] / frames_ep)
    npos0, npos1 = neighborpos[:frames_ep, :], neighborpos[frames_ep:frames_ep * 2, :]
    swapped_neighbor_pos = np.concatenate(
        [npos1, npos0]*(int(round(n_episodes / 2, 0))+1), axis=0)[:neighborpos.shape[0], :]

    position_relative_to_neighbor_swapped = swapped_neighbor_pos - ts.position_smooth().xy
    relPosPol = [mu.cart2pol(position_relative_to_neighbor_swapped.T[0, :], position_relative_to_neighbor_swapped.T[1, :])]
    relPosPolRot = np.squeeze(np.array(relPosPol).T)[:-1, :]
    relPosPolRot[:, 0] = relPosPolRot[:, 0] - ts.heading(**kwargs)

    x = [mu.pol2cart(relPosPolRot[:, 0], relPosPolRot[:, 1])]
    x = np.squeeze(np.array(x).T)

    return x

def plot_vector_field(

        ax,
        hist,
        res,
        bin_edges,
        width=0.3,
        scale=1.,
        sigma=2,
        alpha=1,
        colorvals=None,
        cmap='RdPu',
        clim=(.5, 2.),
        angles=True,
        angles_plot='xy',
        scale_units='xy',
        diffxy=None,
        maxp=False

):
    bin_values = [bins[:-1] + (bins[1] - bins[0]) for bins in bin_edges[:]]

    # Calculate the highest frequency diffxy
    uv_idxs = np.array(
        [np.unravel_index(np.argmax(gfilt(hist[j][i], sigma)), hist[j][i].shape)
         for i in range(hist.shape[1]) for j in range(hist.shape[0])])
    if maxp:
        # Calculate the probability of the highest frequency xy
        uv_max = np.array([np.max(gfilt(hist[j][i], sigma))
                           for i in range(hist.shape[1]) for j in range(hist.shape[0])])

    # Generate meshgrid based on histogram bin edges
    x1, x2 = np.meshgrid(bin_values[0], bin_values[1])

    # Retrieve values for argmax indices for the diffxys
    u = bin_values[2][uv_idxs[:, 0]]
    v = bin_values[3][uv_idxs[:, 1]]

    if angles:

        diffx = np.cos(u) * v
        diffy = np.sin(u) * v
        theta = u

    elif diffxy:

        diffx = diffxy[0]
        diffy = diffxy[1]
        theta = np.arctan2(v, u)

    else:
        # switched x and y for u and v because switched earlier, CORRECTED
        diffx = u
        diffy = v
        theta = np.arctan2(v, u)

    hist_pos = np.sum(hist, axis=(2, 3)) * res[0] * res[1]
    if colorvals is not None:

        theta = colorvals

    else:
        theta = np.array([i + np.pi / 2 if i < np.pi / 2 else -np.pi + i - np.pi / 2 for i in theta])
        clim = (-np.pi, np.pi)

    ax.quiver(x1, x2, diffx, diffy,
              theta,
              clim=clim,
              cmap=cmap,
              units='xy',
              angles=angles_plot,
              scale_units=scale_units,
              scale=scale,
              width=width,
              alpha=alpha,
              color='white'
              )
    return u, v, diffx, diffy, hist_pos

def merge_datasets(

        root='C:/Users/jkappel/PyCharmProjects/jlsocialbehavior/jlsocialbehavior',
        expset_merge=None
):

    """
    Merge datasets into one bout df and one vector array

    :param root: str, data directory
    :param expset_merge: list of lists, -> names of datasets to merge, see below
    :return:
    """

    if not expset_merge:

        expsets_merge = [['jjAblations_smooth_rot', 'jjAblationsGratingLoom_smooth_rot'],
                         ['jjAblations_shifted_smooth_rot', 'jjAblationsGratingLoom_shifted_smooth_rot'],
                         ['jjAblations_smooth_rot_swap_stim', 'jjAblationsGratingLoom_smooth_rot_swap_stim']]

    for mergeset in expset_merge:

        vectors_merged = []
        bout_df_merged = []
        hd_merged = []

        # manually excluded animals, animalIndex from jlsocial df is 1-based, my Animal index as well
        # exdict = {
        #
        #     0: np.array([8, 9, 10, 14, 17, 25, 27, 33, 38, 45, 47, 52, 58, 66, 69, 73]),
        #
        #     1: np.array([6, 13, 24, 34, 37, 38, 39, 45, 48, 50, 56, 57, 58, 62, 63, 64, 76, 84])
        # }

        # exdict = {
        #
        #     0: np.array([4, 8, 9, 10, 13, 14, 17, 18, 19, 20, 21, 25, 27, 34, 35, 36, 38, 39, 41, 45, 47, 58, 62, 66, 69, 73]),
        #
        #     1: np.array([6, 24, 28, 34, 37, 38, 39, 40, 41, 42, 43])
        # }

        exdict = {

            0: np.array([]),

            1: np.array([])

        }
        for exno, expset_name in enumerate(mergeset):

            exclude_animals = exdict[exno]
            print('Exclude animals: ', exclude_animals)
            all_bout_xys = pickle.load(open(os.path.join(root, 'all_bout_xys_{}.p'.format(expset_name)), 'rb'))
            bout_df = pickle.load(open(os.path.join(root, 'bout_df_{}.p'.format(expset_name)), 'rb'))

            anfilt = np.invert([i in exclude_animals for i in bout_df['Animal index'].values])
            df_filt = bout_df[anfilt]

            stim_xys = np.array([j for i in all_bout_xys for j in i[1]])[anfilt]
            stim_hd = np.array([j for i in all_bout_xys for j in i[3]])[anfilt]
            fish_hd = np.array([j for i in all_bout_xys for j in i[4]])[anfilt]

            stim_vectors = np.concatenate([stim_xys[:, 0], stim_xys[:, 1] - stim_xys[:, 0]], axis=1)
            nanfilt = np.invert([any(i) for i in np.isnan(stim_vectors)])
            vectors_filt = stim_vectors[nanfilt]

            startdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 0], fish_hd[:, 0])]
            stopdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 1], fish_hd[:, 1])]
            diffdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stopdiff, startdiff)]
            hd_diffs = np.array([startdiff, stopdiff, diffdiff]).T

            hd_diffs = hd_diffs[nanfilt]
            df_filt = df_filt[nanfilt]

            dset = pd.Series([exno + 1] * df_filt.shape[0], index=df_filt.index)
            df_filt['Dataset'] = dset

            print(vectors_filt.shape)
            vectors_merged.append(vectors_filt)
            bout_df_merged.append(df_filt)
            hd_merged.append(hd_diffs)

        bout_vectors = np.concatenate(vectors_merged, axis=0)
        bout_df_merged = pd.concat(bout_df_merged, sort=False)
        hd_merged = np.concatenate(hd_merged, axis=0)

        pickle.dump(bout_vectors, open(os.path.join(root, 'bout_vectors_{}.p'.format(''.join(mergeset))), 'wb'))
        pickle.dump(hd_merged, open(os.path.join(root, 'hd_diff_{}.p'.format(''.join(mergeset))), 'wb'))
        pickle.dump(bout_df_merged, open(os.path.join(root, 'bout_df_{}.p'.format(''.join(mergeset))), 'wb'))

        return bout_vectors, hd_merged, bout_df_merged


class VectorFieldAnalysis:

    def __init__(self, **kwargs):

        self.base = kwargs.get('base', 'J:/_Projects/J-sq')
        self.expset_name = kwargs.get('expset_name', None)
        self.stim_protocol = kwargs.get('stim_protocol', None)
        self.tag = kwargs.get('tag', '')
        self.swap_stim = kwargs.get('swap_stim', False)
        self.shift = kwargs.get('shift', False)
        self.yflip = kwargs.get('yflip', True)
        self.default_limit = kwargs.get('default_limit', None)
        self.load_expset = kwargs.get('load_expset', False)
        self.cam_height = kwargs.get('cam_height', (105, 180))

        self.bout_crop = kwargs.get('bout_crop', 25)
        self.bout_wl = kwargs.get('bout_wl', 20)
        self.smooth_alg = kwargs.get('smooth_alg', 'hamming')
        self.unique_episodes = kwargs.get('unique_episodes', ['07k01f', '10k20f'])
        self.nmap_res = kwargs.get('nmap_res', (30, 30))
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )
        self. = kwargs.get('', )

        # experiment set parameters
        self.epiDur = kwargs.get('epiDur', 5)
        self.n_episodes = kwargs.get('n_episodes', 60)
        self.inDish = kwargs.get('inDish', 10)
        self.arenaDiameter_mm = kwargs.get('arenaDiameter_mm', 100)
        self.minShift = kwargs.get('minShift', 60)
        self.episodePLcode = kwargs.get('episodePLcode', 0)
        self.recomputeAnimalSize = kwargs.get('recomputeAnimalSize', 0)
        self.SaveNeighborhoodMaps = kwargs.get('SaveNeighborhoodMaps', 1)
        self.computeLeadership = kwargs.get('computeLeadership', 0)
        self.ComputeBouts = kwargs.get('ComputeBouts', 0)
        self. nShiftRuns = kwargs.get('nShiftRuns', 3)
        self.filteredMaps = kwargs.get('filteredMaps', True)

        #setting None attributes
        self.n_animals_sess = None
        self.df = None
        self.limit = None
        self.frames_ep = None
        self.expfile = None
        self.anfile = None
        self.exp_set = None
        self.bout_window = None
        self.all_bout_xys = None

        self.map_paths = glob.glob(os.path.join(base, self.expset_name, '*MapData.npy'))

    def process_dataset(self):

        self.read_experiment_set()
        if self.default_limit is not None:
            self.limit = self.default_limit

        if self.shift:

            self.generate_bout_vectors(

                tag=self.expset_name + '_shifted' + self.tag,
                shifted=True,
                swap_stim=False
            )
        if self.swap_stim:

            self.generate_bout_vectors(

                tag=self.expset_name + '_swap_stim' + self.tag,
                shifted=False,
                swap_stim=True
                    )

        self.generate_bout_vectors(

            tag=self.expset_name + self.tag,
            shifted=False,
            swap_stim=False
        )

        return

    def read_experiment_set(self):

        self.expfile = os.path.join(base, '{}_allExp.xlsx'.format(self.expset_name))
        self.anfile = os.path.join(base, '{}_allAn.xlsx'.format(self.expset_name))
        info = pd.read_excel(self.expfile)
        ix = (info.stimulusProtocol == self.stim_protocol)
        info = info[ix]

        infoAn=pd.read_excel(self.anfile)

        # collect meta information and save to new csv file for batch processing
        posPath = []
        PLPath = []
        expTime = []
        birthDayAll = []
        anIDsAll = []
        camHeightAll = []

        for index, row in info.iterrows():

            start_dir = os.path.join(base, row.path)

            posPath.append(glob.glob(start_dir + 'PositionTxt*.txt')[0])
            PLPath.append(glob.glob(start_dir + 'PL*.*')[0])

            head, tail = os.path.split(posPath[-1])
            currTime = datetime.strptime(tail[-23:-4], '%Y-%m-%dT%H_%M_%S')
            expTime.append(currTime)

            camHeightAll.append(self.cam_height)

            anNrs = row.anNr  # Note that anNrs are 1 based!
            if ':' in anNrs:
                a, b = anNrs.split(sep=':')
                anNrs = np.arange(int(a), int(b) + 1)
            else:
                anNrs = np.array(anNrs.split()).astype(int)

            anIDs = anNrs  # -1 no more 0-based since using pandas merge to find animal numbers
            anIDsAll.extend(anIDs)

            bd = infoAn[infoAn.anNr.isin(anIDs)].bd.values
            # bd=infoAn.bd.values[anIDs-1] #a bit dirty to use anIDs directly here. Should merge
            birthDayAll.append(' '.join(list(bd)))

        info['camHeight'] = camHeightAll
        info['txtPath'] = posPath
        info['pairList'] = PLPath
        info['aviPath'] = 'default'
        info['birthDayAll'] = birthDayAll

        info['epiDur'] = self.epiDur  # duration of individual episodes (default: 5 minutes)
        info['episodes'] = self.n_episodes  # number of episodes to process: -1 to load all episodes (default: -1)
        info['inDish'] = self.inDish  # np.arange(len(posPath))*120     # time in dish before experiments started (default: 10)
        info['arenaDiameter_mm'] = self.arenaDiameter_mm  # arena diameter (default: 100 mm)
        info['minShift'] = self.minShift  # minimum number of seconds to shift for control IAD
        info['episodePLcode'] = self.episodePLcode  # flag if first two characters of episode name encode animal pair matrix (default: 0)
        info['recomputeAnimalSize'] = self.recomputeAnimalSize  # flag to compute animals size from avi file (takes time, default: 1)
        info['SaveNeighborhoodMaps'] = self.SaveNeighborhoodMaps  # flag to save neighborhood maps for subsequent analysis (takes time, default: 1)
        info['computeLeadership'] = self.computeLeadership  # flag to compute leadership index (takes time, default: 1)
        info['ComputeBouts'] = self.ComputeBouts  # flag to compute swim bout frequency (takes time, default: 1)
        # info['set'] = np.arange(len(posPath))   # experiment set: can label groups of experiments (default: 0)
        info['ProcessingDir'] = self.base
        info['outputDir'] = self.base
        info['expTime'] = expTime
        info['nShiftRuns'] = self.nShiftRuns
        info['filteredMaps'] = self.expfile

        csv_file = os.path.join(self.base, 'processingSettings.csv')
        info.to_csv(csv_file, encoding='utf-8')


        if self.load_expset:

            self.exp_set = pickle.load(open(
                os.path.join(self.base, 'exp_set_{}.p'.format(self.expset_name)), 'rb'))

        else:

            self.exp_set = es.experiment_set(csvFile=csv_file, MissingOnly=False)

        csvPath = []
        mapPath = []

        for f in sorted([mu.splitall(x)[-1][:-4] for x in info.txtPath]):

            csvPath.append(glob.glob(self.base + f + '*siSummary*.csv')[0])
            mapPath.append(glob.glob(self.base + f + '*MapData.npy')[0])

        df = pd.DataFrame()
        max_id = 0
        for i, fn in enumerate(sorted(csvPath)):
            print(fn)
            tmp = pd.read_csv(fn, index_col=0, sep=',')
            tmp.animalSet = i
            tmp.animalIndex = tmp.animalIndex + max_id + 1

            # tmp.animalIndex = np.array(anIDsAll)[tmp.animalIndex]

            max_id = tmp.animalIndex.values.max()
            df = pd.concat([df, tmp])

        df['episode'] = [x.strip().replace('_', '') for x in df['episode']]
        self.df = pd.merge(df, infoAn[['anNr', 'line', 'group']], left_on='animalIndex', right_on='anNr', how='left')

        print('df shape', df.shape)

        self.dates_ids = np.unique([(date.split(' ')[0], anid) for date, anid in zip(df['time'].values, df['animalID'].values)],
                              axis=0)
        self.n_animals_sess = [self.dates_ids[np.where(self.dates_ids[:, 0] == date)[0], 1].astype(int).max() + 1 for date in
                          np.unique(self.dates_ids[:, 0])]

        self.limit = info['episodes'].unique()[0] * info['epiDur'].unique()[0] * 30 * 60
        self.frames_ep = (info['epiDur'].unique()[0] * 60 * 30)

        pickle.dump(self.df, open(os.path.join(self.base, 'df_{}.p'.format(self.expset_name)), 'wb'))
        if not self.load_expset:
            pickle.dump(self.exp_set, open(os.path.join(self.base, 'exp_set_{}.p'.format(self.expset_name)), 'wb'))

        return

    def generate_bout_vectors(

        self,
        tag='',
        shifted=False,
        swap_stim=False

    ):
        bout_dir = os.path.join(self.base, 'all_bouts_all_bout_idx_{}.p'.format(tag))
        all_bouts, all_bout_idxs, bout_mean = self.collect_bouts(shifted=shifted)
        pickle.dump([all_bouts, all_bout_idxs, bout_mean], open(
            bout_dir, 'wb'))

        self.bout_window = calc_window(bout_mean)
        all_bout_xys = self.get_bout_positions(

            all_bout_idxs,
            shifted=shifted,
            swap_stim=swap_stim

        )

        pickle.dump(all_bout_xys, open(
            os.path.join(self.base, 'all_bout_xys_{}.p'.format(tag)), 'wb'))

        self.generate_bout_df(all_bout_xys)
        pickle.dump(self.bout_df, open(
            os.path.join(self.base, 'bout_df_{}.p'.format(tag)), 'wb'))

        return

    def collect_bouts(self, shifted=False):

        """
        Collect all bout idxs and all bout periods from all fish
        """

        all_bouts = []
        all_bout_idxs = []
        animal_idx = -1

        for j in range(len(self.n_animals_sess)):

            for i in range(self.n_animals_sess[j]):

                animal_idx += 1
                bouts = []

                print('Animal #', animal_idx + 1)
                if shifted:
                    print('Shift applied to data: ', self.exp_set.experiments[j].shiftList[0])
                    self.exp_set.experiments[j].pair_f[i].shift = [self.exp_set.experiments[j].shiftList[0], 0]

                speed = self.exp_set.experiments[j].pair_f[i].animals[0].ts.speed_smooth()
                bout_idxs = detect_peaks(speed[:self.limit], mph=8, mpd=8)
                for bout in bout_idxs:
                    bouts.append(speed[bout - self.bout_crop:bout + self.bout_crop])

                all_bouts.append((animal_idx, bouts))
                all_bout_idxs.append((animal_idx, bout_idxs))

        bouts_ravel = np.array([j for i in all_bouts for j in i[1] if len(j) == self.bout_crop * 2])
        bout_mean = np.nanmean(bouts_ravel, axis=0)

        return all_bouts, all_bout_idxs, bout_mean

    def get_bout_positions(

            self,
            all_bout_idxs,
            shifted=False,
            swap_stim=False

    ):
        w1, w11, w2, w21 = self.bout_window
        animal_idx = -1
        self.all_bout_xys = dict()

        for j in range(len(self.n_animals_sess)):

            for i in range(self.n_animals_sess[j]):

                animal_idx += 1
                print('Animal #', animal_idx + 1)
                an_episodes = self.df[self.df['animalIndex'] == animal_idx + 1]['episode'].values
                an_group = self.df[self.df['animalIndex'] == animal_idx + 1]['group'].unique()[0]
                bout_idxs = all_bout_idxs[animal_idx][1]

                if animal_idx != all_bout_idxs[animal_idx][0]:
                    raise IndexError('Animal Index does not match! Exiting...')

                if shifted:
                    print('Shift applied to data: ', exp_set.experiments[j].shiftList[0])
                    exp_set.experiments[j].pair_f[i].shift = [exp_set.experiments[j].shiftList[0], 0]

                ts = exp_set.experiments[j].pair_f[i].animals[0].ts
                speed = ts.speed_smooth()[:self.limit]

                if swap_stim:

                    xy_rel = position_relative_to_neighbor_rot_alt_swapped(
                        ts, frames_ep, window=self.smooth_alg, window_len=self.bout_wl)[:self.limit]

                else:

                    xy_rel = ts.position_relative_to_neighbor_rot_alt(window=self.smooth_alg, window_len=self.bout_wl).xy[:self.limit]

                xy_pos = ts.position_smooth().xy[:self.limit]
                hd_f = ts.heading(window=self.smooth_alg, window_len=self.bout_wl)
                hd_s = ts.animal.neighbor.ts.heading(window=self.smooth_alg, window_len=self.bout_wl)

                stim_xys = list()
                fish_xys = list()
                stim_hd = list()
                fish_hd = list()
                bout_episodes = list()

                for bout_idx in bout_idxs:

                    try:

                        idx_pre = speed[bout_idx - w11:bout_idx - w1].argmin() + (bout_idx - w11)
                        idx_post = speed[bout_idx + w2:bout_idx + w21].argmin() + (bout_idx + w2)

                        stim_xys.append((xy_rel[idx_pre], xy_rel[idx_post]))
                        fish_xys.append((xy_pos[idx_pre], xy_pos[idx_post]))
                        stim_hd.append((hd_s[idx_pre], hd_s[idx_post]))
                        fish_hd.append((hd_f[idx_pre], hd_f[idx_post]))

                    except IndexError:

                        print('Could not get pre/post bout idxs: ')
                        print(bout_idx)
                        stim_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                        fish_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                        stim_hd.append((np.nan, np.nan))
                        fish_hd.append((np.nan, np.nan))

                    episode = an_episodes[int(bout_idx / frames_ep)]
                    bout_episodes.append((bout_idx, episode))

                self.all_bout_xys[animal_idx] = {

                    'stim_xys': stim_xys,
                    'fish_xys': fish_xys,
                    'stim_hd': stim_hd ,
                    'fish_hd': fish_hd,
                    'bout_episodes': bout_episodes,
                    'group': an_group

                }
        return

    def calculate_bout_vectors(self):

        '''
        Calculate the distance of each bout, generate df containing all bout information
        '''

        animal_idxs = sorted(self.all_bout_xys.keys())

        # Bout vectors
        self.fish_xys = np.array([fxy for anid in animal_idxs for fxy in self.all_bout_xys[anid]['fish_xys']])
        self.stim_xys = np.array([sxy for anid in animal_idxs for sxy in self.all_bout_xys[anid]['stim_xys']])

        self.fish_vectors = np.concatenate([self.fish_xys[:, 0], self.fish_xys[:, 1] - self.fish_xys[:, 0]], axis=1)
        self.bout_vectors = np.concatenate([self.stim_xys[:, 0], self.stim_xys[:, 1] - self.stim_xys[:, 0]], axis=1)

        self.dist = np.sqrt(self.fish_vectors[:, 3] ** 2 + self.fish_vectors[:, 2] ** 2)
        self.dist[np.isnan(self.dist)] = 0
        print('Mean distance per bout: ', np.nanmean(self.dist))

        # Absolute and relative heading
        self.stim_hd = np.array([sxy for anid in animal_idxs for sxy in self.all_bout_xys[anid]['stim_hd']])
        self.fish_hd = np.array([sxy for anid in animal_idxs for sxy in self.all_bout_xys[anid]['fish']])
        startdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(self.stim_hd[:, 0], self.fish_hd[:, 0])]
        stopdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(self.stim_hd[:, 1], self.fish_hd[:, 1])]
        diffdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stopdiff, startdiff)]
        self.hd_diffs = np.array([startdiff, stopdiff, diffdiff]).T

        # Meta params
        self.bout_animal_idxs = np.concatenate(
            [[anid] * len(self.all_bout_xys[anid]['stim_xys']) for anid in animal_idxs]
            , axis=0)
        self.bout_groups = np.concatenate(
            [[self.all_bout_xys[anid]['group']] * len(self.all_bout_xys[anid]['stim_xys']) for anid in animal_idxs]
            , axis=0)
        self.bout_episodes = np.array([ep[1] for anid in animal_idxs for ep in self.all_bout_xys[anid]['bout_episodes']])
        self.bout_idxs = np.array([ep[0] for anid in animal_idxs for ep in self.all_bout_xys[anid]['bout_episodes']])

        self.bout_df = pd.DataFrame({

            'Episode': self.bout_episodes,
            'Animal index': self.bout_animal_idxs + 1,
            'Bout distance': self.dist,
            'Group': self.bout_groups,
            'Bout index': self.bout_idxs
        })
        print('Shape of bout dataframe: ', self.bout_df.shape)
        return

    def extract_nmaps(self):

        """
        Extracting FILTERED neighborhood maps from each individual animal for all conditions

        :param df:
        :param map_paths:
        :return:

        """

        # Filtered
        self.neighbormaps_bl = np.zeros((self.n_animals, self.nmap_res[0], self.nmap_res[1])) * np.nan
        self.neighbormaps_cont = np.zeros((self.n_animals, self.nmap_res[0], self.nmap_res[1])) * np.nan

        for nmap, stimtype in zip([self.neighbormaps_bl, self.neighbormaps_cont], self.unique_episodes):

            for mapno in range(len(self.map_paths)):

                print(self.map_paths[mapno])
                tmp = np.load(self.map_paths[mapno])
                print(tmp.shape, mapno)
                tmpDf = df[df.animalSet == mapno]
                for a in range(n_animals_sess[mapno]):
                    an = sum(n_animals_sess[:mapno]) + a
                    print(an)
                    dfIdx = (tmpDf.episode == stimtype) & \
                            (tmpDf.animalID == a) & \
                            (tmpDf.inDishTime < self.limit)
                    ix = np.where(dfIdx)[0]
                    nmap[an, :, :] = np.nanmean(tmp[ix, 0, 0, :, :], axis=0)

        # Shifted filtered
        self.sh_neighbormaps_bl = np.zeros((self.n_animals, self.nmap_res[0], self.nmap_res[1])) * np.nan
        self.sh_neighbormaps_cont = np.zeros((self.n_animals, self.nmap_res[0], self.nmap_res[1])) * np.nan

        for nmap, stimtype in zip([self.sh_neighbormaps_bl, self.sh_neighbormaps_cont], self.unique_episodes):

            for mapno in range(len(self.map_paths)):

                print(self.map_paths[mapno])
                tmp = np.load(self.map_paths[mapno])
                print(tmp.shape)
                tmpDf = df[df.animalSet == mapno]
                for a in range(n_animals_sess[mapno]):
                    an = sum(n_animals_sess[:mapno]) + a

                    dfIdx = (tmpDf.episode == stimtype) & \
                            (tmpDf.animalID == a) & \
                            (tmpDf.inDishTime < self.limit)
                    ix = np.where(dfIdx)[0]

                    nmap[an, :, :] = np.nanmean(tmp[ix, 0, 1, :, :], axis=0)
        return

    def generate_mapdict(

        self,
        expset_names,
        exclude_animals,
        datapath='J:/_Projects/J-sq',
        groupsets=(),
        sortlogics=()

    ):
        from collections import defaultdict as ddict
        mapdict = {}
        if isinstance(exclude_animals, dict):
            exan = [idx for key in sorted(exclude_animals.keys()) for idx in exclude_animals[key]]
        else:
            exan = exclude_animals
        if isinstance(expset_names, str):
            expset_names = [expset_names]
        maxsets = [0]

        for sl in sortlogics:
            mapdict[sl] = ddict(list)

        nan = 0
        nex = 0

        for groupset in groupsets:

            gset = '_'.join(groupset)
            gskey_bl, gskey_cont = '_'.join([gset, '10k20f']), '_'.join([gset, '07k01f'])

            print(gskey_bl)
            for dno, dataset in enumerate(expset_names):

                # Dataset # 1- indexed
                gdkey_bl, gdkey_cont = '_'.join([gset, str(dno + 1), '10k20f']), '_'.join([gset, str(dno + 1), '07k01f'])

                print(dataset, gdkey_bl)
                maxset = len(glob.glob(datapath + '/' + dataset + '/*MapData.npy'))
                df_ds = df_merged.loc[
                    (maxsets[dno] <= df_merged.animalSet.values) & (df_merged.animalSet.values < maxsets[dno] + maxset)]
                print(df_ds.animalIndex.unique().shape, '123')
                print(maxsets[dno], maxset + maxsets[dno])
                for group in groupset:

                    if group not in df_ds.group.unique():
                        print(group, 'not found')
                        continue

                    print(group)
                    uan = df_ds.loc[df_ds.group == group].animalIndex.unique()

                    print(group, uan.shape)
                    grkey_bl, grkey_cont = '_'.join([group, '10k20f']), '_'.join([group, '07k01f'])
                    grdkey_bl, grdkey_cont = '_'.join([group, str(dno + 1), '10k20f']), '_'.join(
                        [group, str(dno + 1), '07k01f'])
                    for an in uan:

                        if an in exan:
                            nex += 1
                            nan += 1
                            print('# animals ex', nex)

                            continue

                        nan += 1
                        print('# animals', nan)
                        nmap_bl = neighbormaps_bl[an - 1]
                        nmap_cont = neighbormaps_cont[an - 1]
                        # flip because of UP/DOWN confusion in the raw data acquisition

                        for sl in sortlogics:

                            if sl not in ['dsetwise-gset', 'gsetwise']:

                                mapdict[sl][grkey_bl].append(np.flipud(nmap_bl))
                                mapdict[sl][grkey_cont].append(np.flipud(nmap_cont))

                            else:

                                if not group.endswith('L'):

                                    nmap_bl = np.flipud(nmap_bl)
                                    nmap_cont = np.flipud(nmap_cont)

                                mapdict[sl][gdkey_bl].append(nmap_bl)
                                mapdict[sl][gdkey_cont].append(nmap_cont)

                maxsets.append(maxset)
        pickle.dump(mapdict,
                    open(os.path.join(base, 'mapdict_{}.p'.format(dataset)),
                         'wb'))
        return mapdict

    def collect_stats(

        self,
        bout_df,
        bout_vectors,
        sortlogics=('gsetwise','dsetwise-group', 'dsetwise-gset'),
        groupsets=(),
        statistic=None,
        statvals=None,
        hd_hist=False,
        rel_stim_hd=None,
        dist_type='abs',
        angles=False,
        dist_filter=(0, 30),
        edges_pos=(-20, 20),
        edges_dir=(-12, 12),
        edges_angles=(-np.pi, np.pi),
        edges_dists=(0, 30),
        res=(30, 30, 30, 30)

    ):

        if hd_hist:

            bin_edges = [edges_pos, edges_pos, edges_angles, edges_dists, edges_angles]

        elif angles:

            bin_edges = [edges_pos, edges_pos, edges_angles, edges_dists]

        else:

            bin_edges = [edges_pos, edges_pos, edges_dir, edges_dir]

        bin_edges = [np.linspace(b[0], b[1], res[bno] + 1) for bno, b in enumerate(bin_edges)]

        histograms = {}
        statistics = {}
        for sl in sortlogics:

            histograms[sl] = ddict(list)
            statistics[sl] = ddict(list)

        episodes = bout_df['Episode'].unique()
        datasets = bout_df['Dataset'].unique()
        distances = bout_df['Bout distance'].values

        for dataset in datasets:
            print(''.join(['.'] * 50))
            print('Dataset: ', dataset)
            for groupset in groupsets:

                print('Groupset: ', groupset)
                for group in groupset:

                    anids = bout_df[(bout_df['Group'] == group) & (bout_df['Dataset'] == dataset)]['Animal index'].unique()

                    print('Group: ', group)
                    for anid in anids:

                        print('Animal index: ', anid)
                        for episode in episodes:

                            print('Episode: ', episode)
                            thresh = (
                                    (dist_filter[0] < bout_df['Bout distance'])
                                    & (dist_filter[1] > bout_df['Bout distance'])
                                    & (bout_df['Dataset'] == dataset)
                                    & (bout_df['Group'] == group)
                                    & (bout_df['Animal index'] == anid)
                                    & (bout_df['Episode'] == episode)
                            )

                            vectors_thresh = bout_vectors[np.where(thresh)].copy()

                            print('# of bouts: ', vectors_thresh.shape[0])

                            if group.endswith('L'):

                                vectors_thresh_rev = vectors_thresh.copy()
                                vectors_thresh_rev[:, 0] *= -1
                                vectors_thresh_rev[:, 2] *= -1

                            if dist_type == 'abs':

                                distances_thresh = distances[thresh]

                            else:  # 'rel'

                                distances_thresh = np.sqrt(vectors_thresh[:, 3] ** 2 + vectors_thresh[:, 2] ** 2)

                            if statistic:

                                statvals_thresh = [i[np.where(thresh)] for i in statvals]

                            else:

                                statvals_thresh = None

                            if not hd_hist:

                                hist, uv_stats = calc_stats(

                                    vectors_thresh,
                                    bin_edges,
                                    dist=distances_thresh,
                                    angles=angles,
                                    statistic=statistic,
                                    statvals=statvals_thresh,
                                )
                                if 'dsetwise-gset' in sortlogics or 'gsetwise' in sortlogics:
                                    hist_rev, uv_stats_rev = calc_stats(

                                        vectors_thresh_rev,
                                        bin_edges,
                                        dist=distances_thresh,
                                        angles=angles,
                                        statistic=statistic,
                                        statvals=statvals_thresh,
                                    )

                            else:

                                hist = calc_stats_alt(

                                    vectors_thresh,
                                    bin_edges,
                                    distances_thresh,
                                    rel_stim_hd[thresh]

                                )
                                if 'dsetwise-gset' in sortlogics or 'gsetwise' in sortlogics:
                                    hist_rev = calc_stats_alt(

                                    vectors_thresh_rev,
                                    bin_edges,
                                    distances_thresh,
                                    rel_stim_hd[thresh]

                                    )
                            hist = hist.astype(np.int16)  # changed to int16

                            if group == 'ctr':

                                groupstr = 'wt'

                            else:

                                groupstr = group

                            for sl in sortlogics:

                                if sl == 'groupwise':

                                    gkey = '_'.join([groupstr, episode])
                                    histograms[sl][gkey].append(hist)
                                    if statistic:
                                        statistics[sl][gkey].append(uv_stats)

                                elif sl == 'gsetwise':

                                    gkey = '_'.join([groupset[0], groupset[1], episode])
                                    if groupstr == 'wt':
                                        gkey = '_'.join([groupstr, episode])

                                    histograms[sl][gkey].append(hist_rev)
                                    if statistic:
                                        statistics[sl][gkey].append(uv_stats_rev)

                                elif sl == 'dsetwise-gset':

                                    gkey = '_'.join([groupset[0], groupset[1], str(dataset), episode])
                                    if groupstr == 'wt':
                                        gkey = '_'.join([groupstr, str(dataset), episode])

                                    histograms[sl][gkey].append(hist_rev)
                                    if statistic:
                                        statistics[sl][gkey].append(uv_stats_rev)

                                elif sl == 'dsetwise-group':

                                    gkey = '_'.join([groupstr, str(dataset), episode])
                                    histograms[sl][gkey].append(hist)
                                    if statistic:
                                        statistics[sl][gkey].append(uv_stats)

                            print('Dictionary key: ', gkey)
                            print('Unique hist vals: ', np.unique(hist).shape[0])
                            del vectors_thresh

        return histograms, statistics

    def get_processed_data(

        self,
        base='J:/_Projects/J-sq/',
        expset_name=None,
        exan=None,
        groupsets=[['AblB'], ['CtrE'], ['Bv']],
        sortlogics=['groupwise'],
        dist_type='abs'

    ):

        """
        Get all processed data for analysis
        :param root: str, data directory
        :param expset_merge: list of lists, -> names of datasets to merge, see below
        :return:
        """

        all_bout_xys = pickle.load(open(os.path.join(root, 'all_bout_xys_{}.p'.format(expset_name)), 'rb'))
        bout_df = pickle.load(open(os.path.join(root, 'bout_df_{}.p'.format(expset_name)), 'rb'))

        anfilt = np.invert([i in exan for i in bout_df['Animal index'].values])
        df_filt = bout_df[anfilt]

        stim_xys = np.array([j for i in all_bout_xys for j in i[1]])[anfilt]
        stim_hd = np.array([j for i in all_bout_xys for j in i[3]])[anfilt]
        fish_xys = np.array([j for i in all_bout_xys for j in i[2]])[anfilt]
        fish_hd = np.array([j for i in all_bout_xys for j in i[4]])[anfilt]

        stim_vectors = np.concatenate([stim_xys[:, 0], stim_xys[:, 1] - stim_xys[:, 0]], axis=1)
        nanfilt = np.invert([any(i) for i in np.isnan(stim_vectors)])
        vectors_filt = stim_vectors[nanfilt]

        startdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 0], fish_hd[:, 0])]
        stopdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 1], fish_hd[:, 1])]
        diffdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stopdiff, startdiff)]
        hd_diffs = np.array([startdiff, stopdiff, diffdiff]).T

        hd_diffs = hd_diffs[nanfilt]
        df_filt = df_filt[nanfilt]

        exp_set = pickle.load(open(glob.glob(os.path.join(base, expset_name, 'exp_set*'))[0], 'rb'))

        map_paths = glob.glob(os.path.join(base, expset_name, '/*MapData.npy'))

        df = pickle.load(open(os.path.join(base, 'df_{}.p'.format(expset_name)), 'rb'))
        neighbormaps_bl, neighbormaps_cont, sh_neighbormaps_bl, sh_neighbormaps_cont = extract_nmaps(

            df,
            map_paths,
            analysisstart=0,
            analysisend=540,

        )

        mapdict = generate_mapdict(

            df_merged,
            neighbormaps_bl,
            neighbormaps_cont,
            expset_name,
            exan,
            datapath=base,
            groupsets=groupsets,
            sortlogics=sortlogics
        )
        histograms = pickle.load(open(os.path.join(base, 'histograms_{}_{}_{}_dict.p'.format(
            ''.join(expset_name),
            '_'.join(sortlogics),
            dist_type)), 'rb'))

        return

    def plot_vfs_ind(

            self,
            histograms_abs,
            mapdict,
            sortlogic='dsetwise',
            tag='',
            edges_pos=(-20, 20),
            edges_dir=(-12, 12),
            edges_angles=(-np.pi, np.pi),
            edges_dists=(0, 30),
            res_abs=(30, 30, 90, 90),
            res_rel=(30, 30, 45, 45),
            clim=(0.5, 2.),
            clim_diff=(-.4, .4),
            clim_nmap=(-1, 2),
            cmap='RdPu',
            cmap_diff='coolwarm',
            cmap_nmap='shiftedcwm',
            width=0.25,
            scale_abs=2,
            scale_rel=1,
            sigma=2,
            alpha=.7,
            maxp=False,
            show=False,
            ctr='CtrE'

    ):
        vector_xys_abs = {}
        vector_xys_rel = {}

        be_abs = [np.linspace(b[0], b[1], res_abs[bno] + 1) for bno, b in enumerate([
            edges_pos, edges_pos, edges_angles, edges_dists])]

        be_rel = [np.linspace(b[0], b[1], res_rel[bno] + 1) for bno, b in enumerate([
            edges_pos, edges_pos, edges_dir, edges_dir])]

        groupsets = sorted(histograms_abs.keys())

        for gno, groupset in enumerate(groupsets):

            fig = plt.figure(figsize=(12, 4), dpi=200)
            gs = gridspec.GridSpec(nrows=1, ncols=5, width_ratios=[1, 1, 1, .1, .1], height_ratios=[1])
            gs.update(wspace=0.8, hspace=0.4)

            print(groupset, len(histograms_abs[groupset]))
            hist_abs = np.mean([histogram / np.sum(histogram) for histogram in histograms_abs[groupset]], axis=0)
            n = len(histograms_abs[groupset])
            print(hist_abs.shape, hist_abs.min(), hist_abs.max())
            print(np.where(np.isnan(hist_abs))[0].shape)
            if '07' in groupset:

                label = '_'.join(groupset.split('_')[:-1]) + ' continuous' + ', n=' + str(n)

            else:

                label = '_'.join(groupset.split('_')[:-1]) + ' bout-like' + ', n=' + str(n)

            episode = groupset.split('_')[-1]
            ax0 = plt.subplot(gs[0, 2])
            angles_abs, dists_abs, diffx, diffy, hist_pos = plot_vector_field(

                ax0,
                hist_abs,
                res_abs,
                be_abs,
                width=width,
                scale=scale_abs,
                sigma=sigma,
                cmap='coolwarm',
                clim=clim,
                angles=True,
                angles_plot='xy',
                scale_units='xy',
                maxp=maxp,
                alpha=alpha
            )

            nmap = np.nanmean(mapdict[sortlogic][groupset], axis=0)
            vector_xys_abs[groupset] = (diffx, diffy, angles_abs, dists_abs, hist_pos, nmap)
            ax1 = plt.subplot(gs[0, 0])
            nmap_im = ax1.imshow(nmap.T, origin='lower', cmap=cmap_nmap, clim=clim_nmap, extent=(-19.5, 20.5, -19.5, 20.5))
            ax1.set_xlim(-19.5, 20.5)
            ax1.set_ylim(-19.5, 20.5)
            ax1.set_ylabel(label)
            ax2 = plt.subplot(gs[0, 1])
            bp_im = ax2.imshow(hist_pos.T, origin='lower', clim=clim, extent=(-19.5, 20.5, -19.5, 20.5), cmap=cmap)
            ax2.set_xlim(-19.5, 20.5)
            ax2.set_ylim(-19.5, 20.5)
            ax3 = plt.subplot(gs[0, 3])
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-180, vmax=180))
            # fake up the array of the scalar mappable
            sm._A = []
            clb = plt.colorbar(sm, cax=ax3, use_gridspec=True, label='Relative bout angle', pad=.2)

            ax4 = plt.subplot(gs[0, 4])
            clb = plt.colorbar(nmap_im, cax=ax4, use_gridspec=True, label='Fold-change from chance', pad=.2)

            ax3.yaxis.set_label_position('left')
            ax4.yaxis.set_label_position('left')
            for ax in [ax0, ax1, ax2]:
                ax.set_aspect('equal')

            ax0.set_title('Bout vector field')
            ax1.set_title('Neighbor density')
            ax2.set_title('Bout probability')
            plt.savefig('{}_plot0_{}.png'.format(groupset, tag), bbox_inches='tight')

            if show:

                plt.show()

            else:

                plt.close()

        scales = [scale_abs, scale_rel]
        for gno, groupset in enumerate(groupsets):

            dset = re.findall('_\d+_', groupset)
            print(groupset, dset)
            if len(dset) == 0:

                dset = ''

            else:

                dset = dset[0][:-1]

            wt_bl = '{}{}_10k20f'.format(ctr, dset)
            wt_cont = '{}{}_07k01f'.format(ctr, dset)
            print(wt_bl, wt_cont)
            diffx_abs, diffy_abs, angles_abs, dists_abs, hist_abs, nmap = vector_xys_abs[groupset]
            # dists_rel = np.sqrt(vector_xys_rel[groupset][0] ** 2 + vector_xys_rel[groupset][1] ** 2)

            if groupset == wt_bl or '07k01f' in groupset and not ctr in groupset:

                diffx_cont, diffy_cont, angles_cont, _, hist_cont, nmap_cont = vector_xys_abs[wt_cont]
                # dists_cont = np.sqrt(vector_xys_rel[wt_cont][0] ** 2 + vector_xys_rel[wt_cont][1] ** 2)

                diffangles = np.array([calc_anglediff(i, j, theta=np.pi) for i, j in zip(angles_abs, angles_cont)])
                print(angles_abs.shape, angles_cont.shape, diffangles.shape)
                print(len(diffangles), diffangles[0].shape)
                # diffdists = dists_rel - dists_cont
                hist_pos = hist_abs - hist_cont
                diffdensity = nmap - nmap_cont
            else:

                diffx_bl, diffy_bl, angles_bl, _, hist_bl, nmap_bl = vector_xys_abs[wt_bl]
                # dists_bl = np.sqrt(vector_xys_rel[wt_bl][0] ** 2 + vector_xys_rel[wt_bl][1] ** 2)

                diffangles = np.array(
                    [calc_anglediff(i, j, theta=np.pi) for i, j in zip(angles_abs, angles_bl)])
                # diffdists = dists_rel - dists_bl
                hist_pos = hist_abs - hist_bl
                diffdensity = nmap - nmap_bl

            fig = plt.figure(figsize=(12, 4), dpi=200)
            gs = gridspec.GridSpec(nrows=1, ncols=5, width_ratios=[1, 1, 1, .1, .1], height_ratios=[1])
            gs.update(wspace=0.8, hspace=0.4)

            ax5 = plt.subplot(gs[0, 0])
            bin_values = [bins[:-1] + (bins[1] - bins[0]) for bins in be_abs[:2]]
            #         x1, x2 = np.meshgrid(bin_values[0], bin_values[1])
            #         ax5.quiver(x1, x2, x1/x1, x2/x2,
            #                    diffangles,
            #                    #clim=clim_diff,
            #                    cmap='coolwarm',
            #                    units='xy',
            #                    angles=np.rad2deg(diffangles)-90,
            #                    scale_units=None,
            #                    scale=1,
            #                    width=width,
            #                    alpha=alpha
            #                  )
            ax5.imshow(diffangles.reshape(30, 30), origin='lower', cmap='coolwarm')
            ax5.set_aspect('equal')

            if gno == 0:
                ax5.set_title('Δ Angles')
            ax6 = plt.subplot(gs[0, 1])
            im_diffd = ax6.imshow(
                diffdensity.T,
                origin='lower',
                cmap='coolwarm',
                clim=clim_diff,
                extent=(-29.5, 30.5, -29.5, 30.5)

            )
            ax6.set_xlim(-19.5, 20.5)
            ax6.set_ylim(-19.5, 20.5)

            ax6.set_title('Δ Neighbor density')

            ax7 = plt.subplot(gs[0, 2])
            ax8 = plt.subplot(gs[0, 3])
            ax9 = plt.subplot(gs[0, 4])

            ax7.set_title('Δ Bout probability')

            im = ax7.imshow(hist_pos.T, origin='lower', clim=clim_diff, extent=(-19.5, 20.5, -19.5, 20.5), cmap=cmap_diff)
            clb = plt.colorbar(im_diffd, cax=ax8, use_gridspec=True, label='Δ Fold-change ND', pad=.2)
            ax8.yaxis.set_label_position('left')

            clb = plt.colorbar(im, cax=ax9, use_gridspec=True, label='Δ Fold-change BP', pad=.2)
            ax9.yaxis.set_label_position('left')
            plt.savefig('{}_plot1_{}.png'.format(groupset, tag), bbox_inches='tight')

            if show:

                plt.show()

            else:

                plt.close()
        return vector_xys_abs, vector_xys_rel


    def plot_attraction(

        self,
        df_merged,
        groups=[],
        expset_name=''
    ):
        dfEpiAn = df_merged[np.any([(df_merged['group'] == group) for group in groups], axis=0)]
        dfEpiAn = dfEpiAn.groupby(['episode', 'animalIndex', 'line', 'group', 'anSize'], sort=True).mean().reset_index()
        dfEpiAn = dfEpiAn.sort_values('group')

        fig, ax = plt.subplots(dpi=300)
        # sns.stripplot(data=dfEpiAn,x='episode',y='si',zorder=-1,hue='group',dodge=10, jitter=True, ax=ax, alpha=.5)
        sns.swarmplot(data=dfEpiAn, x='episode', y='si', zorder=-1, hue='group', dodge=10, ax=ax, alpha=.5)

        sns.pointplot(data=dfEpiAn, ci='sd', x='episode', hue='group', y='si', estimator=np.median, ax=ax, dodge=.5,
                      jitter=False, linestyles=['none'] * 5, lw=1)

        ax.set_xticklabels(['continuous', 'bout-like'])
        ax.set_ylabel('Virtual attraction')
        handles, labels = ax.get_legend_handles_labels()
        labels_legend = []
        for group in dfEpiAn['group'].unique():
            labels_legend.append(
                '{}, n={}'.format(group, dfEpiAn[dfEpiAn['group'] == group]['animalIndex'].unique().shape[0]))

        l = plt.legend(handles[0:5], labels_legend, loc='upper left', borderaxespad=0.)
        plt.savefig('attraction_animalsbygroup_{}.png'.format(expset_name), bbox_inches='tight')
        plt.show()





if __name__ == "__main__":

    for_jl = False
    # boolean for real data, y-axis flippled
    yflip = True
    load_data = False
    root = 'C:/Users/jkappel/PyCharmProjects/jlsocialbehavior/jlsocialbehavior'
    if for_jl:

        # If you fill in the exp_set and the respective dataframe here as well as the n_animals_sess and frames_ep,
        # it might just run through with some tweaking, at least until the generation of the histograms. Further explanation
        # for the variables can be found in generate_bout_vectors

        exp_set = ...
        n_animals_sess = [15, 15, 15]
        df = ...
        frames_ep = 9000

        all_bout_xys, bout_df = generate_bout_vectors(

            exp_set,
            n_animals_sess,
            df,
            frames_ep

        )

        bout_vectors, hd_merged, df_merged = merge_datasets(

            root=root,
            expset_merge=[['Data1', 'Data2']]
        )

    elif load_data:

        mergeset = ['jjAblations_shifted_smooth_rot', 'jjAblationsGratingLoom_shifted_smooth_rot']
        bout_vectors = pickle.load(open(os.path.join(root, 'bout_vectors_{}.p'.format(''.join(mergeset))), 'rb'))
        bout_df = pickle.load(open(os.path.join(root, 'bout_df_{}.p'.format(''.join(mergeset))), 'rb'))
        hd_merged = pickle.load(open(os.path.join(root, 'hd_diff_{}.p'.format(''.join(mergeset))), 'rb'))

    else:

        # base = analyse_datasets(
        #
        #     expset_names=('jjAblations', 'jjAblationsGratingLoom'),
        #     stim_protocols=('boutVsSmooth', 'boutVsSmooth_grateloom'),
        #     default_limit=300*30*60,
        #     load_expset=True
        #
        # )
        #mergeset = ['jjAblations_smooth_rot', 'jjAblationsGratingLoom_smooth_rot']


        base = r'J:/_Projects/J-sq'
        base = process_datasets(
            expset_names=['jjAblationsBilateral'],
            stim_protocols=['boutVsSmooth_grateloom'],
            tag='',
            shift=False,
            swap_stim=False
        )
        mergeset = ['jjAblationsBilateral']
        bout_vectors, hd_merged, bout_df = merge_datasets(

            expset_merge=[mergeset],
            root=base
        )

    # switching sign of the first dimension due to flipping of y axis in the data preprocessing
    if yflip:

        bout_vectors[:, 0] *= -1
        bout_vectors[:, 2] *= -1

    dist_types = ['abs']
    #sortlogics = ['gsetwise', 'dsetwise-gset', 'dsetwise-group']
    # groupsets = [
    #
    #     ['LsR', 'LsL'],
    #     ['AblR', 'AblL'],
    #     ['CtrR', 'CtrL'],
    #     ['ctr', 'wt']
    #
    # ]

    sortlogics = ['groupwise', 'dsetwise-group']
    groupsets = [

        ['CtrE'],
        ['AblB'],
        ['Bv']

    ]
    for dist_type, res, angles in zip(dist_types, [(30, 30, 90, 90, 90)], [True]):
        print(hd_merged.shape)
        hists, _ = collect_stats(
            bout_df,
            bout_vectors,
            groupsets=groupsets,
            sortlogics=sortlogics,
            statistic=None,
            statvals=None,
            hd_hist=False,
            rel_stim_hd=hd_merged[:, 0],
            angles=angles,
            dist_type=dist_type,
            dist_filter=(0, 30),
            edges_pos=(-20, 20),
            edges_dir=(-12, 12),
            edges_angles=(-np.pi, np.pi),
            edges_dists=(0, 30),
            res=res
        )

        pickle.dump(hists, open(os.path.join(base, 'histograms_{}_{}_{}_dict.p'.format(
            ''.join(mergeset),
            ('_').join(sortlogics),
            dist_type)), 'wb'))


