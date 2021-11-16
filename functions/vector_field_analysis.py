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
        self.cam_height = kwargs.get('cam_height', [105, 180])
        self.fps = kwargs.get('fps', 30)

        self.bout_crop = kwargs.get('bout_crop', 25)
        self.smooth_wl = kwargs.get('smooth_wl', 20)
        self.smooth_alg = kwargs.get('smooth_alg', 'hamming')
        self.unique_episodes = kwargs.get('unique_episodes', ['07k01f', '10k20f'])
        self.groupsets = kwargs.get('groupsets', [])
        self.sortlogics = kwargs.get('sortlogics', ['groupwise'])

        self.nmap_res = kwargs.get('nmap_res', (30, 30))
        self.dist_filter = kwargs.get('dist_filter', (0, 30))
        self.edges_pos = kwargs.get('edges_pos', (-20, 20))
        self.edges_dir = kwargs.get('edges_dir', (-12, 12))
        self.edges_angles = kwargs.get('edges_angles', (-np.pi, np.pi))
        self.edges_dists = kwargs.get('edges_dists', (0, 30))
        self.vmap_res = kwargs.get('vmap_res', (30, 30, 30, 30))
        self.revtag = kwargs.get('revtag', 'L')
        self.abs_dist = kwargs.get('abs_dist', True)
        self.exclude_anids = kwargs.get('exculde_animals', [])
        self.calc_thigmo_thresh = kwargs.get('calc_thigmo_thresh', True)
        self.thigmothresh = kwargs.get('thigmothresh', 35.)
        # self. = kwargs.get('', )

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
        self.computeBouts = kwargs.get('computeBouts', 0)
        self.nShiftRuns = kwargs.get('nShiftRuns', 3)
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
        self.stim_xys = None
        self.fish_xys = None
        self.stim_vectors = None
        self.fish_vectors = None
        self.bout_vectors = None
        self.hd_diffs = None
        self.dist = None
        self.stim_hd = None
        self.fish_hd = None
        self.bout_df = None
        self.neighbormaps_bl = None
        self.neighbormaps_cont = None
        self.sh_neighbormaps_bl = None
        self.sh_neighbormaps_cont = None
        self.histograms = None
        self.mapdict = None

        self.map_paths = glob.glob(os.path.join(self.base, self.expset_name, '*MapData.npy'))

    def load_dataset(self):

        ";;;"

    def process_dataset(self):

        self.read_experiment_set()
        if self.calc_thigmo_thresh:

            self.exclude_anids, exan_dict = get_thigmo_thresh(self.df, thresh=self.thigmothresh)

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

    def process_nmaps(self):

        self.read_experiment_set()
        self.extract_nmaps()

    def read_experiment_set(self):

        self.expfile = os.path.join(self.base, self.expset_name, '{}_allExp.xlsx'.format(self.expset_name))
        self.anfile = os.path.join(self.base, self.expset_name, '{}_allAn.xlsx'.format(self.expset_name))
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

            print(self.base, self.expset_name, row.path)
            start_dir = os.path.join(self.base, self.expset_name, row.path)
            print(start_dir)
            posPath.append(glob.glob(start_dir + '/PositionTxt*.txt')[0])
            PLPath.append(glob.glob(start_dir + '/PL*.*')[0])

            head, tail = os.path.split(posPath[-1])
            currTime = datetime.strptime(tail[-23:-4], '%Y-%m-%dT%H_%M_%S')
            expTime.append(currTime)

            camHeightAll.append(self.cam_height[('_dn_' in head) * 1])

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
        info['ComputeBouts'] = self.computeBouts  # flag to compute swim bout frequency (takes time, default: 1)
        info['ProcessingDir'] = self.base
        info['outputDir'] = self.base
        info['expTime'] = expTime
        info['nShiftRuns'] = self.nShiftRuns
        info['filteredMaps'] = self.expfile
        #info['readLim'] = 300

        csv_file = os.path.join(self.base, 'processingSettings.csv')
        info.to_csv(csv_file, encoding='utf-8')

        if self.load_expset:

            self.exp_set = pickle.load(open(
                os.path.join(self.base, 'exp_set_{}.p'.format(self.expset_name)), 'rb'))

        else:

            self.exp_set = es.experiment_set(csvFile=csv_file, MissingOnly=False)

        csvPath = []
        mapPath = []
        print([mu.splitall(x)[-1][:-4] for x in info.txtPath])
        for f in sorted([mu.splitall(x)[-1][:-4] for x in info.txtPath]):

            csvPath.append(glob.glob(os.path.join(self.base, f + '*siSummary*.csv'))[0])
            mapPath.append(glob.glob(os.path.join(self.base, f + '*MapData.npy'))[0])

        df = pd.DataFrame()
        max_id = 0
        for i, fn in enumerate(sorted(csvPath)):
            print(fn)
            tmp = pd.read_csv(fn, index_col=0, sep=',')
            tmp.animalSet = i
            tmp.animalIndex = tmp.animalIndex + max_id + 1
            #tmp.animalIndex = np.array(anIDsAll)[tmp.animalIndex]

            max_id = tmp.animalIndex.values.max()
            df = pd.concat([df, tmp])

        df['episode'] = [x.strip().replace('_', '') for x in df['episode']]
        self.df = pd.merge(df, infoAn[['anNr', 'line', 'group']], left_on='animalIndex', right_on='anNr', how='left')

        print('df shape', df.shape)

        self.dates_ids = np.unique([(set_id, anid) for set_id, anid in zip(df['animalSet'].values, df['animalID'].values)],
                              axis=0)
        self.n_animals_sess = [self.dates_ids[np.where(self.dates_ids[:, 0] == set_id)[0], 1].astype(int).max() + 1 for set_id in
                          np.unique(self.dates_ids[:, 0])]

        self.limit = info['episodes'].unique()[0] * info['epiDur'].unique()[0] * self.fps * 60
        self.frames_ep = info['epiDur'].unique()[0] * self.fps * 60

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
        print('generating bout vectors')
        bout_dir = os.path.join(self.base, 'all_bouts_all_bout_idx_{0}{1}.p'.format(self.expset_name, tag))
        all_bouts, all_bout_idxs, bout_mean = self.collect_bouts(shifted=shifted)
        pickle.dump([all_bouts, all_bout_idxs, bout_mean], open(
            bout_dir, 'wb'))

        self.bout_window = calc_window(bout_mean)
        all_bout_xys = self.get_bout_positions(
            all_bout_idxs,
            shifted=shifted,
            swap_stim=swap_stim,
            tag=tag
        )

        self.calc_bout_vectors(all_bout_xys, tag=tag)
        self.extract_nmaps(tag=tag)
        self.collect_vector_hists(tag=tag)

        return

    def collect_bouts(self, shifted=False):

        """
        Collect all bout idxs and all bout periods from all fish
        """

        print('collecting bouts')
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
            swap_stim=False,
            tag=''

    ):
        print('getting bout positions')
        w1, w11, w2, w21 = self.bout_window
        animal_idx = -1
        all_bout_xys = dict()

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
                    print('Shift applied to data: ', self.exp_set.experiments[j].shiftList[0])
                    self.exp_set.experiments[j].pair_f[i].shift = [self.exp_set.experiments[j].shiftList[0], 0]

                ts = self.exp_set.experiments[j].pair_f[i].animals[0].ts

                #print(ts.yflip, 'ts yflip attribute')
                ts.yflip = self.yflip
                ts.animal.neighbor.ts.yflip = self.yflip
                print(ts.yflip, 'ts yflip attribute')

                speed = ts.speed_smooth()[:self.limit]

                if swap_stim:

                    xy_rel = position_relative_to_neighbor_rot_alt_swapped(
                        ts, self.frames_ep, window=self.smooth_alg, window_len=self.smooth_wl)[:self.limit]

                else:

                    xy_rel = ts.position_relative_to_neighbor_rot_alt(window=self.smooth_alg, window_len=self.smooth_wl).xy[:self.limit]

                xy_pos = ts.position_smooth().xy[:self.limit]
                hd_f = ts.heading(window=self.smooth_alg, window_len=self.smooth_wl)
                hd_s = ts.animal.neighbor.ts.heading(window=self.smooth_alg, window_len=self.smooth_wl)

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

                    except:

                        print('Could not get pre/post bout idxs: ')
                        print(bout_idx)
                        stim_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                        fish_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                        stim_hd.append((np.nan, np.nan))
                        fish_hd.append((np.nan, np.nan))

                    episode = an_episodes[int(bout_idx / self.frames_ep)]
                    bout_episodes.append((bout_idx, episode))

                all_bout_xys[animal_idx] = {

                    'stim_xys': stim_xys,
                    'fish_xys': fish_xys,
                    'stim_hd': stim_hd ,
                    'fish_hd': fish_hd,
                    'bout_episodes': bout_episodes,
                    'group': an_group

                }

        pickle.dump(all_bout_xys, open(
            os.path.join(self.base, 'all_bout_xys_{0}{1}.p'.format(self.expset_name, tag)), 'wb'))
        return all_bout_xys

    def calc_bout_vectors(self, all_bout_xys, tag=''):

        '''
        Calculate the distance of each bout, generate df containing all bout information
        '''

        animal_idxs = sorted(all_bout_xys.keys())

        # Bout vectors
        self.fish_xys = np.array([fxy for anid in animal_idxs for fxy in all_bout_xys[anid]['fish_xys']])
        self.stim_xys = np.array([sxy for anid in animal_idxs for sxy in all_bout_xys[anid]['stim_xys']])

        self.fish_vectors = np.concatenate([self.fish_xys[:, 0], self.fish_xys[:, 1] - self.fish_xys[:, 0]], axis=1)
        self.bout_vectors = np.concatenate([self.stim_xys[:, 0], self.stim_xys[:, 1] - self.stim_xys[:, 0]], axis=1)

        self.dist = np.sqrt(self.fish_vectors[:, 3] ** 2 + self.fish_vectors[:, 2] ** 2)
        self.dist[np.isnan(self.dist)] = 0
        print('Mean distance per bout: ', np.nanmean(self.dist))

        # Absolute and relative heading
        self.stim_hd = np.array([sxy for anid in animal_idxs for sxy in all_bout_xys[anid]['stim_hd']])
        self.fish_hd = np.array([sxy for anid in animal_idxs for sxy in all_bout_xys[anid]['fish_hd']])
        startdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(self.stim_hd[:, 0], self.fish_hd[:, 0])]
        stopdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(self.stim_hd[:, 1], self.fish_hd[:, 1])]
        diffdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stopdiff, startdiff)]
        self.hd_diffs = np.array([startdiff, stopdiff, diffdiff]).T

        # Meta params
        bout_animal_idxs = np.concatenate(
            [[anid] * len(all_bout_xys[anid]['stim_xys']) for anid in animal_idxs]
            , axis=0)
        bout_groups = np.concatenate(
            [[all_bout_xys[anid]['group']] * len(all_bout_xys[anid]['stim_xys']) for anid in animal_idxs]
            , axis=0)
        bout_episodes = np.array([ep[1] for anid in animal_idxs for ep in all_bout_xys[anid]['bout_episodes']])
        bout_idxs = np.array([ep[0] for anid in animal_idxs for ep in all_bout_xys[anid]['bout_episodes']])

        self.bout_df = pd.DataFrame({

            'Episode': bout_episodes,
            'Animal index': bout_animal_idxs + 1,  # Adding +1 for 1-indexing
            'Bout distance': self.dist,
            'Group': bout_groups,
            'Bout index': bout_idxs
        })
        print('Shape of bout dataframe: ', self.bout_df.shape)
        pickle.dump(self.bout_df, open(os.path.join(self.base, 'bout_df_{0}{1}.p'.format(self.expset_name, tag)), 'wb'))
        del all_bout_xys

        return

    def extract_nmaps(self, tag=''):

        """
        Extracting FILTERED neighborhood maps from each individual animal for all conditions
        :return:

        """

        n_animals = sum(self.n_animals_sess)
        neighbormaps = np.zeros((n_animals, self.nmap_res[0], self.nmap_res[1])) * np.nan

        for sidx, shift in zip([0, 1], [False, True]):
            for stimtype in self.unique_episodes:

                nmap = neighbormaps.copy()
                for mapno in range(len(self.map_paths)):

                    print(self.map_paths[mapno])
                    tmp = np.load(self.map_paths[mapno])
                    print(tmp.shape, mapno)
                    tmpDf = self.df[self.df.animalSet == mapno]
                    for a in range(self.n_animals_sess[mapno]):
                        an = sum(self.n_animals_sess[:mapno]) + a
                        print(an)
                        dfIdx = (tmpDf.episode == stimtype) & \
                                (tmpDf.animalID == a) & \
                                (tmpDf.inDishTime < self.limit)
                        ix = np.where(dfIdx)[0]
                        nmap[an, :, :] = np.nanmean(tmp[ix, 0, sidx, :, :], axis=0)

                if '07' in stimtype:
                    if shift:
                        self.sh_neighbormaps_cont = nmap
                    else:
                        self.neighbormaps_cont = nmap
                else:
                    if shift:
                        self.sh_neighbormaps_bl = nmap
                    else:
                        self.neighbormaps_bl = nmap

        self.generate_mapdict(tag=tag)
        return

    def generate_mapdict(self, tag=''):

        self.mapdict = {}
        for sl in self.sortlogics:
            self.mapdict[sl] = ddict(list)

        nan = 0
        nex = 0

        if self.groupsets == []:

            groups = sorted(self.df.group.unique())
            self.groupsets = [[group] for group in groups]

        for groupset in self.groupsets:

            if len(groupset) > 1:
                gset = '_'.join(groupset)
                gskey_bl, gskey_cont = '_'.join([gset, '10k20f']), '_'.join([gset, '07k01f'])
            else:
                if 'gsetwise' in self.sortlogics:
                    print('gsetwise in sortlogics but no groupsets defined')
                    pass
                else:
                    pass

            for group in groupset:

                if group not in self.df.group.unique():
                    print(group, 'not found')
                    continue

                anids = sorted(self.df.loc[self.df.group == group].animalIndex.unique())
                print(group, len(list(anids)))
                if group == 'ctr':

                    group = 'wt'

                grkey_bl, grkey_cont = '_'.join([group, '10k20f']), '_'.join([group, '07k01f'])
                for anid in anids:

                    if anid in list(self.exclude_anids):

                        nex += 1
                        nan += 1
                        print('Animal excluded', anid)

                        continue

                    nan += 1
                    print('# animals', nan)
                    nmap_bl = self.neighbormaps_bl[anid - 1]
                    nmap_cont = self.neighbormaps_cont[anid - 1]

                    # yflip because of UP/DOWN confusion in the raw data acquisition
                    # TODO: implement yflip in the nmap calculation
                    if self.yflip:

                        nmap_bl = np.flipud(nmap_bl)
                        nmap_cont = np.flipud(nmap_cont)

                    for sl in self.sortlogics:

                        if sl == 'groupwise':

                            self.mapdict[sl][grkey_bl].append(nmap_bl)
                            self.mapdict[sl][grkey_cont].append(nmap_cont)

                        elif sl == 'gsetwise':

                            if self.revtag:

                                if group.endswith(self.revtag):

                                    nmap_bl = np.flipud(nmap_bl)
                                    nmap_cont = np.flipud(nmap_cont)

                            self.mapdict[sl][gskey_bl].append(nmap_bl)
                            self.mapdict[sl][gskey_cont].append(nmap_cont)

        pickle.dump(self.mapdict,
                    open(os.path.join(self.base, 'mapdict_{0}{1}.p'.format(self.expset_name, tag)),
                         'wb'))
        return

    def collect_vector_hists(self, tag=''):

        bin_edges = [self.edges_pos, self.edges_pos, self.edges_angles, self.edges_dists]
        bin_edges = [np.linspace(b[0], b[1], self.vmap_res[bno] + 1) for bno, b in enumerate(bin_edges)]

        self.histograms = {}

        for sl in self.sortlogics:

            self.histograms[sl] = ddict(list)

        episodes = self.bout_df['Episode'].unique()
        distances = self.bout_df['Bout distance'].values

        for groupset in self.groupsets:

            print('Groupset: ', groupset)
            for group in groupset:

                anids = self.bout_df[(self.bout_df['Group'] == group)]['Animal index'].unique()

                print('Group: ', group)
                for anid in anids:

                    if anid in self.exclude_anids:

                        print('Animal excluded', anid)
                        continue

                    print('Animal index: ', anid)
                    for episode in episodes:

                        print('Episode: ', episode)
                        thresh = (
                                (self.dist_filter[0] < self.bout_df['Bout distance'])
                                & (self.dist_filter[1] > self.bout_df['Bout distance'])
                                & (self.bout_df['Group'] == group)
                                & (self.bout_df['Animal index'] == anid)
                                & (self.bout_df['Episode'] == episode)
                        )

                        vectors_thresh = self.bout_vectors[np.where(thresh)].copy()

                        print('# of bouts: ', vectors_thresh.shape[0])

                        if self.revtag:
                            if group.endswith(self.revtag):

                                vectors_thresh_rev = vectors_thresh.copy()
                                vectors_thresh_rev[:, 0] *= -1
                                vectors_thresh_rev[:, 2] *= -1

                        if self.abs_dist:

                            distances_thresh = distances[thresh]

                        else:  # 'rel', TODO: should be angular distance!

                            distances_thresh = np.sqrt(vectors_thresh[:, 3] ** 2 + vectors_thresh[:, 2] ** 2)

                        hist, uv_stats = calc_stats(

                            vectors_thresh,
                            bin_edges,
                            dist=distances_thresh,
                            angles=True,

                        )
                        if self.revtag:
                            if 'gsetwise' in self.sortlogics:
                                hist_rev, uv_stats_rev = calc_stats(

                                    vectors_thresh_rev,
                                    bin_edges,
                                    dist=distances_thresh,
                                    angles=True,

                                )

                        hist = hist.astype(np.int16)  # changed to int16

                        if group == 'ctr':

                            groupstr = 'wt'

                        else:

                            groupstr = group

                        for sl in self.sortlogics:

                            if sl == 'groupwise':

                                gkey = '_'.join([groupstr, episode])
                                self.histograms[sl][gkey].append(hist)

                            elif sl == 'gsetwise':

                                gkey = '_'.join([groupset[0], groupset[1], episode])
                                if groupstr == 'wt':
                                    gkey = '_'.join([groupstr, episode])
                                if group.endswith(self.revtag):
                                    self.histograms[sl][gkey].append(hist_rev)
                                else:
                                    self.histograms[sl][gkey].append(hist)

                            # elif sl == 'dsetwise-gset':
                            #
                            #     gkey = '_'.join([groupset[0], groupset[1], str(dataset), episode])
                            #     if groupstr == 'wt':
                            #         gkey = '_'.join([groupstr, str(dataset), episode])
                            #
                            #     if group.endswith(self.revtag):
                            #         self.histograms[sl][gkey].append(hist_rev)
                            #     else:
                            #         self.histograms[sl][gkey].append(hist)
                            #
                            # elif sl == 'dsetwise-group':
                            #
                            #     gkey = '_'.join([groupstr, str(dataset), episode])
                            #     self.histograms[sl][gkey].append(hist)

                        print('Dictionary key: ', gkey)
                        print('Unique hist vals: ', np.unique(hist).shape[0])
                        del vectors_thresh

        pickle.dump(self.histograms,
                    open(os.path.join(self.base, 'histograms_{0}{1}.p'.format(self.expset_name, tag)), 'wb'))


def plot_vfs_ind(

        histograms_abs,
        mapdict,
        sortlogic='groupwise',
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
        plotdiff=False,
        show_nmaps_single=True,
        ctr='wt',
        exan_dict={},

):
    vector_xys_abs = {}
    vector_xys_rel = {}

    be_abs = [np.linspace(b[0], b[1], res_abs[bno] + 1) for bno, b in enumerate([
        edges_pos, edges_pos, edges_angles, edges_dists])]

    be_rel = [np.linspace(b[0], b[1], res_rel[bno] + 1) for bno, b in enumerate([
        edges_pos, edges_pos, edges_dir, edges_dir])]

    groupsets = sorted(histograms_abs[sortlogic].keys())

    for gno, groupset in enumerate(groupsets):

        if 'dead' in groupset:

            continue

        hists = [histogram / np.sum(histogram) for histogram in histograms_abs[sortlogic][groupset]]
        hists_thresh = [h for hno, h in enumerate(hists) if hno not in exan_dict[groupset.split('_')[0]]]
        print(groupset, len(hists_thresh))
        hist_abs = np.mean(hists_thresh, axis=0)
        n = len(histograms_abs[sortlogic][groupset])
        print(hist_abs.shape, hist_abs.min(), hist_abs.max())
        print(np.where(np.isnan(hist_abs))[0].shape)
        if '07' in groupset:

            label = '_'.join(groupset.split('_')[:-1]) + ' continuous' + ', n=' + str(n)

        else:

            label = '_'.join(groupset.split('_')[:-1]) + ' bout-like' + ', n=' + str(n)

        fig = plt.figure(figsize=(12, 4), dpi=200)
        gs = gridspec.GridSpec(nrows=1, ncols=5, width_ratios=[1, 1, 1, .1, .1], height_ratios=[1])
        gs.update(wspace=0.8, hspace=0.4)
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

        nmaps = mapdict[sortlogic][groupset]
        nmaps_thresh = [n for nno, n in enumerate(nmaps) if nno not in exan_dict[groupset.split('_')[0]]]
        print(groupset, len(nmaps_thresh))
        nmap = np.nanmean(nmaps_thresh, axis=0)
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

        # if show_nmaps_single:
        #      if not ctr in groupset:
        #         for nno, nmap_ind in enumerate(nmaps_thresh):
        #
        #             print(nno)
        #             fig, ax = plt.subplots(figsize=(3,3))
        #             ax.imshow(nmap_ind.T, origin='lower', cmap=cmap_nmap, clim=clim_nmap,
        #                       extent=(-19.5, 20.5, -19.5, 20.5))
        #             plt.show()

    if plotdiff:

        scales = [scale_abs, scale_rel]
        for gno, groupset in enumerate(groupsets):

            if 'dead' in groupset:

                continue

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


def get_thigmo_thresh(

        df,
        thresh=35.,
        std_f=1.,
        groupwise=False

):
    '''
    Calculating which animals have a thigmotaxis index > thresh
    '''

    n_animals = df.animalIndex.unique().shape[0]

    thigmos = [np.nanmean(df[df['animalIndex'] == i]['thigmoIndex'].values) for i in range(1, n_animals + 1)]
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    _ = plt.hist(thigmos, bins=150)
    if not thresh:
        thresh = np.mean(thigmos) + np.std(thigmos) * std_f

    plt.axvline(thresh, color='red')
    plt.axvline(np.nanmean(thigmos), linestyle=':', color='red')

    plt.savefig('thighmohist.png', dpi=300, bbox_inches='tight')
    plt.close()

    exan = [i for i in range(1, n_animals + 1) if not
                np.nanmean(df[df['animalIndex'] == i]['thigmoIndex'].values) < thresh]
    if groupwise:

        exan_dict = {}
        print(df['group'].unique())
        for group in df['group'].unique():
            gdf = df[df.group == group]
            anids = gdf.animalIndex.unique()
            exan_gwise = [ino for ino, i in enumerate(anids) if not
                        np.nanmean(gdf[gdf['animalIndex'] == i]['thigmoIndex'].values) < thresh]

            exan_dict[group] = exan_gwise
        return exan, exan_dict

    else:

        return exan, None


def plot_si(

        df,
        groups=[],
        expset_name='',
        exan=[],
        exclude_groups=[]

):
    if not len(exan) == 0:
        exbool = np.invert(
            np.any(np.concatenate([[exan[i] == df.animalIndex.values] for i in range(len(exan))], axis=0).T, axis=1))
        df = df[exbool]
        for exgroup in exclude_groups:

            df = df[np.invert(df.group == exgroup)]

    if groups == []:

        groups = df['group'].unique()

    print(groups)

    df_epi_an = df.groupby(['episode', 'animalIndex', 'line', 'group', 'anSize'], sort=True).mean().reset_index()
    df_epi_an = df_epi_an[np.any([(df_epi_an['group'] == group) for group in groups], axis=0)]
    df_epi_an = df_epi_an.sort_values(by=['group', 'episode'])

    fig, ax = plt.subplots(dpi=300)
    # sns.stripplot(data=df_epi_an,x='episode',y='si',zorder=-1,hue='group',dodge=10, jitter=True, ax=ax, alpha=.5)
    sns.swarmplot(data=df_epi_an, x='episode', y='si', zorder=-1, hue='group', dodge=10, ax=ax, alpha=.5)

    sns.pointplot(data=df_epi_an, ci='sd', x='episode', hue='group', y='si', estimator=np.median, ax=ax, dodge=.5,
                  jitter=False, linestyles=['none'] * len(groups), lw=1)

    ax.set_xticklabels(['continuous', 'bout-like'])
    ax.set_ylabel('Virtual attraction')
    handles, labels = ax.get_legend_handles_labels()
    labels_legend = []
    for group in sorted(groups):
        labels_legend.append(
            '{}, n={}'.format(group, df_epi_an[df_epi_an['group'] == group]['animalIndex'].unique().shape[0]))

    l = plt.legend(handles[0:len(groups)], labels_legend, loc='upper left', borderaxespad=0.)
    plt.savefig('attraction_animalsbygroup_{}.png'.format(expset_name), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    ops = {

        'base': 'J:/_Projects/J-sq',
        'expset_name': 'jjAblationsNeuropil',
        'stim_protocol': 'boutVsSmooth_grateloom',
        'tag': '',
        'swap_stim': False,
        'shift': False,
        'yflip': True,
        'default_limit': None, # CARFUL HERE, No False!!!
        'load_expset': False,
        'cam_height': [105, 180],
        'fps': 30,

        'bout_crop': 25,
        'smooth_wl': 20,
        'smooth_alg': 'hamming',
        'unique_episodes': ['07k01f', '10k20f'],
        'groupsets': [],
        'sortlogics': ['groupwise'],
        'ctr': 'CtrE',

        'nmap_res': (30, 30),
        'dist_filter': (0, 30),
        'edges_pos': (-20, 20),
        'edges_dir': (-12, 12),
        'edges_angles': (-np.pi, np.pi),
        'edges_dists': (0, 30),
        'vmap_res': (30, 30, 30, 30),
        'revtag': 'L',
        'abs_dist': True,
        'calc_thigmo_thresh': False,
        'thigmothresh': 35.,

        # experiment set parameters
        'epiDur': 5,
        'n_episodes': 60,
        'inDish': 10,
        'arenaDiameter_mm': 100,
        'minShift': 60,
        'episodePLcode': 0,
        'recomputeAnimalSize': 0,
        'SaveNeighborhoodMaps': 1,
        'computeLeadership': 0,
        'computeBouts': 0,
        'nShiftRuns': 3,
        'filteredMaps': True

    }

    #abl_b = VectorFieldAnalysis(**ops)
    #abl_b.process_dataset()
    #abl_b.process_nmaps()
    ops['expset_name'] = 'jjAblationsFinal'
    ops['edges_pos'] = (-60, 60)
    #ops['stim_protocol'] = 'boutVsSmooth'
    abl_b = VectorFieldAnalysis(**ops)
    abl_b.load_expset = True
    abl_b.process_dataset()
    abl_b.process_nmaps()

    #ops['expset_name'] = 'jjAblationsGratingLoom'
    #ops['stim_protocol'] = 'boutVsSmooth_grateloom'
    #abl_b = VectorFieldAnalysis(**ops)
    #abl_b.process_dataset()
    #abl_b.process_nmaps()

