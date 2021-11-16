import models.experiment as xp
import functions.matrixUtilities_joh as mu
import models.experiment_set as es
import glob
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib
from scipy import stats
import statsmodels.stats.api as sms


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    es=(u1 - u2) / s
    return es, n1, n2, s1, s2, s, u1, u2

def groupCohen(x, cat='wt'):
    dat=x.copy()
    dat.set_index(['animalIndex', cat])

    catLevels = x[cat].unique()
    catLevels.sort()

    if len(catLevels) > 1:
        d1 = dat[dat[cat] == catLevels[1]].dropna().iloc[:,2:]
        d2 = dat[dat[cat] == catLevels[0]].dropna().iloc[:,2:]

        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)
        # calculate the effect size
        return (u2 - u1) / s
    else:
        ret=pd.Series(np.zeros(dat.shape[1]-2) * np.nan, index=dat.columns.values[2:])
        return ret


def groupPower(x, cat='wt',field='si',alpha=0.05,power=0.8):
    dat = x.groupby(['animalIndex', cat])[field].mean().unstack().reset_index()
    names = ['mnA', 'mnB', 'mnDiff', 'p', 'es', 'P', 'nA', 'nB', 'esReal', 'sPooled', 'sensitivity','s1','s2']
    catLevels = x[cat].unique()
    catLevels.sort()
    # if dat.shape[1] > 2:
    if len(catLevels) == 2:
        a = dat[catLevels[0]].dropna()
        b = dat[catLevels[1]].dropna()
        p = stats.ttest_ind(a, b, equal_var=False)[1]

        effectSize = 1
        effectSizeReal, n1, n2, s1, s2, s, u1, u2 = cohend(a, b)
        sd1,sd2=np.std(a),np.std(b)

        r = len(a) / len(b)
        mnDiff = a.mean() - b.mean()

        P = sms.TTestIndPower().solve_power(effectSize, power=None, alpha=alpha, ratio=r, nobs1=n2)
        sens = sms.TTestIndPower().solve_power(effect_size=None, power=power, alpha=alpha, ratio=r, nobs1=n2)

        return pd.Series([a.mean(),
                          b.mean(),
                          mnDiff,
                          p,
                          effectSize,
                          P,
                          len(a),
                          len(b),
                          effectSizeReal,
                          s,
                          sens,
                          sd1,
                          sd2], index=names)
    else:
        return pd.Series(np.zeros(len(names)) * np.nan, index=names)

def readExperiment(csvFile, keepData=False, MissingOnly=True):
    tmp = es.experiment_set(csvFile=csvFile, MissingOnly=MissingOnly)
    if keepData:
        return tmp
    else:
        return 1

def savedCsvToDf(txtPaths, baseDir='d:\\data\\', seachString='*siSummary*.csv',noOfAnimals=15):
    csvPath = []
    for f in [mu.splitall(x)[-1][:-4] for x in txtPaths]:
        csvPath.append(glob.glob(baseDir+f+seachString)[0])

    df=pd.DataFrame()
    i=0
    for fn in csvPath:
        print(fn)
        tmp = pd.read_csv(fn, index_col=0, sep=',')
        tmp.animalSet = i
        tmp.animalIndex = tmp.animalIndex+(i*noOfAnimals)
        df = pd.concat([df, tmp])
        i += 1
    return df


def computeExpTimeOfDay(df):
    d = df.time
    r = datetime(int(df.time.iloc[0][:4]), 1, 1)
    t2 = [pd.to_datetime(x).replace(day=1, month=1, year=r.year)for x in df.time]
    t3 = [(x-r)/pd.Timedelta('1 hour') for x in t2]
    df['t2'] = t2
    df['t3'] = t3
    return df


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def speedFft(data):
    tmp=data-np.nanmean(data)
    tmp=tmp[~np.isnan(tmp)]
    freqs = np.fft.fftfreq(tmp.size,1)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(tmp))**2
    return freqs[idx],ps[idx]

def powerToSample(df):
    result=[]
    for i,r in df.iterrows():
        text="%s,%s,%s,%s,%s,%s,%s"%(
            str(i),
            '{:3.5f}'.format(r.mnA),
            '{:3.5f}'.format(r.s1),
            '{:3.5f}'.format(r.nA),
            '{:3.5f}'.format(r.mnB),
            '{:3.5f}'.format(r.s2),
            '{:3.5f}'.format(r.nB),
        )
        result.append(text)
    return result