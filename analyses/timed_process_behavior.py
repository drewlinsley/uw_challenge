"""Script for processing + plotting CABC and pathfinder experiments."""
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn import metrics
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
from math import exp, sqrt


MAIN_PATH = '/media/data_cifs/Kalpit'
EXT = '.csv'


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def dPrime(y_true, y_pred):
    """Calculate confusion matrix and d prime."""
    df_confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    crs = float(df_confusion[0][0])
    misses = float(df_confusion[0][1])
    fas = float(df_confusion[1][0])
    hits = float(df_confusion[1][1])

    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5 / (hits + misses)
    halfFa = 0.5 / (fas + crs)
 
    # Calculate hitrate and avoid d' infinity
    hitRate = hits / (hits + misses)
    if hitRate == 1: hitRate = 1 - halfHit
    if hitRate == 0: hitRate = halfHit
 
    # Calculate false alarm rate and avoid d' infinity
    faRate = fas / (fas + crs)
    if faRate == 1: faRate = 1 - halfFa
    if faRate == 0: faRate = halfFa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = norm.ppf(hitRate) - norm.ppf(faRate)
    if out['d'] < -1:
        raise RuntimeError('Something went wrong with d-prime calculation.')
    out['beta'] = np.exp((norm.ppf(faRate) ** 2 - norm.ppf(hitRate) ** 2) / 2)
    out['c'] = -(norm.ppf(hitRate) + norm.ppf(faRate)) / 2
    out['Ad'] = norm.cdf(out['d'] / sqrt(2))
    out['cm'] = [hits, misses, fas, crs]
    return out


def process_data(
        path,
        conditions,
        file_ext,
        exp,
        max_time=5000,
        filter_timed=None,
        mi_metric=metrics.normalized_mutual_info_score):
    """Process experiment data and return a pandas dataframe."""
    files = []
    if conditions is not None:
        for cond in conditions:
            files += glob(os.path.join(path, cond, '*%s' % file_ext))
    else:
        files = glob(os.path.join(path, '*%s' % file_ext))

    dfs = []
    for f in tqdm(files, desc='Processing files', total=len(files)):
        # 1. Read data into pandas and filter for timings if requested
        df = pd.read_csv(f)
        if filter_timed is not None and df.iloc[0].rtime != filter_timed:
            continue

        # 2. Filter for response data
        if filter_timed is None:
            df = df[np.logical_and(
                df['part'].str.contains(r'split\d'),
                pd.notna(df['part']))]
        else:
            df = df[np.logical_and(
                np.in1d(df['part'], ['base', 'lev1', 'lev2']),
                pd.notna(df['key_press']))]
            df['dataset'] = df['part']
            df['dataset'][df['dataset'] == 'base'] = 'baseline'
            df['dataset'][df['dataset'] == 'lev1'] = 'ix1'
            df['dataset'][df['dataset'] == 'lev2'] = 'ix2'

        # 3. Add new column with the button masher score
        df['masher'] = np.sum(np.diff(df['key_press']) == 0.)
        df['acc'] = np.mean(df['correct'])

        # 4. Add new column with MI
        df['correct_response'][df['correct_response'] == 'yes'] = 1
        df['correct_response'][df['correct_response'] == 'no'] = 0
        df['correct_response'] = df['correct_response'].astype(np.int32)
        try:
            inverted = df.key_press.iloc[np.where(
                np.logical_and(df.correct, df.correct_response == 1))[0][0]] == 189
        except:
            inverted = df.key_press.iloc[np.where(
                np.logical_and(df.correct, df.correct_response == 0))[0][0]] == 189
        df['inverted'] = inverted
        if inverted:
            df['response'] = (df.key_press == 189).astype(np.int32)
        else:
            df['response'] = (df.key_press == 187).astype(np.int32)
        df['mi'] = mi_metric(
            labels_true=df['correct_response'],
            labels_pred=df['key_press'].astype(np.int32))

        # 5. Add dataset as a column
        dataset = []
        for idx in range(len(df)):
            tmp = df['stimulus'].iloc[idx].split('/')[-1]
            if exp is 'CNIST':
                tmp = tmp.split('-')[0]
                _tmp = tmp.split('_')
                if len(_tmp) > 1:
                    tmp = _tmp[0]
            elif exp is 'PFINDER':
                tmp = tmp.split('_imgs')[0]
            elif 'timed' in exp:
                tmp = df['stimulus'].iloc[idx].split('/')[-3]
                tmp = tmp.split('-')[0]
                _tmp = tmp.split('_')
                if len(_tmp) > 1:
                    tmp = _tmp[0]
            else:
                raise NotImplementedError
            dataset += [tmp]
        df['dataset'] = dataset

        # 6. Throw out ps that didn't respond in time
        df = df[df['rt'] < max_time]

        # 7. Cast correct to float
        df['correct'] = df['correct'].astype(np.float32)

        # 8. Concatenate the df
        dfs += [df]
    assert len(dfs), 'No data found.'
    dfs = pd.concat(dfs)
    return dfs


def filter_data(data, filters):
    """Filter data with filters."""
    filt = np.full(len(data), True)
    for idx, f in enumerate(filters):
        k, v = f.items()[0]
        print('Appling filter %s to %s' % (idx, k))
        filt = np.logical_and(filt, v(data[k]))  # Accumulate filter
    filtered = data[filt]
    return filtered


def plot_results(
        out_path,
        data,
        dv,
        iv='dataset',
        hue=None,
        order=None,
        title=None,
        estimator=np.median,
        ylims=None):
    """Plot the data with the given key as the DV using seaborn."""
    f = plt.figure()
    ax = sns.pointplot(data=data, x=iv, y=dv, order=order, hue=hue)
    if title is not None:
        plt.title(title)
    if ylims is not None:
        plt.ylim(ylims)
    for item in ax.get_xticklabels():
        item.set_rotation(20)
    if hue is None:
        unit = data[dv].median() / 100.
        [
            ax.text(p[0], p[1] + unit, p[1], color='g')
            for p in zip(
                ax.get_xticks(),
                data.groupby(iv).agg(np.mean).reset_index()[dv])]
    plt.savefig(out_path)
    plt.show()
    plt.close(f)


def plot_diagnostics(data, key, title, sd_fun=np.std, mu_fun=np.mean, sds=2, rectify=True):
    """Plot diagnostic histograms."""
    ks = key.split(',')
    if len(ks) > 1:
        key = ' and '.join(key)
        f = plt.figure()
        plt.title(title)
        plt.scatter(data[ks[0]], data[ks[1]])
        plt.xlabel(ks[0])
        plt.ylabel(ks[1])
        rt_mu = mu_fun(data[ks[0]])
        rt_sd = sd_fun(data[ks[0]])
        rt_low = np.maximum(rt_mu - (sds * rt_sd), 0)
        rt_high = rt_mu + (sds * rt_sd)
    else:
        if rectify:
            d = np.maximum(0, data[key])
        else:
            d = data[key] 
        f = plt.figure()
        plt.title(title)
        plt.hist(d, len(np.unique(d)) // 2)
        plt.xlabel(key)
        plt.ylabel('Frequency')
        rt_mu = mu_fun(data[key])
        rt_sd = sd_fun(data[key])
        rt_low = np.maximum(rt_mu - (sds * rt_sd), 0)
        rt_high = rt_mu + (sds * rt_sd)
        plt.axvline(rt_low, color='red')
        plt.axvline(rt_high, color='red')
    print('Found %ss +/-%sSDs: %s, %s' % (key, sds, rt_high, rt_low))
    plt.show()
    plt.close(f)


def main(
        MAIN_PATH,
        EXP,
        DIAGNOSTICS,
        FILTERS,
        OUT_PATH,
        PLOT_DIAGNOSTICS=False,
        EXT='.csv',
        MU_FUN=np.median,
        SD_FUN=mad,
        CONDITIONS=['data_left', 'data_right'],
        TIMED=None,
        DEBUG=False,
        SDS=3):
    """Wrapper to run analyses."""
    df = process_data(
        path=os.path.join(MAIN_PATH, EXP.keys()[0]),
        conditions=CONDITIONS,
        file_ext=EXT,
        filter_timed=TIMED,
        exp=EXP.keys()[0])

    # Create diagnostic plots
    if PLOT_DIAGNOSTICS:
        for d in DIAGNOSTICS:
            k, v = d.items()[0]
            plot_diagnostics(
                data=df,
                key=k,
                title=v,
                mu_fun=MU_FUN,
                sd_fun=SD_FUN,
                sds=SDS)

    # Apply filters
    fdf = filter_data(data=df, filters=FILTERS)

    if not DEBUG:
        # Plot RT and accuracy
        plot_results(
            out_path=os.path.join(OUT_PATH, '%s_rt' % EXP),
            data=fdf,
            dv='rt',
            title=EXP.keys()[0],
            order=EXP.values()[0])
        plot_results(
            out_path=os.path.join(OUT_PATH, '%s_acc' % EXP),
            data=fdf,
            dv='correct',
            title=EXP.keys()[0],
            order=EXP.values()[0],
            ylims=[0, 1])

        # Plot RT x accuracy
        # fdf['rt_idx'] = pd.cut(fdf.rt, 10).index
        rt_acc_df = fdf.groupby([pd.cut(fdf.rt, 10), 'dataset']).agg(np.mean)
        plot_results(
            out_path=os.path.join(OUT_PATH, '%s_acc_X_rt' % EXP),
            data=rt_acc_df.correct.reset_index(),
            dv='correct',
            iv='rt',
            hue='dataset',
            title=EXP.keys()[0],
            ylims=[0, 1])

    # Get d primes
    subs = np.unique(fdf.SubjectID)
    dfs = []
    for sub in tqdm(subs, total=len(subs), desc='Calculating d-prime'):
        it_df = fdf[fdf.SubjectID == sub]
        it_df['d'] = np.nan
        dss = np.unique(it_df.dataset)
        for ds in dss:
            mask = it_df.dataset == ds
            it_data = it_df[mask]
            if len(it_data.correct_response) > 10:
                it_data['d'] = dPrime(it_data.correct_response, it_data.response)['d']
                it_df[mask] = it_data
	dfs += [it_df]
    fdf = pd.concat(dfs)
    sns.stripplot(
        data=fdf.groupby(['SubjectID', 'dataset']).agg(np.median).reset_index(),
        x='dataset',
        y='d')
    plt.savefig(os.path.join(OUT_PATH, '%s_dp' % EXP))

    # Plot RT x dprime
    dp = fdf.groupby([pd.qcut(fdf.rt, 10), 'dataset'], as_index=False).agg(np.median)
    dp.rt = pd.Series(dp.rt).apply(lambda x: int(50. * round(float(x) / 50.)))
    plot_results(
        out_path=os.path.join(OUT_PATH, '%s_dp_X_rt' % EXP),
        data=dp,
        dv='d',
        iv='rt',
        hue='dataset',
        title=EXP.keys()[0],
        ylims=[0, 1])

    # Save processed data to CSVs
    df.to_csv(os.path.join(OUT_PATH, '%s_raw_data.csv' % EXP.keys()[0]))
    fdf.to_csv(os.path.join(OUT_PATH, '%s_filtered_data.csv' % EXP.keys()[0]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--diagnostics',
        dest='PLOT_DIAGNOSTICS',
        action='store_true',
        help='Create diagnostic plots for trimming data.')
    args = parser.parse_args()

    # GLOBALS -- convert into args in the future
    MU_FUN = np.median  # np.mean  or np.median
    SD_FUN = mad  # np.std or mad
    SDS = 2
    DIAGNOSTICS = [
        {'rt': "Histogram of participants' responses."},
        {'masher': 'How often did ps repeat the same buttonpress?'},
        {'mi,acc': 'Mutual information between response/label per participant.'}
    ]
    OUT_PATH = 'analysis_data'
    TIMED = None
    exp = 'CNIST_timed_1600'  # 'PFINDER' or 'CNIST'
    if exp == 'PFINDER':
        CONDITIONS = ['data_left', 'data_right']
        EXP = {
            'PFINDER': [
                'curv_baseline',
                'curv_contour_length_9',
                'curv_contour_length_14'
            ]
        }
        FILTERS = [  # PFINDER
            # {'masher': lambda x: np.logical_and(
            #     np.greater_equal(x, 63),
            #     np.less_equal(x, 91))},
            # {'rt': lambda x: np.logical_and(  # PF
            #     np.greater_equal(x, 378),
            #     np.less_equal(x, 1482))},
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 64),
                np.less_equal(x, 84))},
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 380),  # 219
                np.less_equal(x, 1478))},  # 1859
        ]
    elif exp == 'CNIST':
        CONDITIONS = ['data_left', 'data_right']
        EXP = {
            'CNIST': [
                'baseline',
                'ix1',
                'ix2'
            ]
        }
        FILTERS = [
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 63),
                np.less_equal(x, 91))},
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 227),  # 219
                np.less_equal(x, 1831))},  # 1859
            {'mi': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 0.30),  # 219
                np.less_equal(x, 1))},  # 1859
        ]
    elif exp == 'CNIST_timed_800':
        CONDITIONS = None
        EXP = {
            'CNIST_timed': [
                'baseline',
                'ix1',
                'ix2'
            ]
        }
        FILTERS = [
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 1425.0),  # 219
                np.less_equal(x, 100000))},  # 1859
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 27),
                np.less_equal(x, 57))},
            # {'mi': lambda x: np.logical_and(  # CABC
            #     np.greater_equal(x, 0.25),  # 219
            #     np.less_equal(x, 1))},  # 1859
        ]
        TIMED = 800
    elif exp == 'CNIST_timed_1600':
        CONDITIONS = None
        EXP = {
            'CNIST_timed': [
                'baseline',
                'ix1',
                'ix2'
            ]
        }
        FILTERS = [
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 1425.0),  # 219
                np.less_equal(x, 100000))},  # 1859
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 27),
                np.less_equal(x, 57))},
            {'acc': lambda x: np.greater_equal(x, 0.5)},
        ]
        TIMED = 1600
    else:
        raise NotImplementedError
    main(
        MAIN_PATH=MAIN_PATH,
        EXP=EXP,
        DIAGNOSTICS=DIAGNOSTICS,
        FILTERS=FILTERS,
        EXT=EXT,
        MU_FUN=MU_FUN,
        SD_FUN=SD_FUN,
        CONDITIONS=CONDITIONS,
        TIMED=TIMED,
        SDS=SDS,
        OUT_PATH=OUT_PATH,
        **vars(args))

