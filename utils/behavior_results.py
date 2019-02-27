"""Script for processing + plotting CABC and pathfinder experiments."""
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns


MAIN_PATH = '/media/data_cifs/Kalpit'
EXT = '.csv'


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def process_data(path, conditions, file_ext, exp, max_time=5000):
    """Process experiment data and return a pandas dataframe."""
    files = []
    for cond in conditions:
        files += glob(os.path.join(path, cond, '*%s' % file_ext))

    dfs = []
    for file in tqdm(files, desc='Processing files', total=len(files)):
        # 1. Read data into pandas
        df = pd.read_csv(file)

        # 2. Filter for response data
        df = df[df['part'].str.contains(r'split\d') == True]

        # 3. Add new column with the button masher score
        df['masher'] = np.sum(
            np.diff(df['key_press'].as_matrix().astype(np.int)) == 0.)

        # 4. Add dataset as a column
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
            else:
                raise NotImplementedError
            dataset += [tmp]
        df['dataset'] = dataset

        # 5. Throw out ps that didn't respond in time
        df = df[df['rt'] < max_time]

        # 6. Cast correct to float
        df['correct'] = df['correct'].astype(np.float32)

        # 7. Concatenate the df
        dfs += [df]
    dfs = pd.concat(dfs)
    return dfs


def filter_data(data, filters):
    """Filter data with filters."""
    filt = np.full(len(data), 0)
    for idx, f in enumerate(filters):
        k, v = f.items()[0]
        print('Appling filter %s to %s' % (idx, k))
        filt += v(data[k])  # Accumulate filter
    filtered = data[filt > 0]
    return filtered


def plot_results(
        out_path,
        data,
        dv,
        iv='dataset',
        order=None,
        title=None,
        ylims=None):
    """Plot the data with the given key as the DV using seaborn."""
    f = plt.figure()
    sns.pointplot(data=data, x=iv, y=dv, order=order)
    if title is not None:
        plt.title(title)
    if ylims is not None:
        plt.ylim(ylims)
    plt.savefig(out_path)
    plt.show()
    plt.close(f)


def plot_diagnostics(data, key, title, sd_fun=np.std, mu_fun=np.mean, sds=2):
    """Plot diagnostic histograms."""
    f = plt.figure()
    plt.title(title)
    plt.hist(np.unique(data[key]), len(np.unique(data[key])) // 2)
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
        EXT='.csv',
        MU_FUN=np.median,
        SD_FUN=mad,
        CONDITIONS=['data_left', 'data_right'],
        SDS=3):
    """Wrapper to run analyses."""
    df = process_data(
        path=os.path.join(MAIN_PATH, EXP.keys()[0]),
        conditions=CONDITIONS,
        file_ext=EXT,
        exp=EXP.keys()[0])
    import ipdb;ipdb.set_trace()

    # Create diagnostic plots
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

    # Plot RT and accuracy
    plot_results(
        out_path='%s_rt' % EXP,
        data=fdf,
        dv='rt',
        title=EXP.keys()[0],
        order=EXP.values()[0])
    plot_results(
        out_path='%s_acc' % EXP,
        data=fdf,
        dv='correct',
        title=EXP.keys()[0],
        order=EXP.values()[0],
        ylims=[0, 1])

    # Save processed data to CSVs
    df.to_csv('%s_raw_data.csv' % EXP.keys()[0])
    fdf.to_csv('%s_filtered_data.csv' % EXP.keys()[0])


if __name__ == '__main__':
    MU_FUN = np.median  # np.mean  or np.median
    SD_FUN = mad  # np.std or mad
    SDS = 3
    DIAGNOSTICS = [
        {'rt': "Histogram of participants' responses."},
        {'masher': 'How often did ps repeat the same buttonpress?'}
    ]
    CONDITIONS = ['data_left', 'data_right']
    exp = 'CNIST'  # 'PFINDER' or 'CNIST'
    if exp == 'PFINDER':
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
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 64),
                np.less_equal(x, 84))},
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 380),  # 219
                np.less_equal(x, 1478))},  # 1859
            # {'rt': lambda x: np.logical_and(  # PF
            #     np.greater_equal(x, 378),
            #     np.less_equal(x, 1482))},
        ]
    elif exp == 'CNIST':
        EXP = {
            'CNIST': [
                'baseline',
                'ix1',
                'ix2'
            ]
        }
        FILTERS = [  # PFINDER
            # {'masher': lambda x: np.logical_and(
            #     np.greater_equal(x, 63),
            #     np.less_equal(x, 91))},
            {'masher': lambda x: np.logical_and(
                np.greater_equal(x, 64),
                np.less_equal(x, 84))},
            {'rt': lambda x: np.logical_and(  # CABC
                np.greater_equal(x, 500),  # 219
                np.less_equal(x, 2000))},  # 1859
            # {'rt': lambda x: np.logical_and(  # PF
            #     np.greater_equal(x, 378),
            #     np.less_equal(x, 1482))},
        ]
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
        SDS=SDS)
