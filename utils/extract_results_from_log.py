import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils import py_utils


OUTDIR = 'npy_log'
py_utils.make_dir(OUTDIR)


def main(
        log,
        save_data=False,
        plot_data=False,
        data_check='Validation',
        val_key=r'(?<=(Validation\saccuracy\s=\s))\d\.\d+',
        train_key=r'(?<=(Training\saccuracy\s=\s))\d\.\d+'):
    """Parse a log file for the key string extending to the next whitespace."""
    train_data, val_data = [], []
    with open(log, 'rb') as f:
        for line in f:
            if data_check in line:
                train_data += [float(re.search(train_key, line).group())]
                val_data += [float(re.search(val_key, line).group())]
    val_data = np.array(val_data)

    if plot_data:
        f = plt.figure()
        plt.plot(val_data, label='Validation')
        plt.plot(train_data, label='Training')
        plt.xlabel('Iterations of training (x2000)')
        plt.ylabel('Accuracy')
        plt.title('Max validation: %s' % val_data.max())
        plt.legend()
        plt.show()
        plt.close(f)

    if save_data:
        np.savez(
            os.path.join(OUTDIR, log.split(os.path.sep)[-1]),
            val=val_data,
            train=train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log',
        dest='log',
        type=str,
        default=None,
        help='Path to log file.')
    parser.add_argument(
        '--save_data',
        dest='save_data',
        action='store_true',
        help='Save data to npy.')
    parser.add_argument(
        '--plot_data',
        dest='plot_data',
        action='store_true',
        help='Show plot of data.')
    args = parser.parse_args()
    main(**vars(args))
