import numpy as np
import csv
from db import db
from argparse import ArgumentParser
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import Config
import os
from glob import glob
from scipy import ndimage
from utils import py_utils


def main(
        experiment_name,
        im_ext='.pdf',
        transform_loss=None,  # 'log',
        colors='Paired',
        flip_axis=False,
        port_fwd=False,
        num_steps=np.inf,
        exclude=None,
        list_experiments=False,
        out_dir='analysis_data'):
    """Plot results of provided experiment name."""
    config = Config()
    if list_experiments:
        db.list_experiments()
        return

    if port_fwd:
        config.db_ssh_forward = True
    py_utils.make_dir(out_dir)

    # Get experiment data
    if ',' in experiment_name:
        exps = experiment_name.split(',')
        perf = []
        for exp in exps:
            perf += db.get_performance(experiment_name=exp)
        experiment_name = exps[0]
    else:
        perf = db.get_performance(experiment_name=experiment_name)
    if len(perf) == 0:
        raise RuntimeError('Could not find any results.')

    structure_names = [x['model'].split('/')[-1] for x in perf]
    datasets = [x['val_dataset'] for x in perf]
    steps = [float(x['step']) for x in perf]
    training_loss = [float(x['train_loss']) for x in perf]
    validation_loss = [float(x['val_loss']) for x in perf]
    training_score = [float(x['train_score']) for x in perf]
    validation_score = [float(x['val_score']) for x in perf]
    summary_dirs = [x['summary_path'] for x in perf]
    ckpts = [x['ckpt_path'] for x in perf]
    params = [x['num_params'] for x in perf]

    # Pass data into a pandas DF
    df = pd.DataFrame(
        np.vstack(
            (
                structure_names,
                datasets,
                steps,
                params,
                training_loss,
                training_score,
                validation_loss,
                validation_score,
                summary_dirs,
                ckpts,
            )
        ).transpose(),
        columns=[
            'model names',
            'datasets',
            'training iteration',
            'params',
            'training loss',
            'training accuracy',
            'validation loss',
            'validation accuracy',
            'summary_dirs',
            'checkpoints'])
    df['training loss'] = pd.to_numeric(
        df['training loss'],
        errors='coerce')
    df['validation accuracy'] = pd.to_numeric(
        df['validation accuracy'],
        errors='coerce')
    df['training accuracy'] = pd.to_numeric(
        df['training accuracy'],
        errors='coerce')
    df['training iteration'] = pd.to_numeric(
        df['training iteration'],
        errors='coerce')
    df['params'] = pd.to_numeric(
        df['params'],
        errors='coerce')

    # Plot TTA
    dfs = []
    print(len(df))
    uni_structure_names = np.unique(structure_names)
    max_num_steps = num_steps  # (20000 / 32) * num_epochs
    # min_num_steps = 1
    for m in tqdm(uni_structure_names, total=len(uni_structure_names)):
        it_df = df[df['model names'] == m]
        it_df = it_df[it_df['training iteration'] < max_num_steps]
        # sorted_df = it_df.sort_values('training loss')
        # max_vals = sorted_df.groupby(['datasets']).first()
        sorted_df = []
        different_models = np.unique(it_df['summary_dirs'])
        num_models = len(different_models)
        for model in different_models:
            # Grab each model then sort by training iteration
            sel_data = it_df[it_df['summary_dirs'] == model]
            sel_data = sel_data.sort_values('training iteration')

            # Smooth the sorted validation scores for tta
            sel_data['tta'] = ndimage.gaussian_filter1d(
                sel_data['validation accuracy'], 3)
            sel_data['num_runs'] = num_models
            sorted_df += [sel_data]
        sorted_df = pd.concat(sorted_df)
        dfs += [sorted_df]

    # Get max scores and TTAs
    dfs = pd.concat(dfs)
    scores = dfs.groupby(['datasets', 'model names'], as_index=False).max()  # skipna=True)
    ttas = dfs.groupby(['datasets', 'model names'], as_index=False).mean()  # skipna=True)

    # Combine into a single DF
    scores['tta'] = ttas['validation accuracy']

    # Save datasets to csv
    filename = 'raw_data_%s.csv' % experiment_name
    dfs.to_csv(os.path.join(out_dir, filename))
    filename = 'scores_%s.csv' % experiment_name
    scores.to_csv(os.path.join(out_dir, filename))

    # Save an easy-to-parse csv for test datasets and fix for automated processing
    trim_ckpts, trim_models = [], []
    for idx in range(len(scores)):
        ckpt = scores.iloc[idx]['checkpoints']
        ckpt = '%s-%s' % (ckpt, ckpt.split('.')[0].split('_')[-1])
        model = scores.iloc[idx]['model names']
        trim_ckpts += [ckpt]
        trim_models += [model]
    # trimmed_ckpts = pd.DataFrame(trim_ckpts, columns=['checkpoints'])
    # trimmed_models = pd.DataFrame(trim_models, columns=['model'])
    trimmed_ckpts = pd.DataFrame(trim_ckpts)
    trimmed_models = pd.DataFrame(trim_models)
    trimmed_ckpts.to_csv(
        os.path.join(out_dir, 'checkpoints_%s.csv' % experiment_name))
    trimmed_models.to_csv(
        os.path.join(out_dir, 'models_%s.csv' % experiment_name))

    # Add indicator variable to group different model types during plotting
    scores['model_idx'] = 0
    model_groups = ['fgru', 'resnet', 'unet', 'hgru']
    for idx, m in enumerate(model_groups):
        scores['model_idx'][scores['model names'].str.contains(
            m, regex=False)] = idx
    keep_groups = np.where(~np.in1d(model_groups, 'hgru'))[0]
    scores = scores[scores['model_idx'].isin(keep_groups)]

    # Print scores to console
    print scores

    # Create max accuracy plots and aggregated dataset
    num_groups = len(keep_groups)
    # agg_df = []
    f = plt.figure()
    sns.set(context='paper', font='Arial', font_scale=.5)
    sns.set_style("white")
    sns.despine()
    count = 1
    for idx in keep_groups:
        plt.subplot(1, num_groups, count)
        sel_df = scores[scores['model_idx'] == idx]
        # sel_df = sel_df.groupby(
        #     ['datasets', 'model names'], as_index=False).aggregate('max')
        # agg_df += [sel_df]
        sns.pointplot(
            data=sel_df,
            x='datasets',
            y='validation accuracy',
            hue='model names')
        plt.ylim([0.4, 1.1])
        count += 1
    plt.savefig(os.path.join(out_dir, 'max_%s.png' % experiment_name), dpi=300)
    filename = 'agg_data_%s.csv' % experiment_name
    # agg_df = pd.concat(agg_df)
    # agg_df.to_csv(os.path.join(out_dir, filename))
    plt.close(f)

    # Create tta plots
    f = plt.figure()
    sns.set(context='paper', font='Arial', font_scale=.5)
    sns.set_style("white")
    sns.despine()
    count = 1
    for idx in keep_groups:
        plt.subplot(1, num_groups, count)
        sel_df = scores[scores['model_idx'] == idx]
        # sel_df = sel_df.groupby(
        #     ['datasets', 'model names'], as_index=False).aggregate('mean')
        sns.pointplot(
            data=sel_df,
            x='datasets',
            y='tta',
            hue='model names')
        plt.ylim([0.4, 1.1])
        count += 1
    plt.savefig(os.path.join(out_dir, 'tta_%s.png' % experiment_name), dpi=300)
    plt.close(f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--num_steps',
        dest='num_steps',
        type=int,
        default=np.inf,
        help='Number of training steps to limit analysis to.')
    parser.add_argument(
        '--port_fwd',
        dest='port_fwd',
        action='store_true',
        help='Force a port fwd connection to the DB.')
    parser.add_argument(
        '--list_experiments',
        dest='list_experiments',
        action='store_true',
        help='List experiments.')
    args = parser.parse_args()
    main(**vars(args))
