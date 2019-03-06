import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob
import pandas as pd


class data_processing(object):
    def __init__(self):
        self.name = 'uw_challenge'
        self.output_name = 'uw_challenge'
        self.img_dir = 'imgs'
        self.image_data = '/media/data_cifs/Sheinberg_lab/data/bold5000.npy'
        self.neural_data = '/media/data_cifs/Sheinberg_lab/data/bold5000_sqrt_of_numspk_responsiveCells_2019-01-23.csv'
        self.config = Config()
        self.im_size = [375, 375, 3]  # 600, 600
        self.model_input_image_size = [224, 224, 3]  # [107, 160, 3]
        self.output_size = [235]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.z_score_neurons = True
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.test_data_split = 50
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['resize']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature,
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='float32'),
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def get_data(self, split_start=None, split_size=None):
        """Get the names of files."""
        if split_start is None:
            split_size, split_start = 50, 0  # Take first 50 images for validation
        image_data = np.load(self.image_data)
        ims = []
        for im in image_data:
            im_s = im.shape
            diff = (im_s[0] - 320) // 2
            ims += [im[diff: diff + 320, diff: diff + 320, :]]
        image_data = np.stack(ims, axis=0)

        # image_data = image_data[..., [2, 1, 0]]
        neural_data = pd.read_csv(self.neural_data)
        train_images = image_data[self.test_data_split:]
        test_images = image_data[:self.test_data_split]
        train_labels = neural_data.as_matrix()[self.test_data_split:, 1:]  # First column is index
        test_labels = neural_data.as_matrix()[:self.test_data_split, 1:]

        # Create validation set
        val_idx = np.in1d(np.arange(len(train_images)), np.arange(split_start, split_start + split_size))
        val_images = train_images[val_idx]
        val_labels = train_labels[val_idx]
        # assert not np.sum(np.isnan(val_labels))
        train_images = train_images[~val_idx]
        train_labels = train_labels[~val_idx]
        if self.z_score_neurons:
            train_mean = np.nanmean(train_labels, axis=0, keepdims=True)
            train_std = np.nanstd(train_labels, axis=0, keepdims=True)
            np.savez(os.path.join('moments', self.output_name), mean=train_mean, std=train_std)
            # train_labels = (train_labels - train_mean) / train_std
            # val_labels = (val_labels - train_mean) / train_std
        train_labels[np.isnan(train_labels)] = 0.  #  -99.
        val_labels[np.isnan(val_labels)] = 0.  # -99.

        # Build CV dict
        cv_files, cv_labels, cv_masks = {}, {}, {}
        cv_files[self.folds['train']] = train_images
        cv_files[self.folds['val']] = val_images
        cv_files[self.folds['test']] = test_images
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        cv_labels[self.folds['test']] = test_labels
        cv_masks[self.folds['train']] = train_images
        cv_masks[self.folds['val']] = val_images
        cv_masks[self.folds['test']] = test_images
        return cv_files, cv_labels, cv_masks

