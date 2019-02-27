import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'ix_1'
        self.data_name = 'ix_1'
        self.output_name = 'synth_connectomics_ix_1'
        self.contour_dir = '/media/data_cifs/synth_connectomics'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [250, 250]
        self.model_input_image_size = [224, 224, 1]
        self.max_train_ims = 250000
        self.max_val_ims = 100000
        self.output_size = self.model_input_image_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.negative = 'curv_contour_length_14_neg'
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.im_size
            }
        }

    def list_files(self, meta, directory):
        """List files from metadata."""
        files, gts = [], []
        for f in meta:
            fl = f.tolist()
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    fl[0][0],
                    fl[0][1])]
            gts += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    fl[1][0],
                    fl[1][1])]
        return np.asarray(files), np.asarray(gts)

    def get_data(self):
        """Get the names of files."""
        file_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.data_name,
                self.meta))
        ims, labels = self.list_files(file_meta, self.data_name)
        num_ims = len(ims)
        if self.shuffle:
            shuff = np.random.permutation(num_ims)
            ims = ims[shuff]
            labels = labels[shuff]

        # Create CV folds
        cv_range = np.arange(num_ims)
        train_split = np.round(num_ims * self.cv_split)
        train_idx = cv_range < train_split
        validation_idx = cv_range >= train_split
        train_ims = ims[train_idx]
        validation_ims = ims[validation_idx]
        train_labels = labels[train_idx]
        validation_labels = labels[validation_idx]
        if self.max_train_ims:
            train_ims = train_ims[:self.max_train_ims]
            train_labels = train_labels[:self.max_train_ims]
        if self.max_val_ims:
            validation_ims = validation_ims[:self.max_train_ims]
            validation_labels = validation_labels[:self.max_train_ims]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = validation_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = validation_labels
        return cv_files, cv_labels

