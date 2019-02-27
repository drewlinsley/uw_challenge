import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'seg_cluttered_nist_3_baseline_50k'
        self.output_name = 'seg_cluttered_nist_3_baseline_50k'
        self.data_name = 'baseline'
        self.img_dir = 'imgs'
        self.contour_dir = '/media/data_cifs/cluttered_nist3_seg/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [350, 350]  # 600, 600
        self.model_input_image_size = [160, 160, 1]  # [107, 160, 3]
        self.max_ims = 50000
        self.output_size = self.im_size + [2]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['resize', 'trim_extra_dims']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
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

    def list_files(self, meta, directory, cat=0):
        """List files from metadata."""
        files, labs = [], []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[cat],
                    f[2])]
            labs += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[1],
                    f[2])]
        return np.asarray(files), np.asarray(labs)

    def get_data(self):
        """Get the names of files."""
        positive_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.data_name,
                self.meta))
        ims, labs = self.list_files(positive_meta, self.data_name, cat=0)
        # labs = self.list_files(positive_meta, self.data_name, cat=1)
        rand_idx = np.random.permutation(len(ims))
        if not self.shuffle:
            raise NotImplementedError
        all_ims = ims[rand_idx]
        all_labels = labs[rand_idx]

        if self.max_ims:
            all_ims = all_ims[:self.max_ims]
            all_labels = all_labels[:self.max_ims]
        num_ims = len(all_ims)

        # Create CV folds
        cv_range = np.arange(num_ims)
        train_split = np.round(num_ims * self.cv_split)
        train_idx = cv_range < train_split
        valalidation_idx = cv_range >= train_split
        train_ims = all_ims[train_idx]
        valalidation_ims = all_ims[valalidation_idx]
        train_labels = all_labels[train_idx]
        validation_labels = all_labels[valalidation_idx]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = valalidation_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = validation_labels
        return cv_files, cv_labels

