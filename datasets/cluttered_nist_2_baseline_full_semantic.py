import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'cluttered_nist_2_baseline_full_semantic'
        self.output_name = 'cluttered_nist_2_baseline_full_semantic'
        self.data_name = 'baseline'
        self.img_dir = 'imgs'
        self.contour_dir = '/media/data_cifs/cluttered_nist2_plus/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [350, 350]  # 600, 600
        self.model_input_image_size = [160, 160, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 20000000
        self.output_size = {'output': 1, 'aux': self.nhot_size[0]}
        self.label_size = [1]
        self.default_loss_function = 'cce'
        self.aux_loss = {'nhot': ['bce', 1.]}  # Loss type and scale
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['resize']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'nhot': tf_fun.float_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'nhot': tf_fun.fixed_len_feature(
                dtype='float',
                length=self.nhot_size),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'nhot': {
                'dtype': tf.float32,
                'reshape': self.nhot_size
            },
            'label': {
                'dtype': tf.int64,
                'reshape': self.label_size
            }
        }

    def list_files(self, meta, directory, cat=0):
        """List files from metadata."""
        files, labs, nh = [], [], []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[cat],
                    f[2])]
            labs += [int(f[4])]
            nh += [[int(f[-2]), int(f[-1])]]

        # Derive n-hot labels
        n_hot = np.zeros([len(nh), np.max(nh) + 1])
        for idx, h in enumerate(nh):
            n_hot[idx, h[0]] = 1
            n_hot[idx, h[1]] = 1
        return np.asarray(files), np.asarray(labs), n_hot

    def get_data(self):
        """Get the names of files."""
        positive_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.data_name,
                self.meta))
        ims, labs, n_hot = self.list_files(
            positive_meta, self.data_name, cat=0)
        # labs = self.list_files(positive_meta, self.data_name, cat=1)

        if self.max_ims:
            all_ims = ims[:self.max_ims]
            all_labels = labs[:self.max_ims]
            all_nhot = n_hot[:self.max_ims]
        else:
            all_ims = ims
            all_labels = labs
            all_nhot = n_hot
        num_ims = len(all_ims)

        # Balance CV folds
        train_split = np.round(num_ims * self.cv_split).astype(int) // 2
        pos_idx = np.where(all_labels == 1)[0]
        neg_idx = np.where(all_labels == 0)[0]
        train_pos = pos_idx[:train_split]
        train_neg = neg_idx[:train_split]
        val_pos = pos_idx[train_split:]
        val_neg = neg_idx[train_split:]
        train_rand_idx = np.random.permutation(len(train_pos) * 2)
        train_ims = np.concatenate((
            all_ims[train_pos],
            all_ims[train_neg]))[train_rand_idx]
        train_labels = np.concatenate((
            all_labels[train_pos],
            all_labels[train_neg]))[train_rand_idx]
        train_nhot = np.concatenate((
            all_nhot[train_pos],
            all_nhot[train_neg]))[train_rand_idx]
        val_rand_idx = np.random.permutation(len(val_pos) * 2)
        val_ims = np.concatenate((
            all_ims[val_pos],
            all_ims[val_neg]))[val_rand_idx]
        val_labels = np.concatenate((
            all_labels[val_pos],
            all_labels[val_neg]))[val_rand_idx]
        val_nhot = np.concatenate((
            all_nhot[val_pos],
            all_nhot[val_neg]))[val_rand_idx]

        # Build CV dict
        cv_files, cv_labels, cv_nhot = {}, {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        cv_nhot[self.folds['train']] = train_nhot
        cv_nhot[self.folds['val']] = val_nhot
        return cv_files, cv_labels, cv_nhot
