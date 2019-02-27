import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob


class data_processing(object):
    def __init__(self):
        self.name = 'ilsvrc12'
        self.output_name = 'ilsvrc12'
        self.img_dir = 'imgs'
        self.contour_dir = '/media/data_cifs/clicktionary/webapp_data'
        self.train_dir = 'lmdb_trains'
        self.val_dir = 'lmdb_validations'
        self.im_extension = '.JPEG'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [256, 256, 3]  # 600, 600
        self.model_input_image_size = [224, 224, 1]  # [107, 160, 3]
        self.max_ims = np.inf
        self.output_size = [1]
        self.force_output_size = [1000]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
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
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int64,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        """Get the names of files."""
        train_ims = glob(
            os.path.join(
                self.contour_dir,
                self.train_dir,
                '*%s' % self.im_extension))
        val_ims = glob(
            os.path.join(
                self.contour_dir,
                self.val_dir,
                '*%s' % self.im_extension))
        train_ims = np.array(train_ims)
        val_ims = np.array(val_ims)
        train_labels = np.array([int(x.split(os.path.sep)[-1].split('_')[0]) for x in train_ims])
        val_labels = np.array([int(x.split(os.path.sep)[-1].split('_')[0]) for x in val_ims])
        if not self.shuffle:
            raise NotImplementedError
        rand_idx = np.random.permutation(len(train_ims))
        train_ims = train_ims[rand_idx]
        train_labels = train_labels[rand_idx]
        rand_idx = np.random.permutation(len(val_ims))
        val_ims = val_ims[rand_idx]
        val_labels = val_labels[rand_idx]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels

