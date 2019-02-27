import os
import re
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob
from utils import py_utils
from scipy import interpolate, signal
from tqdm import tqdm
import moviepy.editor as mp


def smooth(x, window_len=3, window='flat', mode='valid'):
    """smooth the data using a window with requested size."""
    if window_len < 3:
        return x
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode=mode)
    return y


def resample_detrend_zscore(
        samp, new_fps, old_fps=15, frame_axis=-1, norm_axes=(0, 1)):
    """Resample video to fps."""
    samp_shape = list(samp.shape)
    shaped_samp = samp.reshape(-1, samp_shape[frame_axis])

    def interp_pixel(x, new_fps, old_fps):
        """Apply interpolation to pixel."""
        len_x = len(x)
        f = interpolate.interp1d(np.arange(len_x), x)
        new_num_frames = new_fps * (len_x / float(old_fps))
        return f(np.linspace(0, len_x - 1, new_num_frames))
    if new_fps:
        res_samp = np.apply_along_axis(
            func1d=lambda x: interp_pixel(x, new_fps, old_fps),
            axis=-1,
            arr=shaped_samp)
    else:
        res_samp = shaped_samp
    samp_shape.pop(frame_axis)
    if frame_axis == 0:
        samp_shape = [res_samp.shape[-1]] + samp_shape
    else:
        samp_shape = samp_shape + [res_samp.shape[-1]]

    # Detrend
    res_samp = signal.detrend(res_samp, axis=0)

    # Zscore
    res_samp = (
        res_samp - res_samp.mean(
            axis=norm_axes, keepdims=True)) / res_samp.std(
                axis=norm_axes, keepdims=True)

    # Reshape
    res_samp = res_samp.reshape(samp_shape)
    return res_samp


class data_processing(object):
    def __init__(self):
        self.output_name = 'deconv_cv_retina_ca_ephys'
        self.data_dir = '/media/data_cifs/retina'
        self.ca_split = 'full_frame_response_after_registration'
        self.dff_split = 'fluorescence_trace_cell_wise'
        self.ephys_split = 'ephys_new_analyses/spike_rates'
        self.neural_dir_ca = os.path.join(self.data_dir, self.ca_split)
        self.neural_dir_ephys = os.path.join(self.data_dir, self.ephys_split)
        self.neural_dir_dff = os.path.join(self.data_dir, self.ephys_split)
        self.video_dir = os.path.join(self.data_dir, 'all_m4vs')
        self.cell_mask = os.path.join(self.data_dir, 'all_ROI_105masks.npy')
        self.temp_dir = os.path.join(
            self.data_dir, 'reprocessed_retina_ca_ephys')
        self.ephys_suffix = ''
        self.dff_suffix = 'dff_traces'
        self.config = Config()
        self.resize_factor = 1
        self.process_data = False
        self.smooth_ephys = False
        self.video_timesteps = 121
        self.binarize_spikes = False
        self.total_frames = 121
        self.model_timesteps = 30
        self.num_dff_cells = 105
        self.aggregate = np.mean  # Fun for aggregating correlations in a cell
        self.video_frame_size = [1080, 1440]  # 600, 600
        self.im_size = [
            self.model_timesteps,
            self.video_frame_size[0] / self.resize_factor,
            self.video_frame_size[1] / self.resize_factor,
            1
        ]
        # self.model_input_image_size = self.im_size
        self.model_input_image_size = [
            self.model_timesteps,
            256,
            256,  # self.video_frame_size[1] / self.resize_factor,
            1
        ]
        self.output_size = [
            self.model_timesteps,
            128,
            256,
            1
        ]
        # self.label_size = self.output_size
        self.label_size = [
            self.video_timesteps,
            128,
            256,  # 320,
            1
        ]
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.balance = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['']  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'video': tf_fun.bytes_feature,
            'ca': tf_fun.bytes_feature,
            'ephys': tf_fun.bytes_feature,
            'index': tf_fun.int64_feature
        }
        self.tf_dict = {
            'video': tf_fun.fixed_len_feature(dtype='string'),
            'ca': tf_fun.fixed_len_feature(dtype='string'),
            'ephys': tf_fun.fixed_len_feature(dtype='string'),
            'index': tf_fun.fixed_len_feature(
                dtype='int64', length=[self.video_timesteps])
        }
        self.tf_reader = {
            'video': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'ca': {
                'dtype': tf.float32,
                'reshape': self.output_size
            },
            'ephys': {
                'dtype': tf.float32,
                'reshape': [self.model_timesteps]
            },
            'index': {
                'dtype': tf.int64,
                'reshape': [self.video_timesteps]
            }
        }

    def list_files(self, meta, directory):
        """List files from metadata."""
        files = []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
        return np.asarray(files)

    def get_data(self):
        """Get the names of files."""
        res_height = self.video_frame_size[0] / self.resize_factor
        if self.process_data:
            py_utils.make_dir(self.temp_dir)
            videos_m4v = glob(
                os.path.join(
                    self.video_dir,
                    '*.m4v'))
            ca2_npy = glob(
                os.path.join(
                    self.neural_dir_ca,
                    '*.npy'))
            ca2_match = filter(
                re.compile('(.+)_MV_(.+)').match, ca2_npy)
            num_frame_sets = np.arange(
                self.total_frames / self.video_timesteps)
            frame_sets_idx = num_frame_sets.reshape(1, -1).repeat(
                self.video_timesteps, axis=1).reshape(-1)
            frame_idx = np.arange(self.total_frames)
            fr = []
            res_data = []
            video_idx = []
            for idx, v in tqdm(enumerate(videos_m4v), total=len(videos_m4v)):
                vname = v.split(os.path.sep)[-1].split('.')[0]

                # Load the video, resize, and normalize to [0, 1]
                # vid = np.load(v.replace('.m4v', '.npy'))
                vid = mp.VideoFileClip(v)
                clip_resized = vid.resize(height=res_height)
                fr += [vid.fps]
                frames = np.asarray(
                    [frame for frame in clip_resized.iter_frames()])
                if frames[0].max() > 1:
                    frames = frames.astype(np.float32) / 255.
                frames = frames[:self.total_frames]
                # Find matching ca2+ vids
                it_ca2 = filter(
                    re.compile('(.+)_%s_(.+)' % vname).match, ca2_match)
                for tr, samp_path in enumerate(it_ca2):
                    # Resample, detrend, zscore
                    samp = resample_detrend_zscore(
                        np.load(samp_path),
                        new_fps=False)
                    samp = samp.transpose(2, 0, 1)
                    samp = samp[:len(frames)]
                    esamp_file = '%s.npy' % samp_path.strip('full_frame.npy')
                        # self.ephys_suffix)
                    esamp_file = esamp_file.replace(
                        self.ca_split, self.ephys_split)
                    esamp = np.load(esamp_file)[:len(frames)]
                    dffsearch = os.path.join(
                        os.path.sep.join(samp_path.split(os.path.sep)[:4]),
                        self.dff_split,
                        self.dff_suffix,
                        samp_path.split(os.path.sep)[-1])
                    dffglob = glob(
                        '%s_cell*' % dffsearch.strip('full_frame.npy'))
                    dffsamp = np.asarray(
                        [np.load(d)[:len(frames)] for d in dffglob])
                    dffsamp = resample_detrend_zscore(
                        dffsamp,
                        new_fps=False)
                    dffsamp = dffsamp.transpose()
                    dffsamp = dffsamp[:len(frames)]
                    for fs in np.unique(frame_sets_idx):
                        fi = frame_sets_idx == fs
                        # Trim samp and vid and save to npz in temp folder
                        fname = os.path.join(
                            self.temp_dir,
                            '%s_%s_fs_%s' % (vname, tr, fs))
                        video_idx += [idx]
                        res_data += [fname]
                        it_esamp = esamp[:self.model_timesteps]
                        samp = samp[:self.model_timesteps]
                        if self.smooth_ephys:
                            it_esamp = smooth(
                                it_esamp,
                                window='flat',
                                mode='same',
                                window_len=5)[5:-5]
                        try:
                            np.savez(
                                fname,
                                ca=samp,  # [fi],  # Make sure this is correct!
                                ephys=it_esamp,
                                frame_idx=frame_idx[fi],
                                dff=dffsamp[:self.model_timesteps],  # [fi],
                                video=frames[fi])
                        except Exception as e:
                            import ipdb;ipdb.set_trace()
            np.save(
                os.path.join(self.temp_dir, 'video_idx'), video_idx)
        else:
            res_data = glob(
                os.path.join(
                    self.temp_dir,
                    '*.npz'))
            res_data = [
                rs for rs in res_data
                if 'retina' not in rs.split(os.path.sep)[-1]]
            # video_idx = np.load(
            #     os.path.join(
            #         self.temp_dir,
            #         'video_idx.npy'))
        assert len(res_data)
        res_data = np.asarray(res_data)
        video_idx = np.asarray(
            [f.split(os.path.sep)[-1].split('g_')[0]
                for f in res_data]).astype(int)

        # Calculate reliability
        unique_videos, video_counts = np.unique(video_idx, return_counts=True)
        repeated_videos = np.in1d(
            video_idx, unique_videos[np.where(
                video_counts > 1)[0]])
        repeated_videos = video_idx[repeated_videos]
        dff_reliability, ca_reliability = [], []
        e_reliability, variability = [], []
        mask = np.load(self.cell_mask)
        for video in tqdm(
                np.unique(repeated_videos),
                desc='Calculating reliability',
                total=len(repeated_videos) // 2):
            it_idx = np.in1d(video_idx, video)
            sel_data = res_data[it_idx]
            data_a = np.load(sel_data[0])
            data_b = np.load(sel_data[1])
            dff_a = data_a['dff']
            dff_b = data_b['dff']
            ephys_a = data_a['ephys']
            ephys_b = data_b['ephys']
            ca_a = data_a['ca'].reshape(dff_a.shape[0], -1)
            ca_b = data_b['ca'].reshape(dff_a.shape[0], -1)
            if self.binarize_spikes:
                ephys_a = (ephys_a > 0).astype(np.int32)
                ephys_b = (ephys_b > 0).astype(np.int32)
            proc_ca_a, proc_ca_b = [], []
            for cell in range(mask.shape[-1]):
                it_cells = mask[:, :, cell].ravel()
                cell_locations = np.where(it_cells)
                proc_ca_a += [np.asarray(
                    [ca_a[idx, cell_locations]
                        for idx in range(ca_a.shape[0])])]
                proc_ca_b += [np.asarray(
                    [ca_b[idx, cell_locations]
                        for idx in range(ca_b.shape[0])])]
            ca_rs = []
            var = []
            for pa, pb in zip(proc_ca_a, proc_ca_b):
                # Loop through cells
                pa = pa.squeeze()
                pb = pb.squeeze()
                rscores = [
                    np.corrcoef(
                        pa[:, idx],
                        pb[:, idx])[0, 1]
                    for idx in range(pa.shape[-1])]
                arg_idx = np.argmax(rscores)
                rscores = self.aggregate(rscores)
                var += [[np.std(pa[:, arg_idx]), np.std(pb[:, arg_idx])]]
                ca_rs += [rscores]
            # ca_rs = self.aggregate(ca_rs)
            ca_reliability += [ca_rs]
            variability += [var]
            dff_reliability += [
                np.mean(
                    [np.corrcoef(dff_a[:, idx], dff_b[:, idx])[0, 1]
                        for idx in range(dff_a.shape[-1])])]
            e_reliability += [np.corrcoef(ephys_a.ravel(), ephys_b.ravel())[0, 1]]
        ca_reliability = np.asarray(ca_reliability)
        dff_reliability = np.asarray(dff_reliability)
        e_reliability = np.asarray(e_reliability)
        variability = np.stack(
            [np.asarray(v) for v in variability]).mean(0).mean(-1)

        # Bimodal difference at 0.8
        mask_thresh_idx = variability > 0.8
        mask = (mask * mask_thresh_idx[None, None, :]).transpose(2, 0, 1)
        np.savez(
            os.path.join(
                self.temp_dir,
                '%s_reliabilities' % self.output_name),
            ca_reliability=ca_reliability,
            dff_reliability=dff_reliability,
            proc_mask=mask,
            mask_thresh_idx=mask_thresh_idx,
            e_reliability=e_reliability,
            variability=variability)
        print 'Ca2+ video reliability: %s' % ca_reliability[
            :, mask_thresh_idx].mean()
        print 'dff video reliability: %s' % np.mean(dff_reliability)
        print 'ephys video reliability: %s' % np.nanmean(e_reliability)

        # Stimulus period is 5.15 seconds. Ca2+ is 111 frames at 15fps.
        # Means that there is 7.326 seconds of recording from Ca2+.
        # Video is variable framerate.
        # For each video resample ca2+ to 25fps then align both.
        # Trim this down to 76 for 5000 ms of ca2+ data

        # Create CV folds
        # num_files = len(np.unique(video_idx))
        # cv_range = np.random.permutation(num_files)
        # cv_split = np.round(num_files * self.cv_split).astype(int)
        # train_idx = cv_range[:cv_split]
        # validation_idx = cv_range[cv_split:]
        # train_idx = np.in1d(video_idx, train_idx)
        # validation_idx = np.in1d(video_idx, validation_idx)
        validation_idx = []
        validation_idx = np.asarray(
            [np.where(video_idx == reps)[0][0]
                for reps in np.unique(repeated_videos)])
        train_idx = ~np.in1d(video_idx, validation_idx)
        train_files = res_data[train_idx]
        validation_files = res_data[validation_idx]  # np.in1d(video_idx, validation_idx)]
        import ipdb;ipdb.set_trace()
        if self.shuffle:
            rand_idx = np.random.permutation(len(train_files))
            train_files = train_files[rand_idx]
            rand_idx = np.random.permutation(len(validation_files))
            validation_files = validation_files[rand_idx]

        # Build CV dict
        cv_files = {}
        cv_files[self.folds['train']] = train_files
        cv_files[self.folds['val']] = validation_files
        return cv_files
