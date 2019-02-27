#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import feedback_hgru as hgru
from config import Config
from ops import model_tools


def experiment_params():
    """Parameters for the experiment."""
    exp = {
        'lr': [1e-3],
        'loss_function': ['cce'],
        'optimizer': ['nadam'],
        'dataset': [
            # 'curv_contour_length_9',
            'cluttered_nist_caps_ix3_100',
            # 'curv_baseline',
        ]
    }
    exp['data_augmentations'] = [
        [
            'grayscale',
            #'left_right',
            #'up_down',
            'uint8_rescale',
            'singleton',
            'center_crop',
	    'resize',
            'zero_one'
        ]]
    exp['val_augmentations'] = exp['data_augmentations']
    exp['batch_size'] = 8  # Train/val batch size.
    exp['epochs'] = 400
    exp['exp_name'] = 'feedback_hgru_nist_caps_ix1'
    exp['model_name'] = 'learned_feedback_hgru'
    exp['save_weights'] = True
    exp['validation_iters'] = 500
    exp['num_validation_evals'] = 200
    exp['shuffle_val'] = True  # Shuffle val data.
    exp['shuffle_train'] = True
    return exp


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            conv_aux = {
                'pretrained': os.path.join(
                    'weights',
                    'gabors_for_contours_7.npy'),
                'pretrained_key': 's1',
                'nonlinearity': 'relu'
            }
            x = conv.conv_layer(
                bottom=data_tensor,
                name='gabor_input',
                stride=[1, 1, 1, 1],
                padding='SAME',
                trainable=training,
                use_bias=True,
                aux=conv_aux)
	    x = x[:,:,:,:24] ### To fit??
        with tf.variable_scope('hGRU', reuse=reuse):
            layer_hgru = hgru.hGRU(
                'hgru',
                x_shape=x.get_shape().as_list(),
                timesteps=8,
                h_ext=15,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'readout': 'l2'},
                train=training)
            x = layer_hgru.build(x)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                fused=True,
                training=training)
            fb = tf.identity(x)

        with tf.variable_scope('readout_1', reuse=reuse):
            x = conv.conv_layer(
                bottom=x,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=True)
            pool_aux = {'pool_type': 'max'}
            x = pooling.global_pool(
                bottom=x,
                name='pre_readout_pool',
                aux=pool_aux)
            x = normalization.batch(
                bottom=x,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            x = tf.layers.flatten(
                x,
                name='flat_readout')
            x = tf.layers.dense(
                inputs=x,
                units=2)
    return x, fb


def main(gpu_device='/gpu:0', cpu_device='/cpu:0'):
    """Run an experiment with hGRUs."""
    config = Config()
    params = experiment_params()
    model_tools.model_builder(
        params=params,
        config=config,
        model_spec=build_model,
        gpu_device=gpu_device,
        cpu_device=cpu_device)


if __name__ == '__main__':
    main()

