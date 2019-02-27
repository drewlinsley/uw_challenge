#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.feedforward import misc
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
            # 'curv_contour_length_14_full',
            # 'curv_baseline',
            'synth_connectomics_baseline',
            # 'synth_connectomics_ix_1',
            # 'synth_connectomics_ix_2',
            # 'synth_connectomics_ix_3',
            # 'shapes_held_out_light'
        ]
    }
    exp['val_dataset'] = 'synth_connectomics_iy_3'
    exp['data_augmentations'] = [
        [
            # 'grayscale',
            'lr_flip_image_label',
            'ud_flip_image_label',
            'uint8_rescale',
            'uint8_rescale_label',
            'singleton',
            'singleton_label',
            'random_crop_image_label',
        ]]
    exp['model_name'] = 'unet'
    exp['exp_name'] = 'unet_ix_1'
    exp['loss_type'] = 'pearson'
    exp['metric_type'] = 'pearson'
    exp['val_augmentations'] = [
        [
            'singleton',
            'singleton_label',
            'uint8_rescale',
            'uint8_rescale_label',
            'center_crop_image_label',
        ]]
    exp['batch_size'] = 2  # Train/val batch size.
    exp['epochs'] = 4
    exp['save_weights'] = True
    exp['validation_iters'] = 1000
    exp['num_validation_evals'] = 50
    exp['shuffle_val'] = True  # Shuffle val data.
    exp['shuffle_train'] = True
    return exp


def conv_block(
        x,
        filters,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        kernel_initializer=tf.initializers.variance_scaling,
        data_format='channels_last',
        activation=tf.nn.relu,
        name=None,
        training=True,
        reuse=False,
        batchnorm=True,
        pool=True):
    """VGG conv block."""
    assert name is not None, 'Give the conv block a name.'
    activity = tf.layers.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        name='%s_conv' % name,
        reuse=reuse)
    if batchnorm:
        activity = normalization.batch(
            bottom=activity,
            name='%s_bn' % name,
            training=training,
            reuse=reuse)
    if pool:
        return pooling.max_pool(
            bottom=activity,
            name='%s_pool' % name)
    else:
        return activity


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        # Unclear if we should include l0 in the down/upsample cascade
        with tf.variable_scope('g1', reuse=reuse):
            # Downsample
            act11 = conv_block(
                x=data_tensor,
                name='l1_1',
                filters=64,
                training=training,
                reuse=reuse,
                pool=False)
            act12 = conv_block(
                x=act11,
                name='l1_2',
                filters=64,
                training=training,
                reuse=reuse)

        with tf.variable_scope('g2', reuse=reuse):
            # Downsample
            act21 = conv_block(
                x=act12,
                name='l2_1',
                filters=128,
                training=training,
                reuse=reuse,
                pool=False)
            act22 = conv_block(
                x=act21,
                filters=128,
                name='l2_2',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g3', reuse=reuse):
            # Downsample
            act31 = conv_block(
                x=act22,
                name='l3_1',
                filters=256,
                training=training,
                reuse=reuse,
                pool=False)
            act32 = conv_block(
                x=act31,
                filters=256,
                name='l3_2',
                training=training,
                reuse=reuse,
                pool=False)
            act33 = conv_block(
                x=act32,
                filters=256,
                name='l3_3',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g4', reuse=reuse):
            # Downsample
            act41 = conv_block(
                x=act33,
                name='l4_1',
                filters=512,
                training=training,
                reuse=reuse,
                pool=False)
            act42 = conv_block(
                x=act41,
                filters=512,
                name='l4_2',
                training=training,
                reuse=reuse,
                pool=False)
            act43 = conv_block(
                x=act42,
                filters=512,
                name='l4_3',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g5', reuse=reuse):
            # Downsample
            act51 = conv_block(
                x=act43,
                name='l5_1',
                filters=512,
                training=training,
                reuse=reuse,
                pool=False)
            act52 = conv_block(
                x=act51,
                filters=512,
                name='l5_2',
                training=training,
                reuse=reuse,
                pool=False)
            act53 = conv_block(
                x=act52,
                filters=512,
                name='l5_3',
                training=training,
                reuse=reuse)

        with tf.variable_scope('resize', reuse=reuse):
            sel_layers = [
                act12,
                act22,
                act33,
                act43,
                act53]
            target_size = act12.get_shape().as_list()[1:3]
            for idx, l in enumerate(sel_layers):
                sel_layers[idx] = misc.resize(
                    x=l,
                    size=target_size,
                    method='bilinear')
                sel_layers = tf.concat(sel_layers, axis=-1)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=sel_layers,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=False)
            pool_aux = {'pool_type': 'max'}
            activity = pooling.global_pool(
                bottom=activity,
                name='pre_readout_pool',
                aux=pool_aux)
            activity = normalization.batch(
                bottom=activity,
                renorm=True,
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=activity,
                units=2)
    return activity, activity


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
