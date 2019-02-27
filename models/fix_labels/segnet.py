#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from config import Config
from ops import model_tools


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


def up_block(
        inputs,
        skip,
        up_filters,
        name,
        training,
        reuse,
        up_kernels=4,
        up_strides=(2, 2),
        up_padding='same',
        nl=tf.nn.relu):
    """Do a unet upsample."""
    upact = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=up_filters,
        kernel_size=up_kernels,
        strides=up_strides,
        name='%s_up' % name,
        padding=up_padding)
    upact = nl(upact)
    upact = normalization.batch(
        bottom=upact,
        name='%s_bn' % name,
        training=training,
        reuse=reuse)
    return conv_block(
        x=upact + skip,
        filters=up_filters,
        name='%s_skipped' % name,
        training=training,
        reuse=reuse)


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    down_pool_kernel = [1, 2, 2, 1]
    down_pool_strides = [1, 2, 2, 1]
    down_pool_padding = 'SAME'
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
                reuse=reuse,
                pool=False)
            poolact12, poolact12inds = tf.nn.max_pool_with_argmax(
                input=act12,
                ksize=down_pool_kernel,
                strides=down_pool_strides,
                padding=down_pool_padding,
                name='l1_2_pool')

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
                reuse=reuse,
                pool=False)
            poolact22, poolact22inds = tf.nn.max_pool_with_argmax(
                input=act22,
                ksize=down_pool_kernel,
                strides=down_pool_strides,
                padding=down_pool_padding,
                name='l2_2_pool')

        with tf.variable_scope('g3', reuse=reuse):
            # Downsample
            act31 = conv_block(
                x=poolact22,
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
                reuse=reuse,
                pool=False)
            poolact33, poolact33inds = tf.nn.max_pool_with_argmax(
                input=act33,
                ksize=down_pool_kernel,
                strides=down_pool_strides,
                padding=down_pool_padding,
                name='l3_3_pool')

        with tf.variable_scope('g4', reuse=reuse):
            # Downsample
            act41 = conv_block(
                x=poolact33,
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
                reuse=reuse,
                pool=False)
            poolact43, poolact43inds = tf.nn.max_pool_with_argmax(
                input=act43,
                ksize=down_pool_kernel,
                strides=down_pool_strides,
                padding=down_pool_padding,
                name='l4_3_pool')

        with tf.variable_scope('g5', reuse=reuse):
            # Downsample
            act51 = conv_block(
                x=poolact43,
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
                reuse=reuse,
                pool=False)
            poolact53, poolact53inds = tf.nn.max_pool_with_argmax(
                input=act53,
                ksize=down_pool_kernel,
                strides=down_pool_strides,
                padding=down_pool_padding,
                name='l5_3_pool')

        with tf.variable_scope('g5_up', reuse=reuse):
            upact5 = pooling.unpool_with_argmax_layer(
                bottom=poolact53,
                ind=poolact53inds,
                filter_size=[3, 3],
                name='l5_unpool')
            uact53 = conv_block(
                x=upact5,
                name='ul5_3',
                filters=512,
                training=training,
                reuse=reuse,
                pool=False)
            uact52 = conv_block(
                x=uact53,
                filters=512,
                name='ul5_2',
                training=training,
                reuse=reuse,
                pool=False)
            uact51 = conv_block(
                x=uact52,
                filters=512,
                name='ul5_1',
                training=training,
                reuse=reuse,
                pool=False)

        with tf.variable_scope('g4_up', reuse=reuse):
            upact4 = pooling.unpool_with_argmax_layer(
                bottom=uact51,
                ind=poolact43inds,
                filter_size=[3, 3],
                name='l4_unpool')
            uact43 = conv_block(
                x=upact4,
                name='ul4_3',
                filters=512,
                training=training,
                reuse=reuse,
                pool=False)
            uact42 = conv_block(
                x=uact43,
                filters=512,
                name='ul4_2',
                training=training,
                reuse=reuse,
                pool=False)
            uact41 = conv_block(
                x=uact42,
                filters=256,
                name='ul4_1',
                training=training,
                reuse=reuse,
                pool=False)

        with tf.variable_scope('g3_up', reuse=reuse):
            upact3 = pooling.unpool_with_argmax_layer(
                bottom=uact41,
                ind=poolact33inds,
                filter_size=[3, 3],
                name='l3_unpool')
            uact33 = conv_block(
                x=upact3,
                name='ul3_3',
                filters=256,
                training=training,
                reuse=reuse,
                pool=False)
            uact32 = conv_block(
                x=uact33,
                filters=256,
                name='ul3_2',
                training=training,
                reuse=reuse,
                pool=False)
            uact31 = conv_block(
                x=uact32,
                filters=128,
                name='ul3_1',
                training=training,
                reuse=reuse,
                pool=False)

        with tf.variable_scope('g2_up', reuse=reuse):
            upact2 = pooling.unpool_with_argmax_layer(
                bottom=uact31,
                ind=poolact22inds,
                filter_size=[3, 3],
                name='l2_unpool')
            uact22 = conv_block(
                x=upact2,
                name='ul2_2',
                filters=128,
                training=training,
                reuse=reuse,
                pool=False)
            uact21 = conv_block(
                x=uact22,
                name='ul2_1',
                filters=64,
                training=training,
                reuse=reuse,
                pool=False)

        with tf.variable_scope('g1_up', reuse=reuse):
            upact1 = pooling.unpool_with_argmax_layer(
                bottom=uact21,
                ind=poolact12inds,
                filter_size=[3, 3],
                name='l1_unpool')
            uact12 = conv_block(
                x=upact1,
                name='ul1_2',
                filters=64,
                training=training,
                reuse=reuse,
                pool=False)
            uact11 = conv_block(
                x=uact12,
                name='ul1_1',
                filters=64,
                training=training,
                reuse=reuse,
                pool=False)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=uact11,
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
    extra_activities = {
        'activity': activity
    }

    return activity, extra_activities

