#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling


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
        pool=False,
        reuse=reuse)


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    elif isinstance(output_shape, dict):
        nhot_shape = output_shape['aux']
        output_shape = output_shape['output']
        use_aux = True
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
            poolact12 = pooling.max_pool(
                bottom=act12,
                name='l1_2_pool')

        with tf.variable_scope('g2', reuse=reuse):
            # Downsample
            act21 = conv_block(
                x=poolact12,
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
            poolact22 = pooling.max_pool(
                bottom=act22,
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
            poolact33 = pooling.max_pool(
                bottom=act33,
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
            poolact43 = pooling.max_pool(
                bottom=act43,
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
            poolact53 = pooling.max_pool(
                bottom=act53,
                name='l5_3_pool')

        with tf.variable_scope('g5_skip', reuse=reuse):
            upact5 = up_block(
                inputs=poolact53,
                skip=act53,
                up_filters=512,
                name='ul5',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g4_skip', reuse=reuse):
            upact4 = up_block(
                inputs=upact5,
                skip=act43,
                up_filters=512,
                name='ul4',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g3_skip', reuse=reuse):
            upact3 = up_block(
                inputs=upact4,
                skip=act33,
                up_filters=256,
                name='ul3',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g2_skip', reuse=reuse):
            upact2 = up_block(
                inputs=upact3,
                skip=act22,
                up_filters=128,
                name='ul2',
                training=training,
                reuse=reuse)

        with tf.variable_scope('g1_skip', reuse=reuse):
            upact1 = up_block(
                inputs=upact2,
                skip=act12,
                up_filters=64,
                name='ul1',
                training=training,
                reuse=reuse)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=upact1,
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
            pre_activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=pre_activity,
                units=output_shape)
        if use_aux:
            nhot = tf.layers.dense(inputs=pre_activity, units=nhot_shape)
        else:
            nhot = tf.constant(0.)
    extra_activities = {
        'activity': activity,
        'nhot': nhot
    }
    return activity, extra_activities
