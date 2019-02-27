#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import pure_h_fgru_v2 as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.skinny_input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=20,
            conv_activation=tf.nn.relu,  # [tf.nn.relu, None],
            conv_kernel_size=7,
            pool=True,
            name='l0')
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=[{'h1': [15, 15]}],
            strides=[1, 1, 1, 1],
            hgru_ids=[{'h1': 20}],
            hgru_idx=[{'h1': 0}],
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [],
                'intermediate_ks': [],
                'intermediate_repeats': [0],
                'while_loop': False,
                'skip': False,
                'dtype': tf.float32,
                'force_horizontal': True,
                'symmetric_weights': True,
                'include_pooling': False
            },
            pool_strides=[2, 2],
            pooling_kernel=[2, 2],
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
