#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru_v3 as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=20,
            conv_activation=[tf.nn.relu, None],
            conv_kernel_size=7,
            pool_kernel_size=[1, 2, 2, 1],
            pool_kernel_strides=[1, 2, 2, 1],
            pool=True,
            name='l0')
        layer_hgru = hgru.hGRU(
            'fgru',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=1,
            h_ext=[{'h1': [15, 15]}, {'h2': [1, 1]}, {'fb1': [1, 1]}],
            strides=[1, 1, 1, 1],
            hgru_ids=[{'h1': 20}, {'h2': 128}, {'fb1': 20}],
            hgru_idx=[{'h1': 0}, {'h2': 1}, {'fb1': 2}],
            padding='SAME',
            aux={
                'readout': 'fb',
                'intermediate_ff': [32, 128],
                'intermediate_ks': [[3, 3], [3, 3]],
                'intermediate_repeats': [3, 3],
                'while_loop': False,
                'skip': False,
                'symmetric_weights': True,
                'include_pooling': True
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

