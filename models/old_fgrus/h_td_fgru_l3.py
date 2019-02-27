#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import constrained_h_td_fgru_v3 as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    dtype = data_tensor.dtype
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.input_layer_v2(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=24,
            conv_activation=tf.nn.relu,
            conv_kernel_size=7,
            pool_kernel_size=[1, 2, 2, 1],
            pool_kernel_strides=[1, 2, 2, 1],
            pool=True,
            name='l0')
        layer_hgru = hgru.hGRU(
            name='fgru',
            X=in_emb,
            timesteps=8,
            padding='SAME',
            train=training,
            skip=False,
            readout='td1',
            symmetric_weights=True,
            aux={
                'while_loop': False,
                'dtype': dtype,
                'return_activations': False,
            },
            down_layers=[
                {'h1': {
                    'ff': {
                        'features': 24,
                        'kernels': [7, 7],
                        'strides': [1, 1, 1, 1],
                        'repeats': 1,
                        'dilation': [1, 1, 1, 1]},  # Add pool dict
                    'recurrent': {
                        'h_kernel': [15, 15],
                        'g_kernel': [1, 1],
                        'features': 24,
                        'dilation': [1, 1, 1, 1, 1]},
                    'pool': {
                        'pool_kernel': [1, 2, 2, 1],
                        'pool_stride': [1, 2, 2, 1]}
                }},
                {'h2': {
                    'ff': {
                        'features': 32,
                        'kernels': [3, 3],
                        'strides': [1, 1, 1, 1],
                        'repeats': 3,
                        'dilation': [1, 1, 1, 1]},
                    'recurrent': {
                        'h_kernel': [5, 5],
                        'g_kernel': [1, 1],
                        'features': 32,
                        'dilation': [1, 1, 1, 1, 1]},
                    'pool': {
                        'pool_kernel': [1, 2, 2, 1],
                        'pool_stride': [1, 2, 2, 1]}
                }},
                {'h3': {
                    'ff': {
                        'features': 128,
                        'kernels': [3, 3],
                        'strides': [1, 1, 1, 1],
                        'repeats': 1,
                        'dilation': [1, 1, 1, 1]},
                    'recurrent': {
                        'h_kernel': [1, 1],
                        'g_kernel': [1, 1],
                        'features': 128,
                        'dilation': [1, 1, 1, 1, 1]},
                    'pool': {
                        'pool_kernel': [1, 2, 2, 1],
                        'pool_stride': [1, 2, 2, 1]}
                }},
            ],
            up_layers=[
                {'td2': {
                    'up': {
                        'pool_kernel': [1, 4, 4, 1],
                        'pool_stride': [1, 2, 2, 1]},
                    'recurrent': {
                        'h_kernel': [1, 1], 
                        'features': 32,
                        'dilation': [1, 1, 1, 1, 1]}}},
                {'td1': {
                    'up': {
                        'pool_kernel': [1, 4, 4, 1],
                        'pool_stride': [1, 2, 2, 1]},
                    'recurrent': {
                        'h_kernel': [1, 1],  
                        'features': 24,
                        'dilation': [1, 1, 1, 1, 1]}}}
            ]
        )
        h2 = layer_hgru.build()
        h2 = normalization.batch_contrib(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        h2 = conv.up_layer(
            layer_name='readout_up',
            bottom=h2,
            reuse=reuse,
            kernel_size=[4, 4],
            num_filters=24,
            training=training,
            stride=[2, 2],
            use_bias=True)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            dtype=dtype,
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities

