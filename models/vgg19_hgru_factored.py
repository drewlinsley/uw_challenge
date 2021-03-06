#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import vgg19, conv
from layers.recurrent import hgru_bn_for as hgru
from layers.feedforward import normalization


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        mask = conv.create_mask(data_tensor)  # , dilation=[3., 3., 1.])
        with tf.variable_scope('freeze', reuse=reuse):
            net = vgg19.Model(
                vgg19_npy_path='/media/data_cifs/uw_challenge/checkpoints/vgg19.npy')
            x, mask = net.build(
                rgb=data_tensor,
                up_to='c2',
                mask=mask,
                training=training)
        with tf.variable_scope('scratch', reuse=reuse):
            x = tf.layers.conv2d(
                inputs=x,
                filters=32,
                kernel_size=(1, 1),
                padding='same')
            # x *= mask
            layer_hgru = hgru.hGRU(
                layer_name='hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=8,
                h_ext=7,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'reuse': False, 'constrain': False, 'recurrent_nl': tf.nn.relu},
                train=training)
            h2 = layer_hgru.build(x)
            # h2 *= mask

        with tf.variable_scope('scratch_readout', reuse=reuse):
            x = normalization.batch(
                bottom=h2,
                # renorm=True,
                name='hgru_bn',
                reuse=reuse,
                training=training)        
            crop = x[:, 21:35, 22:33, :]
            x = tf.contrib.layers.flatten(crop)
            x = tf.layers.dense(inputs=x, units=output_shape)
            # h2 *= mask
        # x, ro_weights = conv.full_mask_readout(
        #     activity=x,
        #     reuse=reuse,
        #     training=training,
        #     mask=mask,
        #     output_shape=output_shape,
        #     # kernel_size=[21, 21],
        #     REDUCE=tf.reduce_max,
        #     learnable_pool=False)
    extra_activities = {
        'activity': net.conv1_1,
        'h2': h2,
        'mask': mask,
        'crop': crop
        # 'ro_weights': ro_weights
    }
    return x, extra_activities

