#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import hgru_loop as hgru
from config import Config
from utils import py_utils
from ops import data_loader
from ops import smooth_gradients


def experiment_params():
    """Parameters for the experiment."""
    exp = {
        'lr': [1e-3],
        'loss_function': ['cce'],
        'optimizer': ['nadam'],
        'dataset': [
            # 'curv_contour_length_9',
            'curv_contour_length_14',
            # 'curv_baseline',
        ]
    }
    exp['data_augmentations'] = [
        [
            'singleton',
            'grayscale',
            # 'left_right',
            # 'up_down',
            'uint8_rescale',
            'resize',
            # 'per_image_standardization',
            'zero_one'
        ]]
    exp['val_augmentations'] = exp['data_augmentations']
    exp['batch_size'] = 20  # Train/val batch size.
    exp['epochs'] = 4
    exp['exp_name'] = 'placeholder_hgru_pathfinder_14'
    exp['model_name'] = 'hgru'
    # exp['clip_gradients'] = 7.
    exp['save_weights'] = True
    exp['validation_iters'] = 1000
    exp['num_validation_evals'] = 50
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
                'nonlinearity': 'square'
            }
            x = conv.conv_layer(
                bottom=data_tensor,
                name='gabor_input',
                stride=[1, 1, 1, 1],
                padding='SAME',
                trainable=training,
                use_bias=True,
                aux=conv_aux)
            layer_hgru = hgru.hGRU(
                'hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=8,
                h_ext=15,
                strides=[1, 1, 1, 1],
                padding='SAME',
                aux={'symmetric_weights': False},
                train=training)
            h2, e_grad_acts, i_grad_acts = layer_hgru.build(x)
            nh2 = normalization.batch(
                bottom=h2,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=nh2,
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
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=activity,
                units=2)
    return activity, e_grad_acts, i_grad_acts


def main(
        experiment_name='hgru',
        gpu_device='/gpu:0',
        cpu_device='/cpu:0',
        # images='/media/data_cifs/curvy_2snakes_300/curv_contour_length_14/imgs/1',
        images='/media/data_cifs/drew/pathfinder_examples/both',
        restore='/media/data_cifs/contextual_circuit/checkpoints/hgru_2018_07_25_10_36_40_503580/model_75000.ckpt-75000',
        im_ext='.png'):
    """Run an experiment with hGRUs."""

    # Prepare to run the model
    config = Config()
    params = experiment_params()
    config = py_utils.add_to_config(
        d=params,
        config=config)
    exp_label = '%s_%s' % (params['exp_name'], py_utils.get_dt_stamp())
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=config.dataset)
    dataset_module = dataset_module.data_processing()  # hardcoded class name
    val_key = [k for k in dataset_module.folds.keys() if 'val' in k]
    if not len(val_key):
        val_key = 'train'
    else:
        val_key = val_key[0]
    val_data, val_means_image, val_means_label = py_utils.get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=val_key)

    # Create data tensors
    with tf.device(cpu_device):
        val_images = tf.placeholder(
            dtype=dataset_module.tf_reader['image']['dtype'],
            shape=[config.batch_size] + dataset_module.im_size,
            name='test_images')
        val_labels = tf.placeholder(
            dtype=dataset_module.tf_reader['label']['dtype'],
            shape=[config.batch_size] + dataset_module.output_size,
            name='test_labels')
        proc_val_images, _ = data_loader.placeholder_image_augmentations(
            images=val_images,
            model_input_image_size=dataset_module.model_input_image_size,
            data_augmentations=config.val_augmentations,
            batch_size=config.batch_size,
            labels=val_labels)

    # Build training and val models
    with tf.device(gpu_device):
        val_logits, exc_val_activities, inh_val_activities = build_model(
            data_tensor=proc_val_images,
            reuse=None,
            training=False)

    # Derive gradients
    exc_val_gradients = tf.gradients(exc_val_activities[-1], val_images)
    # exc_val_gradients_f = tf.gradients(exc_val_activities[0], val_images)
    # exc_val_gradients = exc_val_gradients[0] - exc_val_gradients_f[0]
    inh_val_gradients = tf.gradients(inh_val_activities[-1], val_images)
    # inh_val_gradients_f = tf.gradients(inh_val_activities[0], val_images)
    # inh_val_gradients = inh_val_gradients[0] - inh_val_gradients_f[0]

    # Initialize tf variables
    saver = tf.train.Saver(
        var_list=tf.global_variables(),
        max_to_keep=5)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))

    # Create dictionaries of important training and validation information
    val_dict = {
        'val_images': val_images,
        'proc_val_images': proc_val_images,
        'val_logits': val_logits,
        'exc_val_gradients': exc_val_gradients,
        'inh_val_gradients': inh_val_gradients,
        'exc_val_activities': exc_val_activities[-1] - exc_val_activities[0],
        'inh_val_activities': inh_val_activities[-1] - inh_val_activities[0]
    }

    # Start training loop
    smooth_gradients.evaluate(
        images=images,
        im_ext=im_ext,
        config=config,
        sess=sess,
        saver=saver,
        val_dict=val_dict,
        exp_label=exp_label,
        restore=restore)


if __name__ == '__main__':
    main()
