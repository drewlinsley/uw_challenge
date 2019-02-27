#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import resnet
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
            'cluttered_nist_baseline',
            # 'curv_baseline',
        ]
    }
    exp['data_augmentations'] = [
        [
            'grayscale',
            'center_crop',
            # 'left_right',
            # 'up_down',
            'uint8_rescale',
            'singleton',
            'zero_one'
        ]]
    exp['val_augmentations'] = [
        [
            'grayscale',
            'center_crop',
            # 'left_right',
            # 'up_down',
            'uint8_rescale',
            'singleton',
            'zero_one'
        ]]
    exp['batch_size'] = 32  # Train/val batch size.
    exp['epochs'] = 4
    exp['model_name'] = 'resnet'
    exp['exp_name'] = exp['model_name'] + '_' + exp['dataset'][0] 
    exp['save_weights'] = True
    exp['validation_iters'] = 1000
    exp['num_validation_evals'] = 200
    exp['shuffle_val'] = True  # Shuffle val data.
    exp['shuffle_train'] = True
    return exp


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('hGRU', reuse=reuse):
            net = resnet.model(
                trainable=True,
                num_classes=2,
                resnet_size=152)
            x = net.build(
                rgb=data_tensor,
                training=training)
    return x, x


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

