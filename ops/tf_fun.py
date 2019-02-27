"""Misc. functions for tensorflow model construction and training."""
import numpy as np
import tensorflow as tf
import json


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def count_parameters(var_list, keyword='symm', print_count=False):
    """Count the parameters in a tf model."""
    params = []
    for v in var_list:
        if keyword in v.name:  # TODO: ID vars w/ grad overrides
            count = np.maximum(np.prod(
                [x for x in v.get_shape().as_list()
                    if x > 1]), 1)
            count = (count / 2) + v.get_shape().as_list()[-1]
            params += [count]
        else:
            params += [
                np.maximum(
                    np.prod(
                        [x for x in v.get_shape().as_list()
                            if x > 1]), 1)]
    if print_count:
        param_list = [
            {v.name: [str(v.get_shape().as_list()), k]} for k, v in zip(
                params, var_list)]
        print json.dumps(param_list, indent=4)
        print 'Found %s variables' % len(param_list)
    return np.sum(params)


def check_shapes(scores, labels):
    """Check and fix the shapes of scores and labels."""
    if not isinstance(scores, list):
        if len(
                scores.get_shape()) != len(
                    labels.get_shape()):
            score_shape = scores.get_shape().as_list()
            label_shape = labels.get_shape().as_list()
            if len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                labels = tf.expand_dims(labels, axis=-1)
            elif len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                scores = tf.expand_dims(scores, axis=-1)
    return scores, labels


def bytes_feature(values):
    """Bytes features for writing TFRecords."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Int64 features for writing TFRecords."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Float features for writing TFRecords."""
    if isinstance(values, np.ndarray):
        values = [v for v in values]
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fixed_len_feature(length=[], dtype='int64'):
    """Features for reading TFRecords."""
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'int32':
        return tf.FixedLenFeature(length, tf.int32)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float' or dtype == 'float32':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')


def image_summaries(
        images,
        tag):
    """Wrapper for creating tensorboard image summaries.

    Parameters
    ----------
    images : tensor
    tag : str
    """
    im_shape = [int(x) for x in images.get_shape()]
    tag = '%s images' % tag
    if im_shape[-1] <= 3 and (
            len(im_shape) == 3 or len(im_shape) == 4):
        tf.summary.image(tag, images)
    elif im_shape[-1] <= 3 and len(im_shape) == 5:
        # Spatiotemporal image set
        res_ims = tf.reshape(
            images,
            [im_shape[0] * im_shape[1]] + im_shape[2:])
        tf.summary.image(tag, res_ims)
