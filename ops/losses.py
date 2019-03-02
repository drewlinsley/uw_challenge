import os
import numpy as np
import tensorflow as tf


def derive_loss(labels, logits, loss_type, loss_weights):
    """Wrapper for multi-loss training."""
    if loss_weights is None:
        loss_weights = np.ones(len(loss_type))
    if isinstance(loss_type, list):
        loss_list = []
        for lt, lw in zip(loss_type, loss_weights):
            loss_list += [derive_loss_fun(labels=labels, logits=logits, loss_type=lt) * lw]
        return tf.add_n(loss_list), {k: v for k, v in zip(lt, loss_list)}
    else:
        return derive_loss_fun(labels=labels, logits=logits, loss_type=loss_type), None


def derive_loss_fun(labels, logits, loss_type):
    """Derive loss_type between labels and logits."""
    assert loss_type is not None, 'No loss_type declared'
    if loss_type == 'sparse_ce':  #  or loss_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=logits))
    elif loss_type == 'sparse_ce_image':
        logits = tf.cast(logits, tf.float32)
        label_shape = labels.get_shape().as_list()
        if label_shape[-1] > 1:
            raise RuntimeError('Label shape is %s.' % label_shape)
        labels = tf.squeeze(labels)
        labels = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'cce_image':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=logits,
                dim=-1))
    elif loss_type == 'bce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int64)
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)
    elif loss_type == 'l2':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.nn.l2_loss(tf.reshape(labels, [-1]) - logits)
    elif loss_type == 'mse':
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits))
    elif loss_type == 'mse_nn':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        mask = tf.cast(tf.greater_equal(labels, 0.), tf.float32)
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=mask))
    elif loss_type == 'mse_nn_unnorm':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        moments = np.load(os.path.join('moments', '%s.npz' % dataset))
        mu = moments['mean']
        sigma = moments['std']
        logits = sigma * logits + mu
        labels = sigma * labels + mu
        mask = tf.cast(tf.greater_equal(labels, 0.), tf.float32)
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=mask))
    elif loss_type == 'l2_image':
        logits = tf.cast(logits, tf.float32)
        return tf.nn.l2_loss(labels - logits)
    elif loss_type == 'pearson' or loss_type == 'correlation':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            REDUCE=tf.reduce_mean)
    elif loss_type == 'pearson_nn':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            mask=tf.greater_equal(labels, 0.),
            REDUCE=tf.reduce_mean)
    else:
        raise NotImplementedError(loss_type)


def derive_score(labels, logits, score_type, loss_type, dataset):
    """Derive score_type between labels and logits."""
    assert score_type is not None, 'No score_type declared'
    if score_type == 'sparse_ce' or score_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=logits))
    elif score_type == 'l2':
        logits = tf.cast(logits, tf.float32)
        return tf.nn.l2_loss(tf.reshape(labels, [-1]) - logits)
    elif score_type == 'mse':
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits))
    elif score_type == 'mse_nn':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        mask = tf.cast(tf.greater_equal(labels, 0.), tf.float32)
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=mask))
    elif score_type == 'mse_nn_unnorm':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        moments = np.load(os.path.join('moments', '%s.npz' % dataset))
        mu = moments['mean']
        sigma = moments['std']
        logits = sigma * logits + mu
        labels = sigma * labels + mu
        mask = tf.cast(tf.greater_equal(labels, 0.), tf.float32)
        return tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=mask))
    elif score_type == 'pearson' or score_type == 'correlation':
        logits = tf.cast(logits, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            REDUCE=tf.reduce_mean)
    elif score_type == 'accuracy':
        logits = tf.squeeze(logits)
        logits = tf.cast(logits, tf.float32)
        labels = tf.squeeze(labels)
        labs = tf.cast(labels, tf.float32)
        if loss_type == 'cce':
            preds = tf.argmax(logits, axis=-1)
            preds = tf.cast(preds, tf.float32)
        elif loss_type == 'cce_image':
            preds = tf.argmax(logits, axis=-1)
            preds = tf.cast(preds, tf.float32)
            labels = tf.argmax(labels, axis=-1)
            labels = tf.cast(labels, tf.float32)
        elif loss_type == 'bce':
            preds = tf.round(tf.sigmoid(logits))
        else:
            raise NotImplementedError('Cannot understand requested metric w/ loss.')
        return tf.reduce_mean(tf.cast(tf.equal(preds, labs), tf.float32))
    else:
        raise NotImplementedError(loss_type)


def pearson_dissimilarity(labels, logits, REDUCE, mask=None, eps_1=1e-4, eps_2=1e-12):
    """Calculate pearson diss. loss."""
    pred = logits
    x_shape = pred.get_shape().as_list()
    y_shape = labels.get_shape().as_list()
    if x_shape[-1] == 1 and len(x_shape) == 2:
        # If calculating score across exemplars
        pred = tf.squeeze(pred)
        x_shape = [x_shape[0]]
        labels = tf.squeeze(labels)
        y_shape = [y_shape[0]]

    if len(x_shape) > 2:
        # Reshape tensors
        x1_flat = tf.contrib.layers.flatten(pred)
    else:
        # Squeeze off singletons to make x1/x2 consistent
        x1_flat = tf.squeeze(pred)
    if len(y_shape) > 2:
        x2_flat = tf.contrib.layers.flatten(labels)
    else:
        x2_flat = tf.squeeze(labels)
    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + eps_1
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + eps_1

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(
                x1_flat_normed, x2_flat_normed),
            -1),
        count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x1_flat - x1_mean),
                -1),
            count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x2_flat - x2_mean),
                -1),
            count))
    corr = cov / (tf.multiply(x1_std, x2_std) + eps_2)
    if mask is not None:
        corr *= mask
    if REDUCE is not None:
        corr = REDUCE(corr)
    return 1 - corr

