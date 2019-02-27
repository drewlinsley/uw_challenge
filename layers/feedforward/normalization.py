import tensorflow as tf


def batch(
        bottom,
        name,
        scale=True,
        center=True,
        fused=False,
        renorm=False,
        data_format='NHWC',
        dtype=tf.float32,
        reuse=False,
        training=True):
    """Wrapper for layers batchnorm."""
    if data_format == 'NHWC' or data_format == 'channels_last':
        axis = -1
    elif data_format == 'NCHW' or data_format == 'channels_first':
        axis = 1
    else:
        raise NotImplementedError(data_format)
    return tf.layers.batch_normalization(
        inputs=bottom,
        name=name,
        scale=scale,
        center=center,
        beta_initializer=tf.zeros_initializer(dtype=dtype),
        gamma_initializer=tf.ones_initializer(dtype=dtype),
        moving_mean_initializer=tf.zeros_initializer(dtype=dtype),
        moving_variance_initializer=tf.ones_initializer(dtype=dtype),
        fused=fused,
        renorm=renorm,
        reuse=reuse,
        axis=axis,
        training=training)


def batch_contrib(
        bottom,
        name,
        scale=True,
        center=True,
        fused=None,
        renorm=False,
        dtype=tf.float32,
        reuse=False,
        training=True):
    """Wrapper for contrib layers batchnorm."""
    param_initializer = {
        'moving_mean': tf.constant_initializer(0.),
        'moving_variance': tf.constant_initializer(1.),
        'gamma': tf.constant_initializer(0.1)
    }
    return tf.contrib.layers.batch_norm(
        inputs=bottom,
        scale=scale,
        center=center,
        param_initializers=param_initializer,
        updates_collections=None,
        fused=fused,
        renorm=renorm,
        is_training=training)
