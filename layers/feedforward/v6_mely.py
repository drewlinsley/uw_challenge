import warnings
import numpy as np
import tensorflow as tf
import ffn.training.models.prc.initialization
from ffn.training.models.prc.pooling import max_pool3d
import ffn.training.models.prc.gradients

"""
HGRU MODEL IN MELY STYLE:
   1. Both h1 and h2 are recurrent
   2. Additive/multiplicative downstream and purely additive upstream
   3. ReLU activation at both X(h1) and Y(h2). Alpha and mu are strictly positive to enforce inhibition (w can be negative though)
ADDITIONAL TRICKS
   1. h1 and h2 update regulated by mix (instead of time constant and decay)
   2. Use batchnorm (not in original mely but in hGRU) with additive bias.
   3. Optional Fan-out in H1.
"""

class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            var_scope,
            in_k,
            fanout_k,
            h2_k,
            filter_siz,
            use_3d=False,
            symmetric_weights=True,
            bn_reuse=False,
            train=True,
            dtype=tf.bfloat16):

        # Structural
        self.train = train
        self.var_scope = var_scope
        self.bn_reuse=bn_reuse
        self.symmetric_weights= symmetric_weights
        self.dtype = dtype

        # RF Hyperparams
        self.use_3d = use_3d
        self.in_k = in_k
        self.fanout_k = fanout_k
        self.h1_k = fanout_k*in_k
        self.h2_k = h2_k
        self.h_siz = filter_siz

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):
        local_shape = [1, 1] if not self.use_3d else [1, 1, 1]
        self.bn_param_initializer = {
                            'moving_mean': tf.constant_initializer(0., dtype=self.dtype),
                            'moving_variance': tf.constant_initializer(1., dtype=self.dtype),
                            'beta': tf.constant_initializer(0., dtype=self.dtype),
                            'gamma': tf.constant_initializer(0.1, dtype=self.dtype)
        }
        with tf.variable_scope(self.var_scope):
            with tf.variable_scope('f1'):
                tf.get_variable(
                    name='w',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=self.h_siz + [self.h1_k, self.h2_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='alpha',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=[1] + local_shape + [1],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='mu',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=[1] + local_shape + [self.h1_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            with tf.variable_scope('h1_mix'):
                tf.get_variable(
                    name='w0',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=local_shape + [self.h2_k, self.h2_k*2],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='w1',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=local_shape + [self.h2_k*2, self.h1_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            with tf.variable_scope('h2_mix'):
                tf.get_variable(
                    name='w0',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=local_shape + [self.h1_k, self.h2_k*2],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='w1',
                    dtype=self.dtype,
                    initializer=ffn.training.models.prc.initialization.xavier_initializer(
                        shape=local_shape + [self.h2_k*2, self.h2_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            if self.bn_reuse:
                # Make the batchnorm variables
                scopes = ['f1/bn', 'f2/bn', 'h1_mix/bn0', 'h1_mix/bn1', 'h2_mix/bn0', 'h2_mix/bn1']
                shapes = [h1_k, h2_k, h2_k*2, h1_k, h2k*2, h2*k]
                bn_vars = ['moving_mean', 'moving_variance', 'beta', 'gamma']
                for (scp,shp) in zip(scopes, shapes):
                    with tf.variable_scope(scp):
                        for v in bn_vars:
                            tf.get_variable(
                                trainable=self.trainable,
                                name=v,
                                dtype=self.dtype,
                                shape=[shp],
                                initializer=self.bn_param_initializer)

    def symmetrize_weights(self, w):
        """Apply symmetric weight sharing."""
        if self.use_3d:
            conv_w_flipped = tf.reverse(w, [0, 1, 2])
        else:
            conv_w_flipped = tf.reverse(w, [0, 1])
        conv_w_symm = 0.5 * (conv_w_flipped + w)
        return conv_w_symm

    def conv_op(
            self,
            data,
            weights,
            strides=None,
            deconv_size=None,
            symmetric_weights=False,
            padding=None):
        """3D convolutions for hgru."""
        if padding is None:
            padding='SAME'
        if strides is None:
            if self.use_3d:
                strides = [1, 1, 1, 1, 1]
            else:
                strides = [1, 1, 1, 1]
        if symmetric_weights:
            weights = self.symmetrize_weights(weights)
        if deconv_size is None:
            if self.use_3d:
                activities = tf.nn.conv3d(
                                    data,
                                    weights,
                                    strides=strides,
                                    padding=padding)
            else:
                activities = tf.nn.conv2d(
                                    data,
                                    weights,
                                    strides=strides,
                                    padding=padding)
        else:
            if self.use_3d:
                activities = tf.nn.conv3d_transpose(
                                    data,
                                    weights,
                                    output_shape=deconv_size,
                                    strides=strides,
                                    padding=padding)
            else:
                activities = tf.nn.conv2d_transpose(
                                    data,
                                    weights,
                                    output_shape=deconv_size,
                                    strides=strides,
                                    padding=padding)
        return activities

    def gate(self, activities, weights_list, var_scope):
        for i, weights in enumerate(weights_list):
            activities = self.conv_op(
                                activities,
                                weights)
            if self.bn_reuse:
                with tf.variable_scope(
                        '%s/bn%i' % (var_scope, i),
                        reuse=tf.AUTO_REUSE) as scope:
                    activities = tf.contrib.layers.batch_norm(
                        inputs=activities,
                        scale=True,
                        center=True,
                        fused=True,
                        renorm=False,
                        reuse=tf.AUTO_REUSE,
                        scope=scope,
                        is_training=self.train)
            else:
                activities = tf.contrib.layers.batch_norm(
                    inputs=activities,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.bn_param_initializer,
                    is_training=self.train)
            if i < (len(weights_list)-1):
                activities = tf.square(activities)
            else:
                gate = tf.nn.sigmoid(activities)
        return gate

    def f1(self, h2, h1, ff_replicated, weights, combine_coeffs, var_scope, symmetric_weights=False):
        # Run f1 which combines downstream (deconvolution) activity with upstream (fan-out) activity
        downstream = self.conv_op(h2,
                             weights,
                             strides=None,
                             deconv_size=ff_replicated.get_shape().as_list(),
                             symmetric_weights=symmetric_weights,
                             padding=None)

        if self.bn_reuse:
            with tf.variable_scope(
                    '%s/bn' % (var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                downstream = tf.contrib.layers.batch_norm(
                    inputs=downstream,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    is_training=self.train)
        else:
            downstream = tf.contrib.layers.batch_norm(
                inputs=downstream,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                is_training=self.train)
        h1_raw = ff_replicated - tf.nn.relu(combine_coeffs[0] * downstream * h1 + combine_coeffs[1] * downstream)
        return tf.nn.relu(h1_raw)

    def f2(self, h1, weights, var_scope, symmetric_weights=False):
        h2_raw = self.conv_op(h1,
                         weights,
                         strides=None,
                         deconv_size=None,
                         symmetric_weights=symmetric_weights,
                         padding=None)
        if self.bn_reuse:
            with tf.variable_scope(
                    '%s/bn' % (var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                h2_raw = tf.contrib.layers.batch_norm(
                    inputs=h2_raw,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    is_training=self.train)
        else:
            h2_raw = tf.contrib.layers.batch_norm(
                inputs=h2_raw,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                is_training=self.train)
        return tf.nn.relu(h2_raw)

    def run(self, x, h1, h2):
        """hGRU body."""
        # Compute H1
        current_scope = self.var_scope + '/f1'
        with tf.variable_scope(current_scope, reuse=True):
            w = tf.get_variable("w")
            mu = tf.square(tf.get_variable("mu"))
            alpha = tf.square(tf.get_variable("alpha")) ### strict positivity on mu and alpha
        x_replicated = tf.tile(x, multiples=[1] * (len(x.get_shape().as_list()) - 1) + [self.fanout_k])
        h1_candidate = self.f1(h2, h1, x_replicated, w, [mu, alpha], current_scope, symmetric_weights=self.symmetric_weights)
        # Mix H1
        current_scope = self.var_scope + '/h1_mix'
        with tf.variable_scope(current_scope, reuse=True):
            w0 = tf.get_variable("w0")
            w1 = tf.get_variable("w1")
        h1m = self.gate(h2, [w0, w1], current_scope)
        h1 = h1m*h1_candidate + (1-h1m)*h1
        # Compute H2 Candidate (reusing W from f1)
        current_scope = self.var_scope + '/f2'
        h2_candidate = self.f2(h1, w, current_scope, symmetric_weights=self.symmetric_weights)
        # Mix H2
        current_scope = self.var_scope + '/h2_mix'
        with tf.variable_scope(current_scope, reuse=True):
            w0 = tf.get_variable("w0")
            w1 = tf.get_variable("w1")
        h2m = self.gate(h1, [w0, w1], current_scope)
        h2 = h2m*h2_candidate + (1-h2m)*h2

        return h1, h2