import warnings
import numpy as np
import tensorflow as tf
import ops.initialization
import ops.gradients

"""
HGRU MODEL IN LN STYLE:
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
            symmetric_weights=False,
            bn_reuse=False,
            train=True,
            dtype=tf.bfloat16,
            swap_mix_sources=False,
            swap_gate_sources=False,
            turn_off_gates=False,
            featurewise_control=False,
            no_relu_h1=False):

        # Structural
        self.train = train
        self.var_scope = var_scope
        self.bn_reuse =bn_reuse
        self.symmetric_weights = symmetric_weights
        self.dtype = dtype

        # RF Hyperparams
        self.use_3d = use_3d
        self.in_k = in_k
        self.fanout_k = fanout_k
        if fanout_k <= 1:
            raise ValueError('fanout of leq 1 doesn not make sense')
        self.h1_k = fanout_k * in_k
        self.h2_k = h2_k
        self.h_siz = filter_siz

        # Optional lesions
        self.swap_mix_sources = swap_mix_sources
        self.swap_gate_sources = swap_gate_sources
        self.turn_off_gates = turn_off_gates
        self.featurewise_control = featurewise_control
        self.no_relu_h1=no_relu_h1

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
                    initializer=ops.initialization.xavier_initializer(
                        shape=self.h_siz + [self.h1_k, self.h2_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            with tf.variable_scope('f2'):
                tf.get_variable(
                    name='alpha',
                    dtype=self.dtype,
                    shape=[1]+local_shape+[1],
                    initializer=tf.constant_initializer(1.0, dtype=self.dtype),
                    trainable=self.train)
                tf.get_variable(
                    name='beta',
                    dtype=self.dtype,
                    shape=[1] + local_shape + [1],
                    initializer=tf.constant_initializer(1.0, dtype=self.dtype),
                    trainable=self.train)
            with tf.variable_scope('h1_mix'):
                if self.swap_mix_sources:
                    source_k = self.h2_k
                else:
                    source_k = self.h1_k
                tf.get_variable(
                    name='w0',
                    dtype=self.dtype,
                    initializer=ops.initialization.xavier_initializer(
                        shape=local_shape + [source_k, source_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='w1',
                    dtype=self.dtype,
                    initializer=ops.initialization.xavier_initializer(
                        shape=local_shape + [source_k, (1 if self.featurewise_control else self.h1_k)],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            with tf.variable_scope('h2_mix'):
                if self.swap_mix_sources:
                    source_k = self.h1_k
                else:
                    source_k = self.h2_k
                tf.get_variable(
                    name='w0',
                    dtype=self.dtype,
                    initializer=ops.initialization.xavier_initializer(
                        shape=local_shape + [source_k, source_k],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
                tf.get_variable(
                    name='w1',
                    dtype=self.dtype,
                    initializer=ops.initialization.xavier_initializer(
                        shape=local_shape + [source_k, (1 if self.featurewise_control else self.h2_k)],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            if not self.turn_off_gates:
                with tf.variable_scope('g1'):
                    if self.swap_gate_sources:
                        source_k = self.h2_k
                    else:
                        source_k = self.h1_k
                    tf.get_variable(
                        name='w0',
                        dtype=self.dtype,
                        initializer=ops.initialization.xavier_initializer(
                            shape=local_shape + [source_k, source_k],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=self.train)
                    tf.get_variable(
                        name='w1',
                        dtype=self.dtype,
                        initializer=ops.initialization.xavier_initializer(
                            shape=local_shape + [source_k, (1 if self.featurewise_control else self.h2_k)],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=self.train)
                with tf.variable_scope('g2'):
                    if self.swap_gate_sources:
                        source_k = self.h1_k
                    else:
                        source_k = self.h2_k
                    tf.get_variable(
                        name='w0',
                        dtype=self.dtype,
                        initializer=ops.initialization.xavier_initializer(
                            shape=local_shape + [source_k, source_k],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=self.train)
                    tf.get_variable(
                        name='w1',
                        dtype=self.dtype,
                        initializer=ops.initialization.xavier_initializer(
                            shape=local_shape + [source_k, (1 if self.featurewise_control else self.in_k)],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=self.train)
            if self.bn_reuse:
                # Make the batchnorm variables
                scopes = ['f1/bn', 'f2/bn', 'h1_mix/bn0', 'h1_mix/bn1', 'h2_mix/bn0', 'h2_mix/bn1']
                shapes = [self.h1_k, self.h2_k, self.h2_k*2, self.h1_k, self.h2k*2, self.h2*k]
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

    # def capsulewise_sum(self, x, order='l2'):


    def f1(self, h2, x, weights, var_scope, symmetric_weights=False):
        # Run f1 which combines downstream (deconvolution) activity with upstream (fan-out) activity
        x_shape = x.get_shape().as_list()
        downstream = self.conv_op(h2,weights,
                                 strides=None,
                                 deconv_size=x_shape[:-1] + [x_shape[-1]*self.fanout_k],
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
        x_norm = tf.reduce_sum(tf.square(x), axis=3, keep_dims=True) + 0.1 * x_shape[-1]
        x_replicated_normalized = tf.tile(x/x_norm, multiples=[1] * (len(x_shape) - 1) + [self.fanout_k])
        h1_raw = x_replicated_normalized * downstream
        h1_sum = tf.reduce_sum(tf.reshape(h1_raw, x_shape + [self.fanout_k]), axis=(4 if self.use_3d else 3), keep_dims=True) #b, h, w, 1, fanout_k
        return tf.nn.relu(h1_raw), h1_sum

    def f2(self, h1, h1_sum, h2, x, weights, combine_params, var_scope, symmetric_weights=False):
        x_coeff, h2_coeff = combine_params
        x_shape = x.get_shape().as_list()
        h1_shape = h1.get_shape().as_list()

        if self.no_relu_h1:
            h1_relu = h1
        else:
            h1_relu = tf.nn.relu(h1)
        h1_weights = tf.reshape(tf.tile(tf.nn.softmax(h1_sum, dim=-1), [1]*(4 if self.use_3d else 3) + [self.in_k, 1]), h1_shape)
        h1_weighted = h1_relu*h1_weights
        x_weighted = tf.tile(x, multiples=[1] * (len(x_shape) - 1) + [self.fanout_k])
        if self.h1_k % self.h2_k >0:
            raise ValueError('h1_k is not an integer multiple of h2_k')
        h2_weighted = tf.tile(h2, multiples=[1] * (len(x_shape) - 1) + [self.h1_k/self.h2_k])
        combined = h1_weighted + tf.square(x_coeff)*x_weighted + tf.square(h2_coeff)*h2_weighted
        h2_raw = self.conv_op(combined,
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
        h1_candidate, h1_sum = self.f1(h2, x, w, current_scope, symmetric_weights=self.symmetric_weights)
        # Mix H1
        current_scope = self.var_scope + '/h1_mix'
        with tf.variable_scope(current_scope, reuse=True):
            w0 = tf.get_variable("w0")
            w1 = tf.get_variable("w1")
        h1m = self.gate(h1, [w0, w1], current_scope)
        h1 = h1m*h1_candidate + (1-h1m)*h1
        # gate x and h2
        if not self.turn_off_gates:
            current_scope = self.var_scope + '/g1'
            with tf.variable_scope(current_scope, reuse=True):
                w0 = tf.get_variable("w0")
                w1 = tf.get_variable("w1")
            g1 = self.gate(h1, [w0, w1], current_scope)
            h2_gated = h2*g1
            current_scope = self.var_scope + '/g2'
            with tf.variable_scope(current_scope, reuse=True):
                w0 = tf.get_variable("w0")
                w1 = tf.get_variable("w1")
            g2 = self.gate(h2, [w0, w1], current_scope)
            x_gated = x*g2
        else:
            h2_gated = h2
            x_gated = x
        # Compute H2 Candidate (reusing W from f1)
        current_scope = self.var_scope + '/f2'
        with tf.variable_scope(current_scope, reuse=True):
            alpha = tf.get_variable("alpha")
            beta = tf.get_variable("beta")
        h2_candidate = self.f2(h1, h1_sum, h2_gated, x_gated, w, [alpha, beta], current_scope, symmetric_weights=self.symmetric_weights)
        # Mix H2
        current_scope = self.var_scope + '/h2_mix'
        with tf.variable_scope(current_scope, reuse=True):
            w0 = tf.get_variable("w0")
            w1 = tf.get_variable("w1")
        h2m = self.gate(h2, [w0, w1], current_scope)
        h2 = h2m*h2_candidate + (1-h2m)*h2

        return h1, h2
