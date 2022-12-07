import tensorflow as tf
import numpy as np


class ReLUModel:
    def __init__(self, scope, input_dim, layers=2, dout=1, d_hidden=32):
        """
        ReLUModel here is used to construct task reward
        because we need to create resuable weights but not construct the actual computation

        :param scope:
        :param input_dim:
        :param layers:
        :param dout:
        :param d_hidden:
        """
        dX = input_dim
        self.layers = layers
        self.dout = dout
        self.d_hidden = d_hidden
        self.Ws = []
        self.bs = []
        with tf.variable_scope(scope):
            for i in range(layers):
                with tf.variable_scope("layer_%d" % i):
                    self.Ws.append(tf.get_variable('W', shape=(dX, d_hidden)))
                    self.bs.append(tf.get_variable('b', initializer=tf.constant(np.zeros(d_hidden).astype(np.float32))))
                dX = d_hidden
            with tf.variable_scope("layer_last"):
                self.Ws.append(tf.get_variable('W', shape=(d_hidden, dout)))
                self.bs.append(tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32))))
        self.grad_and_vars = []
        self.grad_map_vars = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
            ph = tf.placeholder(dtype=tf.float32, shape=var.shape)
            self.grad_and_vars.append((ph, var))
            self.grad_map_vars[var.name] = ph
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.step = self.optimizer.apply_gradients(self.grad_and_vars)

    def __call__(self, x):
        out = x
        for i in range(self.layers):
            out = tf.nn.relu(tf.matmul(out, self.Ws[i]) + self.bs[i])
        out = tf.matmul(out, self.Ws[self.layers]) + self.bs[self.layers]
        return out
