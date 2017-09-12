import tensorflow as tf
import math


def xavier_weight_init():
    def _xavier_initializer(shape, name, **kwargs):

        val = math.sqrt(6. / sum(shape))
        out = tf.get_variable(shape = list(shape), dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-val, maxval= val, dtype=tf.float32),
                              trainable= True, name = name)
        return out
    return _xavier_initializer
