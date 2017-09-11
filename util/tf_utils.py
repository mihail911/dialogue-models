import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest

linear = rnn_cell._linear


def batch_linear(args, output_size, bias):
    """
    Apply linear map to a batch of matrices.
    :param args: a 3D Tensor or a list of 3D, batch x n x m, Tensors.
    :param output_size:
    :param bias:
    """
    if not nest.is_sequence(args):
        args = [args]
    batch_size = args[0].get_shape().as_list()[0] or tf.shape(args[0])[0]
    flat_args = []
    for arg in args:
        m = arg.get_shape().as_list()[2]
        if not m:
            raise ValueError('batch_linear expects shape[2] of arguments: %s' % str(m))
        flat_args.append(tf.reshape(arg, [-1, m]))
    flat_output = linear(flat_args, output_size, bias)
    output = tf.reshape(flat_output, [batch_size, -1, output_size])
    return output


def add_gradient_noise(t, stddev=1e-3):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    :param t: Input tensor (should be a gradient)
    :param stddev: Stddev of gaussian sampling from for noise
    """
    t = tf.convert_to_tensor(t, name="t")
    gn = tf.random_normal(tf.shape(t), stddev=stddev)
    return tf.add(t, gn)