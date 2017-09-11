from __future__ import absolute_import
from __future__ import division

import math
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs


class CustomEmbeddingWrapper(rnn_cell.RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes, embedding_size, use_pretrained_embeds=False,
               pretrained_embeddings=None, initializer=None, entity_encoding=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      use_pretrained_embeds: Whether to use pretrained embeddings
      pretrained_embeddings: Word embedding matrix if relevant (None if 'use_pretrained_embeds' is False)
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.
      entity_encoding: Embedding mat for entity encoding

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer
    self.use_pretrained_embeds = use_pretrained_embeds
    self.pretrained_embeddings = pretrained_embeddings
    self.entity_encoding = entity_encoding

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with vs.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with ops.device("/cpu:0"):
        if self._initializer:
          initializer = self._initializer
        elif vs.get_variable_scope().initializer:
          initializer = vs.get_variable_scope().initializer
        else:
          # Default initializer for embeddings should have variance=1.
          sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
          initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

        if type(state) is tuple:
          data_type = state[0].dtype
        else:
          data_type = state.dtype

        if not self.use_pretrained_embeds:
          embedding = vs.get_variable(
              "embedding", [self._embedding_classes, self._embedding_size],
              initializer=initializer,
              dtype=data_type)

          type_embedding = vs.get_variable(
              "type_embedding", [self.entity_encoding.shape[0], self.entity_encoding.shape[1]],
              initializer=tf.constant_initializer(self.entity_encoding), trainable=False,
              dtype=data_type)
        else:
          print "Initializing pretrained embeddings..."
          # Initialize embedding matrix with pretrained embeddings
          embedding = vs.get_variable(
              "embedding", [self._embedding_classes, self._embedding_size],
              initializer=tf.constant_initializer(self.pretrained_embeddings),
              dtype=data_type)

        # Each input is [token_1, token_2, ...], [type_1, type_2,...]
        if inputs.get_shape().ndims != 1:
          word_indices = inputs[:, 0]
          type_indices = inputs[:, 1]
          embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(word_indices, [-1]))
          type_embedded = embedding_ops.embedding_lookup(
            type_embedding, array_ops.reshape(type_indices, [-1]))
          # Concatenate word embeddings and type embeddings along dim 1
          embedded = tf.concat(1, [embedded, type_embedded])
        else:
          embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(inputs, [-1]))

    return self._cell(embedded, state)