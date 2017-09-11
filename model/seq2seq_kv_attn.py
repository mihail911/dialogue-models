import copy
import numpy as np
import tensorflow as tf

from custom_rnn_cell import CustomEmbeddingWrapper
from tensorflow.python import shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from util.tf_utils import batch_linear

"""
Seq2Seq model that also allows softly attending over a KB represented in a key
 value format
"""

linear = rnn_cell._linear

def _extract_argmax_and_embed(embedding, output_projection=None):
    """Get a loop_function that extracts the previous symbol and embeds it.

    :param embedding: embedding tensor for symbols.
    :param output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    :param update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

    Returns:
    A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        return emb_prev
    return loop_function


def kv_attention_decoder(cell,
                        decoder_inputs,
                        kb_inputs,
                        kb_mask_inputs,
                        initial_state,
                        attention_states,
                        num_decoder_symbols,
                        embedding_size,
                        output_size,
                        output_projection=None,
                        feed_previous=False,
                        attn_type="linear",
                        enc_attn=False,
                        enc_query=False,
                        scope=None,
                        dtype=None):
    """
    Run decoding which includes an attention over both the encoder states and the KB
    :param cell:
    :param encoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs)
    :param decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs)
    :param kb_inputs: Tensor containing KB to be used for decoding
    :param kb_col_inputs: Tensor containing col indices for batch of dialogues (batch_size, num_cols)
    :param kb_mask_inputs: Tensor containing KB masks to be used for zeroing out PAD embeddings in KB
    :param initial_state: Initial encoder state fed into the decoder
    :param attention_states: Embedded encoder attention states (batch_size, attn_length, attn_size)
    :param num_decoder_symbols: Vocab size for decoding
    :param embedding_size: Size of embedding vector
    :param output_size: Size of output vectors
    :param output_projection:
    :param feed_previous:
    :param scope:
    :param dtype:
    :return:
    """
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_decoder_symbols])

    with variable_scope.variable_scope(scope or "kb_attention_decoder",
                                       dtype=dtype) as scope:
        embedding = variable_scope.get_variable("embedding",
                                                [num_decoder_symbols,
                                                 embedding_size])
        loop_function = _extract_argmax_and_embed(
                embedding, output_projection) if feed_previous else None
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in
                   decoder_inputs]
        # Needed for reshaping.
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to
        # reshape before.
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        # Size of query vectors for attention.
        attention_vec_size = attn_size

        if attn_type == "linear" or attn_type == "two-mlp":
            k = variable_scope.get_variable("AttnW",
                                      [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k,
                                                 [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV", [attention_vec_size]))

        # Initialize mask embedding table
        np_mask = np.array([[0.]*embedding_size, [1.] * embedding_size])
        embedding_mask = variable_scope.get_variable("embedding_mask",
                            [2, embedding_size],
                            initializer=tf.constant_initializer(np_mask),
                            trainable=False)
        embedded_kb_mask_batch = tf.nn.embedding_lookup(embedding_mask,
                                                        kb_mask_inputs)
        # Mask for zeroing out attns over PAD tokens
        kb_attn_mask = tf.cast(kb_mask_inputs[:, :, 0, 0], tf.float32)
        # Embed kb
        embedded_kb_batch = tf.nn.embedding_lookup(embedding, kb_inputs)
        embedded_kb_batch = embedded_kb_batch * embedded_kb_mask_batch

        embedded_kb_batch = math_ops.reduce_sum(embedded_kb_batch, [3])
        # Split into value, type tensors
        num_triples = embedded_kb_batch.get_shape()[1].value

        embedded_kb_key = embedded_kb_batch[:, :, :2, :]
        # Summing head + relation
        embedded_kb_key = math_ops.reduce_sum(embedded_kb_key, [2])

        # Dim: (?, num_triples,)
        value_idx = kb_inputs[:, :, 3, 0]
        # Query will usually be of (batch_size, rnn_size)
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            # Results of attention reads will be stored here.
            ds = []
            # Will store masks over encoder context
            attn_masks = []
            # Store attention logits
            attn_logits = []
             # If the query is a tuple (LSTMStateTuple), flatten it.
            if nest.is_sequence(query):
                query_list = nest.flatten(query)
                # Check that ndims == 2 if specified.
                for q in query_list:
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            with variable_scope.variable_scope("Attention"):
                if attn_type == "linear":
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[0] * math_ops.tanh(hidden_features[0] + y), [2, 3])
                elif attn_type == "bilinear":
                    query = tf.tile(tf.expand_dims(query, 1), [
                        1, attn_length, 1])
                    query = batch_linear(query, attn_size, bias=True)
                    hid = tf.squeeze(hidden, [2])
                    s = tf.reduce_sum(tf.mul(query, hid), [2])
                else:
                    # Two layer MLP
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    layer1 = math_ops.tanh(hidden_features[0] + y)
                    k2 = variable_scope.get_variable("AttnW_1",
                                          [1, 1, attn_size, attention_vec_size])
                    layer2 = nn_ops.conv2d(layer1, k2, [1, 1, 1, 1], "SAME")
                    s = math_ops.reduce_sum(
                        v[0] * math_ops.tanh(layer2), [2, 3])

                a = nn_ops.softmax(s)
                attn_masks.append(a)
                attn_logits.append(s)
                # Now calculate the attention-weighted vector d.
                # Hidden is encoder hidden states
                d = math_ops.reduce_sum(
                  array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                  [1, 2])
                ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds, attn_masks, attn_logits


        def attention_kb_triple(query):
            """
            Compute attention over kb triples given decoder hidden state as a query
            :param query:
            :return:
            """
            # Expand dims so can concatenate with embedded_key
            with variable_scope.variable_scope("Attention_KB_Triple"):
                if attn_type == "two-mlp":
                    query = tf.expand_dims(query, [1])
                    with variable_scope.variable_scope("KB_key_W1"):
                        key_layer_1 = batch_linear(embedded_kb_key,
                                                   attention_vec_size,
                                                   bias=True)

                    with variable_scope.variable_scope("Query_W1"):
                        query_layer_1 = batch_linear(query, attention_vec_size,
                                                     bias=True)

                    layer_1 = math_ops.tanh(key_layer_1 + query_layer_1)
                    with variable_scope.variable_scope("KB_Query_W2"):
                        layer_2 = batch_linear(layer_1, attention_vec_size,
                                               bias=True)

                    layer_2 = math_ops.tanh(layer_2)
                    with variable_scope.variable_scope("KB_Query_W3"):
                        layer_3 = batch_linear(layer_2, 1, bias=True)

                    layer_3_logits = tf.squeeze(layer_3, [2])
                    layer_3 = nn_ops.softmax(layer_3_logits)

                    return layer_3, layer_3_logits
                elif attn_type == "linear":
                    query = tf.expand_dims(query, [1])
                    with variable_scope.variable_scope("KB_key_W1"):
                        key_layer_1 = batch_linear(embedded_kb_key,
                                                   attention_vec_size,
                                                   bias=True)

                    with variable_scope.variable_scope("Query_W1"):
                        query_layer_1 = batch_linear(query, attention_vec_size,
                                                     bias=True)

                    layer_1 = math_ops.tanh(key_layer_1 + query_layer_1)
                    with variable_scope.variable_scope("KB_Query_W2"):
                        layer_2 = batch_linear(layer_1, 1, bias=True)

                    layer_2_logits = tf.squeeze(layer_2, [2])
                    layer_2 = nn_ops.softmax(layer_2_logits)
                    return layer_2, layer_2_logits

        state = initial_state
        outputs = []
        switch_outputs = []
        attn_kb_outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)]
        first_indices = tf.tile(tf.expand_dims(tf.range(batch_size), dim=1),
                        [1, num_triples])
        # Use encoding of query
        if enc_query:
            encoder_q = array_ops.concat(1, [state.c, state.h])
            attn_kb, attn_kb_logits = attention_kb_triple(encoder_q)
        # Ensure the second shape of attention vectors is set.
        for a in attns:
            a.set_shape([None, attn_size])

        for i, inp in enumerate(emb_inp):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of
            # the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s"
                                 % inp.name)

            if enc_attn:
                # Use encoder attention as well
                x = linear([inp] + attns, input_size, True)
            else:
                x = linear([inp], input_size, True)

            # Run the RNN.
            cell_output, state = cell(x, state)
            # If the query is a tuple (LSTMStateTuple), flatten it.
            if nest.is_sequence(state):
                query_list = nest.flatten(state)
                # Check that ndims == 2 if specified.
                for q in query_list:
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                concat_state = array_ops.concat(1, query_list)

            if enc_attn:
                attns, attn_masks, attn_logits = attention(state)

            if not enc_query:
                attn_kb, attn_kb_logits = attention_kb_triple(concat_state)

            attn_kb_logits = attn_kb_logits * kb_attn_mask
            # Gather values from KB
            gather_indices = tf.pack([first_indices, value_idx], axis=2)
            updated_p = tf.scatter_nd(gather_indices, attn_kb_logits,
                                      [batch_size, num_decoder_symbols])
            attn_kb_outputs.append(attn_kb_logits)

            with variable_scope.variable_scope("AttnOutputProjection"):
                if enc_attn:
                    output = linear([cell_output] + attns, output_size, True)
                else:
                    output = linear([cell_output], output_size, True)
            # Simply add output logits and attn kb logits
            output = updated_p + output
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, attn_kb_outputs, switch_outputs


def embedding_kv_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                kb_inputs,
                                kb_mask_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                output_projection=None,
                                feed_previous=False,
                                attn_type="linear",
                                enc_attn=False,
                                use_types=False,
                                type_to_idx=None,
                                use_bidir=False,
                                seq_lengths=None,
                                enc_query=False,
                                dtype=None,
                                scope=None):
    """Embedding sequence-to-sequence model with attention over a KB.

    This model first embeds encoder_inputs by a newly created embedding
    (of shape [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs
    as well as an embedded KB.

    Warning: when output_projection is None, the size of the attention vectors
    and variables will be made proportional to num_decoder_symbols, can be large.

    Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    kb_inputs: Kbs for the given batch of dialogues
    kb_col_inputs: Column indices for given batch of kbs
    kb_mask_inputs: Kb masks for the given batch of dialogues
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_kb_attention_seq2seq".

    Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    if type_to_idx is not None:
        # Mapping from entity type to idx for augmenting encoder input
        num_entity_types = len(type_to_idx.keys())
        entity_encoding = np.zeros((num_entity_types, num_entity_types - 1),
                                   dtype=np.float32)
        for idx in range(num_entity_types - 1):
            entity_encoding[idx, idx] = 1.

    with variable_scope.variable_scope(
        scope or "embedding_kb_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        if use_types:
            print "Typed Encoder Inputs..."
            # Augment encoder inputs
            encoder_cell = CustomEmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size, entity_encoding=entity_encoding)
        else:
            print "Regular encoding..."
            # Just regular encoding
            encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)

        # Use bidirectional encoding
        if use_bidir:
            encoder_cell_backward = copy.deepcopy(encoder_cell)
            encoder_outputs, encoder_state_fw, encoder_state_bw =\
                rnn.bidirectional_rnn(encoder_cell, encoder_cell_backward,
                                      encoder_inputs, dtype=dtype,
                                      sequence_length=seq_lengths)
            combined_c = tf.concat(1, [encoder_state_fw.c, encoder_state_bw.c])
            combined_h = tf.concat(1, [encoder_state_fw.h, encoder_state_bw.h])
            encoder_state = rnn_cell.LSTMStateTuple(c=combined_c, h=combined_h)
        else:
            encoder_outputs, encoder_state = rnn.rnn(
                encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs
        # to put attention on.
        if use_bidir:
            top_states = [array_ops.reshape(e, [-1, 1, 2 * cell.output_size])
                      for e in encoder_outputs]
        else:
            top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]

        attention_states = array_ops.concat(1, top_states)
        if output_projection is None:
            if use_bidir:
                # Modify dimension of decoder rnn_size
                cell = rnn_cell.BasicLSTMCell(2 * cell.output_size,
                                              state_is_tuple=True)
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols
        else:
            output_size = cell.output_size

        if isinstance(feed_previous, bool):
            return kv_attention_decoder(
                            cell,
                            decoder_inputs,
                            kb_inputs,
                            kb_mask_inputs,
                            encoder_state,
                            attention_states,
                            num_decoder_symbols,
                            embedding_size=embedding_size,
                            output_size=output_size,
                            feed_previous=feed_previous,
                            attn_type=attn_type,
                            enc_attn=enc_attn,
                            enc_query=enc_query,
                            dtype=dtype)
