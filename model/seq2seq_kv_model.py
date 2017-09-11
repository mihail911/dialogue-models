import numpy as np
import tensorflow as tf

from seq2seq_kv_attn import embedding_kv_attention_seq2seq


class Seq2SeqKV(object):
    """
    Seq2Seq with KV attention model
    """
    cell_types = {'gru': tf.nn.rnn_cell.GRUCell,
                  'rnn': tf.nn.rnn_cell.BasicRNNCell,
                  'lstm': tf.nn.rnn_cell.BasicLSTMCell}

    def __init__(self, rnn_size,
                 vocab_size,
                 encoder_len,
                 decoder_len,
                 batch_size,
                 stop_symbols,
                 cell_type='lstm',
                 num_layers=1,
                 embedding_size=100,
                 tied=False,
                 dropout_keep_prob=1.0,
                 l2_reg=0.001,
                 use_attn=False,
                 do_decode=False,
                 attn_type="linear",
                 enc_attn=False,
                 use_types=False,
                 type_to_idx=None,
                 use_bidir=False,
                 enc_query=False):
        self.name = "seq2seq-kv"
        self.rnn_size = rnn_size
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_attn = use_attn
        self.enc_attn = enc_attn
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg = l2_reg
        self.attn_type = attn_type
        self.use_types = use_types
        # Entity type to idx
        self.type_to_idx = type_to_idx
        self.use_bidir = use_bidir
        self.enc_query = enc_query

        # Note these stop symbols are the numerical indices in the vocab
        self.stop_symbols = stop_symbols
        self.do_decode = do_decode
        # Whether to tie encoder/decoder weights
        self.tied = tied
        self._build_model()


    def get_numerical_prediction_tf(self, outputs):
        """
        Return numerical predicted vocab from output/logits
        (batch_size x vocab_size).
        :param outputs: RNN outputs
        :return Predictions for unrolled RNN
        """
        # batch_size x time_step
        preds = np.argmax(outputs, axis=1)
        return preds


    def _set_cell_type(self):
        """
        Set and return the recurrent cell type which will be used for the model
        :return:
        """
        # Casework necessary because of tensorflow convention on
        # how to return lstm hidden/output state
        if self.cell_type == 'lstm':
            cell = Seq2SeqKV.cell_types[self.cell_type](self.rnn_size,
                                                        state_is_tuple=True)
        else:
            cell = Seq2SeqKV.cell_types[self.cell_type](self.rnn_size)
        # Apply dropout
        if self.dropout_keep_prob < 1.0:
            print "Applying dropout keep prob of ", self.dropout_keep_prob
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                    input_keep_prob=self.dropout_keep_prob,
                                    output_keep_prob=self.dropout_keep_prob)
        # Handle multilayer case
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

        return cell


    def _build_inputs(self):
        """
        Set up the input variables for model
        :return:
        """
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.targets = []
        self.kb_inputs = []
        for i in xrange(self.encoder_len):
            if self.use_types:
                self.encoder_inputs.append(tf.placeholder(tf.int32,
                                                shape=[None, 2],
                                                name="encoder{0}".format(i)))
            else:
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))

        for i in xrange(self.decoder_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))
            self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="target{0}".format(i)))

        # Can change these dims appropriately based on dataset
        self.kb_inputs = tf.placeholder(tf.int32, shape=[None, 200, 5, 2], name="kb")
        self.kb_mask_inputs = tf.placeholder(tf.int32, shape=[None, 200, 5, 2],
                                             name="kb_mask")
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
                                          name="seq_lengths")


    def update_feed_dict(self, feed_dict, encoder_inputs, decoder_inputs,
                         kb_inputs, kb_mask_inputs, targets=None,
                         target_weights=None, seq_lengths=None):
        """
        Update feed dict with data to provide the model. All inputs are
        appropriately-shaped numpy arrays
        :param feed_dict: Feed dict to populate
        :param encoder_inputs: Encoder data
        :param decoder_inputs: Decoder data
        :param kb_inputs: Kb data
        :param kb_mask: Kb masks
        :param targets: Targets data
        :param target_weights: Target weights data
        :param seq_lengths: Seq lengths data
        :return:
        """
        for i in range(self.encoder_len):
            feed_dict[self.encoder_inputs[i]] = encoder_inputs[i]
        for i in range(self.decoder_len + 1):
            feed_dict[self.decoder_inputs[i]] = decoder_inputs[i]

        feed_dict[self.kb_inputs] = kb_inputs
        feed_dict[self.kb_mask_inputs] = kb_mask_inputs

        if targets is not None:
            for i in range(self.decoder_len + 1):
                feed_dict[self.targets[i]] = targets[i]
        if target_weights is not None:
            for i in range(self.decoder_len + 1):
                feed_dict[self.target_weights[i]] = target_weights[i]

        # Use variable length sequences if bidirectional encoding
        if self.use_bidir and seq_lengths is not None:
            feed_dict[self.seq_lengths] = seq_lengths


    def _build_model(self):
        """
        Builds a model either for training or testing
        :return:
        """
        cell = self._set_cell_type()
        self._build_inputs()
        output_projection = None
        print "Embedding size: ", self.embedding_size

        if self.use_attn:
            print "Using attention over encoder + kb... of type ", \
                self.attn_type
            # Also returns attn weights over encoder, kb cols, and kb rows
            self.outputs, self.attn_kb_weights, self.attn_switch_outputs = \
                embedding_kv_attention_seq2seq(
              self.encoder_inputs,
              self.decoder_inputs,
              self.kb_inputs,
              self.kb_mask_inputs,
              cell,
              num_encoder_symbols=self.vocab_size,
              num_decoder_symbols=self.vocab_size,
              embedding_size=self.embedding_size,
              output_projection=output_projection,
              feed_previous=self.do_decode,
              attn_type=self.attn_type,
              enc_attn=self.enc_attn,
              use_types=self.use_types,
              type_to_idx=self.type_to_idx,
              use_bidir=self.use_bidir,
              seq_lengths=self.seq_lengths,
              enc_query=self.enc_query,
              dtype=tf.float32)

        # Compute loss -- averaged across batch + with l2 loss added
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Only get non-bias terms
        non_bias_vars = [v for v in trainable_vars if "Bias" not in v.name]
        l2_loss = tf.add_n([self.l2_reg * tf.nn.l2_loss(nb) for nb in non_bias_vars])
        self.total_loss = tf.nn.seq2seq.sequence_loss(self.outputs,
                                                      self.targets,
                                                      self.target_weights) \
                                                      + l2_loss
