import numpy as np
import tensorflow as tf

from custom_seq2seq import embedding_attention_seq2seq


class Seq2Seq(object):
    """
    Seq2Seq model with other mechanisms like copying
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
                 copy=False,
                 attn_type="linear"
                 ):
        self.name = "seq2seq"
        self.rnn_size = rnn_size
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_attn = use_attn
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg = l2_reg
        self.copy = copy
        self.attn_type = attn_type
        # Note these stop symbols are the numerical indices in the vocab
        self.stop_symbols = stop_symbols
        self.do_decode = do_decode
        # Whether to tie encoder/decoder weights
        self.tied = tied
        self._build_model()

    def get_numerical_prediction_tf(self, outputs):
        """
        Return numerical predicted vocab from output/logits (batch_size x vocab_size).
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
            cell = Seq2Seq.cell_types[self.cell_type](self.rnn_size,
                                                      state_is_tuple=True)
        else:
            cell = Seq2Seq.cell_types[self.cell_type](self.rnn_size)

        # Apply dropout
        if self.dropout_keep_prob < 1.0:
            print "Applying dropout keep prob of ", self.dropout_keep_prob
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 input_keep_prob=self.dropout_keep_prob,
                                                 output_keep_prob=self.dropout_keep_prob)
        # Handle multilayer case
        if self.num_layers > 1:
            # Check just to be sure
            assert cell is not None
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
        for i in xrange(self.encoder_len):
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(self.decoder_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))
            self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="target{0}".format(i)))


    def update_feed_dict(self, feed_dict, encoder_inputs, decoder_inputs, targets=None, target_weights=None):
        """
        Update feed dict with data to provide the model
        :param feed_dict: Feed dict to store mapping from var to data
        :param encoder_inputs: Encoder data
        :param decoder_inputs: Decoder data
        :param targets: Targets data
        :param target_weights: Target weights data
        :return:
        """
        for i in range(self.encoder_len):
            feed_dict[self.encoder_inputs[i]] = encoder_inputs[i]
        for i in range(self.decoder_len + 1):
            feed_dict[self.decoder_inputs[i]] = decoder_inputs[i]
        if targets is not None:
            for i in range(self.decoder_len + 1):
                feed_dict[self.targets[i]] = targets[i]
        if target_weights is not None:
            for i in range(self.decoder_len + 1):
                feed_dict[self.target_weights[i]] = target_weights[i]

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
            if self.copy:
                print "Using attention of form ", self.attn_type, " with copy mechanism..."
            else:
                print "Using attention of form ", self.attn_type
            self.outputs, self.states, self.attn_outputs = embedding_attention_seq2seq(
              self.encoder_inputs,
              self.decoder_inputs,
              cell,
              num_encoder_symbols=self.vocab_size,
              num_decoder_symbols=self.vocab_size,
              embedding_size=self.embedding_size,
              output_projection=output_projection,
              feed_previous=self.do_decode,
              dtype=tf.float32,
              copy=self.copy,
              attn_type=self.attn_type)
        else:
            print "Using vanilla seq2seq..."
            self.outputs, self.state = tf.nn.seq2seq.embedding_rnn_seq2seq(
              self.encoder_inputs,
              self.decoder_inputs,
              cell,
              num_encoder_symbols=self.vocab_size,
              num_decoder_symbols=self.vocab_size,
              embedding_size=self.embedding_size,
              output_projection=output_projection,
              feed_previous=self.do_decode,
              dtype=tf.float32)
            self.attn_outputs = None

        # Compute loss -- averaged across batch + with l2 loss added
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Only get non-bias terms
        non_bias_vars = [v for v in trainable_vars if "Bias" not in v.name]
        l2_loss = tf.add_n([self.l2_reg * tf.nn.l2_loss(nb) for nb in non_bias_vars])

        # Compute loss -- averaged across batch
        self.total_loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, self.target_weights) + l2_loss
