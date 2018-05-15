import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, args, training=True):
    
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length])
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[args.batch_size])

        # Embedding
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        encoder_inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # inputs will have size [batch_size, seq_length, rnn_size]
    
        # Build RNN cell
        encoder_cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        encoder_cells = []
        for _ in range(args.num_layers):
		encoder_cells.append(encoder_cell_fn(args.rnn_size))
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)

        encoder_initial_state = encoder_cell.zero_state(args.batch_size, tf.float32)

        # Run Dynamic RNN
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn( \
        encoder_cell, encoder_inputs, self.sequence_lengths, encoder_initial_state)
        # encoder_outputs will have size [batch_size, seq_length, rnn_size]
        
        """
        # Run Bidirectional Dynamic RNN
        encoder_cell_fn_bw = tf.nn.rnn_cell.BasicLSTMCell
        encoder_cells_bw = []
        for _ in range(args.num_layers):
		encoder_cells_bw.append(encoder_cell_fn_bw(args.rnn_size))
        encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_bw)

        encoder_bw_initial_state = encoder_cell_bw.zero_state(args.batch_size, tf.float32)

        encoder_outputs_tuple, encoder_state = tf.nn.bidirectional_dynamic_rnn( \
        encoder_cell, encoder_cell_bw, encoder_inputs, sequence_length=self.sequence_lengths, \
        initial_state_fw=encoder_initial_state, initial_state_bw=encoder_bw_initial_state)
        # It returns a tuple instead of a single concatenated `Tensor`. 
        # If the concatenated one is preferred, the forward and backward outputs 
        # can be concatenated as `tf.concat(outputs, 2)`.
        encoder_outputs = tf.concat(encoder_outputs_tuple, 2)
        # encoder_outputs will have size [batch_size, seq_length, 2 * rnn_size]
        """

        # Build RNN cell
        # decoder_cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        # decoder_cell = decoder_cell_fn(args.rnn_size)

        # decoder_initial_state = decoder_cell.zero_state(args.batch_size, tf.float32)

        # decoder_inputs = encoder_outputs

        # Helper
        # helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, )
        
        weights = tf.get_variable("weights", [args.rnn_size, args.vocab_size])
        # weights = tf.get_variable("weights", [2 * args.rnn_size, args.vocab_size])
        bias = tf.get_variable("bias", [args.vocab_size])

        outputs = tf.reshape(encoder_outputs, [-1, args.rnn_size])
        # outputs = tf.reshape(encoder_outputs, [-1, 2 * args.rnn_size])

        self.logits = tf.split(tf.matmul(outputs, weights) + bias, args.batch_size, axis=0)
        self.probs = tf.nn.softmax(self.logits)

        if training:
            self.targets = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length])
            # need first to convert targets to one hot
            self.labels = tf.one_hot(self.targets, args.vocab_size)
            # labels will have size [batch_size, seq_length, vocab_size]
            
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.labels)
            self.cost = tf.reduce_mean(loss)

            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.cost)
