

import tensorflow as tf
from tensorflow.python.ops import nn_ops


class Encoder:
    def __init__(self, para_dict, embedded_title, encoder_dim, is_training = True):
        self.para_dict = para_dict

        # calculate question representation
        for i in range(self.para_dict['num_encoder_layer']):
            with tf.variable_scope('layer-{}'.format(i)):
                context_lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.para_dict['neighbor_vector_dim'] * 4) #  * 4
                context_lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.para_dict['neighbor_vector_dim'] * 4) #
                if is_training:
                    context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - self.para_dict['dropout_rate']))
                    context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - self.para_dict['dropout_rate']))
                ((passage_context_representation_fw, passage_context_representation_bw),(passage_forward, passage_backward)) \
                    = tf.nn.bidirectional_dynamic_rnn(context_lstm_cell_fw, context_lstm_cell_bw, embedded_title, dtype=tf.float32)  # [batch_size, passage_len, context_lstm_dim]

            self.graph_hiddens_f = passage_context_representation_fw
            self.graph_hiddens_b = passage_context_representation_bw
        self.fc = passage_forward.c
        self.bc = passage_backward.c
        self.fh = passage_forward.h
        self.bh = passage_backward.h

        self.encoder_states = tf.concat([self.graph_hiddens_f, self.graph_hiddens_b], 2)

        # attention
        with tf.variable_scope("attention_decoder"):
            input_shape = tf.shape(self.encoder_states)
            batch_size = input_shape[0]
            passage_len = input_shape[1]
            encoder_features = tf.expand_dims(self.encoder_states, axis=2)  # now is shape [batch_size, passage_len, 1, encoder_dim]
            W_h = tf.get_variable("W_h", [1, 1, encoder_dim, self.para_dict['attention_vec_size']])
            encoder_features = nn_ops.conv2d(encoder_features, W_h, [1, 1, 1, 1],"SAME")  # [batch_size, passage_len, 1, attention_vec_size]
            self.encoder_features = tf.reshape(encoder_features,[batch_size, passage_len, self.para_dict['attention_vec_size']])


        # initializing decoder state
        with tf.variable_scope('initial_state_for_decoder'):
            w_reduce_c = tf.get_variable('w_reduce_c', [encoder_dim, para_dict['gen_hidden_size']], dtype=tf.float32) # + para_dict['n_latent']
            w_reduce_h = tf.get_variable('w_reduce_h', [encoder_dim, para_dict['gen_hidden_size']], dtype=tf.float32)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [para_dict['gen_hidden_size']], dtype=tf.float32)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [para_dict['gen_hidden_size']], dtype=tf.float32)

            old_c = tf.concat(values=[self.fc, self.bc], axis=1)
            old_h = tf.concat(values=[self.fh, self.bh], axis=1)
            new_c = tf.nn.tanh(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.tanh(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
            self.init_decoder_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)







