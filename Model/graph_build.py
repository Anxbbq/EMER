import tensorflow as tf
from Model.encoder import Encoder
from Model.decoder import CovCopyAttenGen
from Model.rating_predicting import RatingPrediction



class graph_build:
    def __init__(self, para_dict, is_training = True, mode_gen='ce_loss'):

        # inputs
        self.input_title = tf.placeholder(tf.int32, [None, para_dict['title_len']], name='input_title')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_y_tip = tf.placeholder(tf.int32, [None, para_dict['max_tip_len']], name="input_y_tip")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        
        self.embedding_tipp = tf.placeholder(tf.float32, [para_dict['tip_vocab_size'], para_dict['embedding_tip_size']])
        self.embedding_titp = tf.placeholder(tf.float32, [para_dict['title_vocab_size'], para_dict['embedding_title_size']])
        
        self.fine_tuning = False


        with tf.variable_scope('rating_prediction'):
            self.uidW = tf.get_variable("uidW", [para_dict['user_num'], para_dict['embedding_id_size']], dtype=tf.float32)
            self.iidW = tf.get_variable("iidW", [para_dict['item_num'], para_dict['embedding_id_size']], dtype=tf.float32)
            self.embeded_uid = tf.nn.embedding_lookup(self.uidW, self.input_uid)
            self.embeded_iid = tf.nn.embedding_lookup(self.iidW, self.input_iid)
            self.embeded_uid = tf.reshape(self.embeded_uid, [-1, para_dict['embedding_id_size']])
            self.embeded_iid = tf.reshape(self.embeded_iid, [-1, para_dict['embedding_id_size']])
            self.rp = RatingPrediction(para_dict)
            self.r, self.feas_rating, self.rating_loss, self.rmse, self.mae = self.rp.rating_prediction(self.embeded_uid, self.embeded_iid, self.input_y)


        with tf.variable_scope('title_infor'):
            self.W_title = tf.get_variable("W_title",[para_dict['title_vocab_size'], para_dict['embedding_title_size']], dtype=tf.float32)
            self.W_title = self.W_title.assign(self.embedding_titp)
            self.embeded_title = tf.nn.embedding_lookup(self.W_title, self.input_title)


        with tf.variable_scope("sentence_generation"):
            self.W_tip = tf.get_variable("W_tip", [para_dict['tip_vocab_size'], para_dict['embedding_tip_size']], dtype=tf.float32)
            self.uidW_tip = tf.get_variable("uidW_tip", [para_dict['user_num'], para_dict['embedding_tip_size']], dtype=tf.float32)
            self.iidW_tip = tf.get_variable("iidW_tip", [para_dict['item_num'], para_dict['embedding_tip_size']], dtype=tf.float32)
            self.W_tip = self.W_tip.assign(self.embedding_tipp)
            #self.iidW_tip = self.iidW_tip.assign(self.embedding_tipp)

            self.embeded_uid_tip = tf.nn.embedding_lookup(self.uidW_tip, self.input_uid)
            self.embeded_iid_tip = tf.nn.embedding_lookup(self.iidW_tip, self.input_iid)

            self.embeded_result = tf.concat([self.embeded_uid_tip, self.embeded_iid_tip, self.embeded_title], axis = 1)
            self.encoder_dim = para_dict['neighbor_vector_dim'] * 2 * 4
            self.encoder = Encoder(para_dict, self.embeded_result, self.encoder_dim, is_training)

            self.nodes_mask_title = tf.to_float(tf.not_equal(self.input_title, para_dict['title_vocab_size'] - 1))
            self.nodes_mask = tf.concat([tf.ones([para_dict['batch_size'], 2]), self.nodes_mask_title], axis = 1)
            self.loss_weights = tf.to_float(tf.not_equal(self.input_y_tip[:,1:], int(para_dict['tip_vocab_size'] - 1)))

            self.generator = CovCopyAttenGen(para_dict, is_training, self.W_tip)
            self.answer_inp = self.input_y_tip[:, : -1]
            self.answer_ref = self.input_y_tip[:, 1: ]
            self.g_loss, self.sampled_words, self.attn_dists = self.generator.train_mode(self. feas_rating, self.encoder_dim,
                                                          self.encoder.encoder_states, self.encoder.encoder_features, self.nodes_mask,
                                                          self.encoder.init_decoder_state, self.answer_inp, self.answer_ref, self.loss_weights,
                                                          mode_gen)

        with tf.variable_scope("op"):
            clipper = 50
            self.global_step = tf.Variable(0, trainable=False, name='Global_step')
            all_vars = tf.trainable_variables()
            trainable_vars = all_vars
            self.variable_names = [v.name for v in all_vars]
            
            #if self.fine_tuning == True:
            #    trainable_vars = [t for t in trainable_vars if not t.name.startswith('Model/rating_prediction')]
            
            self.learning_rate = tf.Variable(para_dict['learning_rate'], trainable=False, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars if v.get_shape().ndims > 1])
            self.loss = self.g_loss + self.rating_loss + para_dict['lambda_l2'] * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars))

















