# from DataProcess.parameter_get import batches_train, batches_test, meta_infor, para_dict
from Model.graph_build import graph_build
import tensorflow as tf
import numpy as np
import pickle
import os
import re
import argparse
from collections import Counter, defaultdict
from rouge_metric import PyRouge
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
rouge = PyRouge(rouge_n=(1, 2), rouge_l=True,rouge_su=True, skip_gap=4)

def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--data_set', type=str,
                        help='Dataset Home_and_Kitchen, Movies_and_TV, Electronics, kindle, toys, cds, yelp',
                        required=True)
    # parser.add_argument('-n', '--split_num', type=str, help='Split_num 0-9', required=True)
    # parser.add_argument('-b', '--batch_size', type=int, help='batch_size', required=True)
    # parser.add_argument('-l', '--learning_rate', type=float, help='batch_size', required=True)
    # parser.add_argument('-y', '--learning_rate_decay', type=float, help='batch_size', required=True)
    args = parser.parse_args()
    return args


args = get_args()
CATEGORY = args.data_set
DIR = '../data/Yelp/%s_2/' % (CATEGORY)
TPS_DIR = 'data/%s' % (CATEGORY)

batches_train = pickle.load(open(os.path.join(DIR, CATEGORY + '.train'), 'rb'))
batches_test = pickle.load(open(os.path.join(DIR, CATEGORY + '.test'), 'rb'))
meta_infor = pickle.load(open(os.path.join(DIR, CATEGORY + '.meta_infor'), 'rb'))
para_dict = pickle.load(open(os.path.join(DIR, CATEGORY + '.para'), 'rb'))
para_dict['category_num'] = 20
para_dict['embedding_title_size'] = 300
para_dict['embedding_category_size'] = 300
para_dict['embedding_tip_size'] = 300
para_dict['embedding_id_size'] = 300

para_dict['n_latent'] = 400
para_dict['dropout_rate'] = 0.5
para_dict['neighbor_vector_dim'] = 300
para_dict['num_encoder_layer'] = 1
para_dict['gen_hidden_size'] = 512
para_dict['attention_vec_size'] = 512
para_dict['use_coverage'] = True
para_dict['cov_loss_wt'] = 0.1

para_dict['num_decoder_layer'] = 2
para_dict['pointer_gen'] = False

para_dict['learning_rate'] = 0.001
para_dict['learning_rate_decay'] = 0.9
para_dict['lambda_l2'] = 1e-4

para_dict['batch_size'] = 200
# para_dict['batch_size'] = args.batch_size
para_dict['num_epochs'] = 10
para_dict['save_path'] = TPS_DIR + '/save_path/'


def read_data(data_train, graph, is_test=False):
    uid, iid, y, y_tip, y_train_review = zip(*data_train)
    
    w2v_tip = np.load(DIR + 'w2v_tip.npy', encoding='bytes')
    w2v_tit = np.load(DIR + 'w2v_tit.npy', encoding='bytes')
    
    feed_dict = {graph.input_title: y_train_review,
                 graph.input_y: y,
                 graph.input_y_tip: y_tip,
                 graph.input_uid: uid,
                 graph.input_iid: iid,
                 graph.embedding_tipp:w2v_tip,
                 graph.embedding_titp:w2v_tit}
    return feed_dict, uid, iid, y


def load_model(saver, sess):
    if os.path.exists(para_dict['save_path'] + 'checkpoint'):
        modle_file = tf.train.latest_checkpoint(para_dict['save_path'])
        saver.restore(sess, modle_file)
        epoch_begin = int(''.join(filter(str.isdigit, modle_file))) + 1
    else:
        epoch_begin = 0
    return epoch_begin


def model_test(sess, epoch):
    print('validating...', len(batches_test), ltest)
    if not os.path.exists(TPS_DIR + '/test/epoch_' + str(epoch) + '/'):
         os.makedirs(TPS_DIR + '/test/epoch_' + str(epoch) + '/')
    f_epoch = open(TPS_DIR + '/test/epoch_' + str(epoch) + '/sys_val_rate_' + str(epoch) + '.txt', 'w',encoding = "utf-8")
    f_rouge = open(TPS_DIR + '/test/epoch_' + str(epoch) + '/rouge_score_' + str(epoch) + '.txt', 'w',encoding = "utf-8")
    rmse, mae = 0, 0
    docs = []
    refs = []
    for batch_num in range(ltest):
        start_index = batch_num * para_dict['batch_size']
        end_index = min((batch_num + 1) * para_dict['batch_size'], para_dict['test_length'])
        data_test = batches_test[start_index: end_index]
        feed_dict, uid, iid, y = read_data(data_test, valid_graph, is_test=True)
        syn_sent, rate, rmse1, mae1 = sess.run(
            (valid_graph.sampled_words, valid_graph.r, valid_graph.rmse, valid_graph.mae), feed_dict=feed_dict)
        rmse += rmse1
        mae += mae1

        val_sent = feed_dict[valid_graph.input_y_tip]
        syn_sent = [[para_dict['tip_idx_vocab'][xx] for xx in x if
                     xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx !=
                     para_dict['tip_vocab_size'] - 2] for x in syn_sent]

        val_sent = [[para_dict['tip_idx_vocab'][xx] for xx in x if
                     xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx !=
                     para_dict['tip_vocab_size'] - 2] for x in val_sent]
        syn_sents, val_sents = map(list, zip(
            *[[' '.join(sent), ' '.join(val_sent[num])] for num, sent in enumerate(syn_sent)]))

        # if not os.path.exists(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/' + str(batch_num) + '/'):
        #     os.makedirs(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/' + str(batch_num) + '/')
        # if not os.path.exists(TPS_DIR + '/test/model_summaries/' + str(epoch) + '/' + str(batch_num) + '/'):
        #     os.makedirs(TPS_DIR + '/test/model_summaries/' + str(epoch) + '/' + str(batch_num) + '/')

        for idx, sys_sent in enumerate(syn_sents):
            sys_sent = re.sub(' +', ' ', sys_sent)
            val_sent = re.sub(' +', ' ', val_sents[idx])
            f_epoch.write(str(uid[idx]) + '//' + str(iid[idx]) + '//' + sys_sent + '//' + val_sents[idx] + '//' + str(rate[idx][0]) + '//' + str(y[idx][0]) + '\n')

            if re.search(r'[a-z]+', val_sent):
                doc = [sys_sent]
                ref = [[val_sent]]
                docs.append(sys_sent)
                refs.append([val_sent])
                scores = rouge.evaluate(doc, ref)
                f_rouge.write(str(scores)+'\n')
    print('mse: %f, mae: %f' % (rmse / ltest, mae / ltest))
    score_all = rouge.evaluate(docs, refs)
    print(score_all)
    f_epoch.close()
    f_rouge.close()


if __name__ == '__main__':
    print("start running")
    init_scale = 0.01
    ltrain = int(para_dict['train_length'] / para_dict['batch_size'])
    ltest = int(para_dict['test_length'] / para_dict['batch_size'])

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                train_graph = graph_build(para_dict)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = graph_build(para_dict, is_training=False, mode_gen='greedy')

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        config = tf.ConfigProto(allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #print(train_graph.variable_names)

            epoch_begin = load_model(saver, sess)
            prev_loss_total, curr_loss_total, g_loss_total, rating_loss_total = 0.1, 0.1, 0.1, 0.1
            for epoch in range(epoch_begin, para_dict['num_epochs']):

                # train
                np.random.seed(2018)
                shuffle_indices = np.random.permutation(np.arange(para_dict['train_length']))
                shuffled_data = batches_train[shuffle_indices]

                if epoch > 2: 
                    model_test(sess, epoch)
                    train_graph.fine_tuning = True
                for batch_num in range(ltrain):
                    start_index = batch_num * para_dict['batch_size']
                    end_index = min((batch_num + 1) * para_dict['batch_size'], para_dict['train_length'])
                    data_train = shuffled_data[start_index: end_index]
                    feed_dict, _, _, _ = read_data(data_train, train_graph)
                    _, loss, g_loss, rating_loss = sess.run(
                        [train_graph.train_op, train_graph.loss, train_graph.g_loss, train_graph.rating_loss],
                        feed_dict=feed_dict)
                    curr_loss_total += loss
                    g_loss_total += g_loss
                    rating_loss_total += rating_loss
                    if batch_num % 10 == 0:
                        print('training_epoch: %d, batch_num: %d, loss: %f, g_loss: %f, rating_loss: %f'
                              % (epoch, batch_num, curr_loss_total / (batch_num + 1), g_loss_total / (batch_num + 1),
                                 rating_loss_total / (batch_num + 1)))

                current_learning_rate = sess.run(train_graph.learning_rate)
                current_learning_rate *= para_dict['learning_rate_decay']
                sess.run(train_graph.learning_rate.assign(current_learning_rate))

                print("Previous epoch loss: ", prev_loss_total)
                print("Current epoch loss: ", curr_loss_total)
                prev_loss_total, curr_loss_total, g_loss_total, rating_loss_total = curr_loss_total, 0, 0, 0
                saver.save(sess, para_dict['save_path'] + 'TextGAN_train.ckpt', global_step=epoch)






