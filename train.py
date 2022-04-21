from DataProcess.parameter_get import batches_train, batches_valid, meta_infor, para_dict
from Model.graph_build import graph_build
import tensorflow as tf
import numpy as np
import os
import re
from collections import Counter, defaultdict
from pyrouge import Rouge155
r = Rouge155()
from DataProcess.Constants import TPS_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"



def read_data(data_train, graph, is_test = False):
    uid, iid, y, y_tip = zip(*data_train)

    infor_title = []

    for i in range(len(uid)):
        i_infor = meta_infor[iid[i][0]]
        infor_title.append(i_infor[1])
    infor_title = np.array(infor_title)

    feed_dict = {graph.input_title: infor_title,
                 graph.input_y: y,
                 graph.input_y_tip: y_tip,
                 graph.input_uid: uid,
                 graph.input_iid: iid}
    return feed_dict


def load_model(saver, sess):
    if os.path.exists(para_dict['save_path'] + 'checkpoint'):
        modle_file = tf.train.latest_checkpoint(para_dict['save_path'])
        saver.restore(sess, modle_file)
        epoch_begin = int(''.join(filter(str.isdigit, modle_file))) + 1
    else:
        epoch_begin = 0
    return epoch_begin

def model_valid(sess, epoch):
    print('validating...', len(batches_valid), lvalid)
    if not os.path.exists(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/'):
        os.makedirs(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/')
    f_epoch = open(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/' + str(epoch) + '.txt', 'w')
    rmse, mae = 0, 0
    for batch_num in range(lvalid):
        start_index = batch_num * para_dict['batch_size']
        end_index = min((batch_num + 1) * para_dict['batch_size'], para_dict['test_length'])
        data_valid = batches_valid[start_index: end_index]
        feed_dict = read_data(data_valid, valid_graph, is_test = True)
        syn_sent, rate, rmse1, mae1= sess.run((valid_graph.sampled_words, valid_graph.r, valid_graph.rmse, valid_graph.mae), feed_dict=feed_dict)
        rmse += rmse1
        mae += mae1

        val_sent = feed_dict[valid_graph.input_y_tip]
        syn_sent = [[para_dict['tip_idx_vocab'][xx] for xx in x if xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx != para_dict['tip_vocab_size'] - 2] for x in syn_sent]

        val_sent = [[para_dict['tip_idx_vocab'] [xx] for xx in x if xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx != para_dict['tip_vocab_size'] - 2]
                    for x in val_sent]
        syn_sents, val_sents = map(list, zip(*[[' '.join(sent), ' '.join(val_sent[num])] for num, sent in enumerate(syn_sent)]))

        if not os.path.exists(TPS_DIR + '/test/system_summaries/' + str(epoch) + '/' + str(batch_num) + '/'):
            os.makedirs(TPS_DIR + '/test/system_summaries/'+ str(epoch) + '/' + str(batch_num) + '/')
        if not os.path.exists(TPS_DIR + '/test/model_summaries/'+ str(epoch) + '/' + str(batch_num) + '/'):
            os.makedirs(TPS_DIR + '/test/model_summaries/'+ str(epoch) + '/' + str(batch_num) + '/')

        for idx, sys_sent in enumerate(syn_sents):
            sys_sent = re.sub(' +', ' ', sys_sent)
            val_sent = re.sub(' +', ' ', val_sents[idx])
            f_epoch.write(sys_sent + '//' + val_sents[idx] + '//' + str(rate[idx]) +  '\n')
            if re.search(r'[a-z]+', val_sent):
                with open(TPS_DIR + '/test/system_summaries/'+ str(epoch) + '/' + str(batch_num) +'/text.' + str(batch_num * para_dict['batch_size'] + idx) + '.txt', 'w') as f1: f1.write(sys_sent)
                with open(TPS_DIR + '/test/model_summaries/'+ str(epoch) + '/' + str(batch_num) +'/text.VAL.' + str(batch_num * para_dict['batch_size'] + idx) + '.txt', 'w') as f2: f2.write(val_sent)

    # test
    result = defaultdict(float)
    for batch_num in range(lvalid):
        r.system_dir = TPS_DIR + '/test/system_summaries/'+ str(epoch) + '/' + str(batch_num)
        r.model_dir = TPS_DIR + '/test/model_summaries/' + str(epoch) + '/' +str(batch_num)
        r.system_filename_pattern = 'text.(\d+).txt'
        r.model_filename_pattern = 'text.[A-Z]+.#ID#.txt'

        output = r.convert_and_evaluate()
        output_dict = r.output_to_dict(output)
        result = dict(Counter(result) + Counter(output_dict))

    result = list(map(lambda x: (x[0], x[1] / lvalid), result.items()))
    result = [(term[0], term[1]) for term in result if 'ce' not in term[0] and 'cb' not in term[0]]
    for term in result: print(term)

    print('mse: %f, mae: %f' % (rmse / lvalid, mae / lvalid))
    f_epoch.close()


if __name__ == '__main__':
    init_scale = 0.01
    ltrain = int(para_dict['train_length'] / para_dict['batch_size'])
    lvalid = int(para_dict['valid_length'] / para_dict['batch_size'])

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse = False, initializer = initializer):
                train_graph = graph_build(para_dict)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse = True, initializer = initializer):
                valid_graph = graph_build(para_dict, is_training = False, mode_gen = 'greedy')


        config = tf.ConfigProto(allow_soft_placement = True, graph_options = tf.GraphOptions(build_cost_model = 1))
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            epoch_begin = load_model(saver, sess)
            prev_loss_total, curr_loss_total, g_loss_total, rating_loss_total = 0.1, 0.1, 0.1, 0.1
            for epoch in range(epoch_begin, para_dict['num_epochs']):

                # train
                np.random.seed(2018)
                shuffle_indices = np.random.permutation(np.arange(para_dict['train_length']))
                shuffled_data = batches_train[shuffle_indices]

                if epoch > 2: model_valid(sess, epoch)
                for batch_num in range(ltrain):
                    start_index = batch_num * para_dict['batch_size']
                    end_index = min((batch_num + 1) * para_dict['batch_size'] , para_dict['train_length'])
                    data_train = shuffled_data[start_index: end_index]
                    feed_dict = read_data(data_train, train_graph)
                    _, loss, g_loss, rating_loss = sess.run([train_graph.train_op, train_graph.loss, train_graph.g_loss, train_graph.rating_loss], feed_dict=feed_dict)
                    curr_loss_total += loss
                    g_loss_total += g_loss
                    rating_loss_total += rating_loss
                    if batch_num % 10 == 0:
                        print('training_epoch: %d, batch_num: %d, loss: %f, g_loss: %f, rating_loss: %f'
                              % (epoch, batch_num, curr_loss_total / (batch_num + 1), g_loss_total / (batch_num + 1),  rating_loss_total / (batch_num + 1)))


                current_learning_rate = sess.run(train_graph.learning_rate)
                current_learning_rate *= para_dict['learning_rate_decay']
                sess.run(train_graph.learning_rate.assign(current_learning_rate))

                print("Previous epoch loss: ", prev_loss_total)
                print("Current epoch loss: ", curr_loss_total)
                prev_loss_total, curr_loss_total, g_loss_total, rating_loss_total = curr_loss_total, 0, 0, 0
                saver.save(sess, para_dict['save_path'] + 'TextGAN_train.ckpt', global_step = epoch)






