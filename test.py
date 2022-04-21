from DataProcess.parameter_get import batches_test, meta_infor, para_dict
from Model.graph_build import graph_build
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from Constants import TPS_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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

def model_test(sess):
    for batch_num in range(ltest):
        start_index = batch_num * para_dict['batch_size']
        end_index = min((batch_num + 1) * para_dict['batch_size'], para_dict['test_length'])
        data_test = batches_test[start_index: end_index]
        feed_dict = read_data(data_test, valid_graph, is_test = True)
        syn_sent, attn_dists= sess.run((valid_graph.sampled_words, valid_graph.attn_dists), feed_dict=feed_dict)


        val_sent = feed_dict[valid_graph.input_y_tip]
        syn_sent = [[para_dict['tip_idx_vocab'][xx] for xx in x if xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx != para_dict['tip_vocab_size'] - 2] for x in syn_sent]

        val_sent = [[para_dict['tip_idx_vocab'] [xx] for xx in x if xx != para_dict['tip_vocab_size'] - 1 and xx != para_dict['tip_vocab_size'] - 3 and xx != para_dict['tip_vocab_size'] - 2]
                    for x in val_sent]
        syn_sents, val_sents = map(list, zip(*[[' '.join(sent), ' '.join(val_sent[num])] for num, sent in enumerate(syn_sent)]))



        for idx, sys_sent in enumerate(syn_sents):
            sys_sent = re.sub(' +', ' ', sys_sent)
            if sys_sent.startswith("the best knives i 've ever owned !"):
                if not os.path.exists(TPS_DIR + '/final/'):
                    os.makedirs(TPS_DIR + '/final/')
                with open(TPS_DIR + '/final/sentence.txt', 'w') as f:
                    title = [title_idx_vocab[x] for x in feed_dict[valid_graph.input_title][idx] if title_idx_vocab[x] != "<END/>"]
                    title = str(feed_dict[valid_graph.input_uid][idx][0]) + ' '+ str(feed_dict[valid_graph.input_iid][idx][0]) +  ' ' + ' '.join(title)
                    f.write(sys_sent + '\n')
                    f.write(title + '\n')
                    attn_result = []
                    for term in attn_dists[:8]:
                        attn_result.append(term[idx][:len(title.split(' '))])

                    plt.matshow(np.array(attn_result))
                    plt.colorbar()

                    ax = plt.gca()
                    ax.set_xticks(np.arange(len(title.split(' '))))
                    ax.set_xticklabels(title.split(' '))
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(sys_sent.split(' ')[:8])


                    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
                    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,ha="left", va="center", rotation_mode="anchor")


                    plt.savefig('./test.png')
                    plt.show()
                    return





if __name__ == '__main__':
    init_scale = 0.01
    ltrain = int(para_dict['train_length'] / para_dict['batch_size'])
    ltest = int(para_dict['test_length'] / para_dict['batch_size'])
    title_idx_vocab = {}
    for words, idx in para_dict['title_voc'].items(): title_idx_vocab[idx] = words

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
            saver.restore(sess,  para_dict['save_path'] + 'TextGAN_train.ckpt-4')
            model_test(sess)

