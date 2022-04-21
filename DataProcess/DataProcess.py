import pickle
import os
import csv
import tensorflow as tf
import numpy as np
from collections import defaultdict
import sys
import re
import demjson

from Constants import META_TPS_DIR, TPS_DIR, CATEGORY, TIP_LEN, CATEGORY_NUM, TITLE_LEN

tf.flags.DEFINE_string("train_data", os.path.join(TPS_DIR,  'train.csv'), "Data for training")
tf.flags.DEFINE_string("valid_data", os.path.join(META_TPS_DIR, CATEGORY +  '_valid.csv'), " Data for validation")
tf.flags.DEFINE_string("test_data", os.path.join(TPS_DIR, 'test.csv'), "Data for testing")
tf.flags.DEFINE_string("meta_data", os.path.join(META_TPS_DIR, CATEGORY + '_infor_meta.csv'), "meta_data")


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def pad_tips(sentences, max_len):
    result = []
    for sentence in sentences:
        if max_len - 3 > len(sentence):
            num_padding = max_len - len(sentence) - 2
            new_sentence = ["<START/>"] + sentence + ["<END/>"] + ["<NULL/>"] * num_padding
            result.append(new_sentence)
        else:
            new_sentence = ["<START/>"] + sentence[:max_len - 3] + ["<END/>"] + ["<NULL/>"]
            result.append(new_sentence)
    return result


def build_vocab_tip(y_tip, is_tip = False, word_frequency = 5):
    voc_num = {}
    for x in y_tip:
        for xx in x:
            if xx not in voc_num:
                voc_num[xx] = 0
            voc_num[xx] += 1

    result = [term for term, num in voc_num.items() if num > word_frequency]
    vocabulary_tip = dict((term, num) for num, term in enumerate(result))
    vocabulary_tip["<END/>"] = len(vocabulary_tip)
    if is_tip:
        vocabulary_tip["<START/>"] = len(vocabulary_tip)
        vocabulary_tip["<NULL/>"] = len(vocabulary_tip)
    return vocabulary_tip


def review_process(y_review):
    reviews = []
    for idx, review in enumerate(y_review):
        review_dict = {}
        for term in review:
            if term not in review_dict: review_dict[term] = 0
            review_dict[term] += 1
        value_sum = sum(review_dict.values())
        review_dict = dict((key, value/value_sum) for key, value in review_dict.items())
        reviews.append(review_dict)
    return reviews

def review_pad(y_review, review_len, review_vocab_size):
    vocab_list, frequency_list = [], []
    for review_dict in y_review:
        vocab = list(review_dict.keys())
        frequency = list(review_dict.values())
        if len(vocab) < review_len:
            vocab = vocab + [review_vocab_size - 1] * (review_len - len(vocab))
            frequency = frequency + [0.0] * (review_len - len(frequency))
        vocab_list.append(np.array(vocab))
        frequency_list.append(np.array(frequency))
    return np.array(vocab_list), np.array(frequency_list)


def load_data(train_data, valid_data, test_data):
    y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip, \
    uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test\
        = load_data_and_labels(train_data, valid_data, test_data)

    user_num = len(pickle.load(open(os.path.join(META_TPS_DIR, 'user2id'), 'rb')))
    item_num = len(pickle.load(open(os.path.join(META_TPS_DIR, 'item2id'), 'rb')))
    print("load data done")

    # tip process
    vocabulary_tip = build_vocab_tip(y_train_tip + y_valid_tip + y_test_tip, True)  # tip词集
    tip_idx_vocab = {}
    for words, idx in vocabulary_tip.items(): tip_idx_vocab[idx] = words
    y_train_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_train_tip]
    y_valid_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_valid_tip]
    y_test_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_test_tip]
    max_tip_len = TIP_LEN + 3  # tip_len
    y_train_tip = pad_tips(y_train_tip, max_tip_len)
    y_valid_tip = pad_tips(y_valid_tip, max_tip_len)
    y_test_tip = pad_tips(y_test_tip, max_tip_len)
    y_train_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_train_tip])
    y_valid_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_valid_tip])
    y_test_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_test_tip])
    print("pad tip done")



    return [y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip, vocabulary_tip, tip_idx_vocab, max_tip_len,
            uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num]


def load_data_and_labels(train_data, valid_data, test_data):
    print("training...")
    uid_train, iid_train = [], []
    y_train, y_train_tip = [], []
    uid_train_dict, iid_train_dict = {}, {}
    f_train = csv.reader(open(train_data, "r", encoding='utf-8'))
    for line in f_train:
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        uid_train_dict[int(line[0])] = 0
        iid_train_dict[int(line[1])] = 0
        y_train.append(float(line[2]))
        y_train_tip.append(line[3].split(" "))

    print("validing...")
    uid_valid, iid_valid = [], []
    y_valid, y_valid_tip, y_valid_review = [], [], []
    f_valid = csv.reader(open(valid_data, "r", encoding='utf-8'))
    for line in f_valid:
        if int(line[0]) in uid_train_dict and int(line[1]) in iid_train_dict:
            uid_valid.append(int(line[0]))
            iid_valid.append(int(line[1]))
            y_valid.append(float(line[2]))
            y_valid_tip.append(line[3].split(" "))


    print("testing...")
    uid_test, iid_test = [], []
    y_test, y_test_tip, y_test_review = [], [], []
    f_test = csv.reader(open(test_data, "r", encoding='utf-8'))
    for line in f_test:
        if int(line[0]) in uid_train_dict and int(line[1]) in iid_train_dict:
            uid_test.append(int(line[0]))
            iid_test.append(int(line[1]))
            y_test.append(float(line[2]))
            y_test_tip.append(line[3].split(" "))

    return [np.array(y_train), np.array(y_valid), np.array(y_test),
            y_train_tip, y_valid_tip, y_test_tip,
            np.array(uid_train), np.array(iid_train),
            np.array(uid_valid), np.array(iid_valid),
            np.array(uid_test), np.array(iid_test)]


def pad_sentences(description, description_len, padding_word="<END/>"):
    if description_len > len(description):
        num_padding = description_len - len(description)
        new_sentence = description + [padding_word] * num_padding
    else:
        new_sentence = description[: description_len]
    return new_sentence


def title_description_process(sentence_list, len_proportion):
    vocabulary_sentence = build_vocab_tip(sentence_list, is_tip=False)
    sentence_list = [[xx for xx in x if xx in vocabulary_sentence] for x in sentence_list]
    sentence_len_list = np.array([len(x) for x in sentence_list])
    X = np.sort(sentence_len_list)
    sentence_len = X[int(len_proportion * len(sentence_len_list)) - 1]
    sentence_list = [pad_sentences(x, sentence_len) for x in sentence_list]
    sentence_list = [[vocabulary_sentence[word] for word in words if word in vocabulary_sentence] for words in sentence_list]
    return sentence_len, vocabulary_sentence, sentence_list


def meta_data_process(meta_data):
    category_num = defaultdict(int)
    item_id, category, title = [], [], []
    f_meta = csv.reader(open(meta_data, "r", encoding='utf-8'))
    for line in f_meta:
        line[1] = demjson.decode(line[1])
        for x in line[1]:
            for xx in x:
                category_num[xx] += 1
        item_id.append(line[0])
        category.append(line[1])
        title.append(clean_str(line[3]).split(' '))

    category_num = sorted(category_num.items(), key=lambda x: x[1], reverse=True)
    category_select = [cate for cate, num in category_num[0: CATEGORY_NUM]]
    category2id = dict([(cate, i) for i, cate in enumerate(category_select)])
    for num, x in enumerate(category):
        vec = [0] * len(category_select)
        for xx in x:
            for xxx in xx:
                if xxx in category_select:
                    vec[category2id[xxx]] = 1
        category[num] = vec

    title_len, title_voc, title = title_description_process(title, TITLE_LEN)

    meta_infor = {}
    for idx, id in enumerate(item_id):
        meta_infor[int(id)] = [np.array(category[idx]), np.array(title[idx])]

    para = {}
    para['category2id'] = category2id
    para['category_num'] = CATEGORY_NUM
    para['title_voc'] = title_voc
    para['title_vocab_size'] = len(title_voc)
    para['title_len'] = title_len
    para['item_infor'] = meta_infor
    return meta_infor, para


if __name__ == '__main__':

    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()

    meta_infor, para_dict = meta_data_process(FLAGS.meta_data)
    pickle.dump(meta_infor, open(os.path.join(TPS_DIR, CATEGORY + '.meta_infor'), 'wb'))

    y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip,  vocabulary_tip, tip_idx_vocab, max_tip_len, \
    uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num = load_data(FLAGS.train_data,
                                                                                                   FLAGS.valid_data,
                                                                                                   FLAGS.test_data)

    y_train = y_train[:, np.newaxis]  # 插入新维度
    uid_train = uid_train[:, np.newaxis]
    iid_train = iid_train[:, np.newaxis]

    y_valid = y_valid[:, np.newaxis]
    uid_valid = uid_valid[:, np.newaxis]
    iid_valid = iid_valid[:, np.newaxis]

    y_test = y_test[:, np.newaxis]
    uid_test = uid_test[:, np.newaxis]
    iid_test = iid_test[:, np.newaxis]

    print('ziping...')
    batches_train = np.array(list(zip(uid_train, iid_train, y_train, y_train_tip)))
    batches_valid = np.array(list(zip(uid_valid, iid_valid, y_valid, y_valid_tip)))
    batches_test = np.array(list(zip(uid_test, iid_test, y_test, y_test_tip)))
    pickle.dump(batches_train, open(os.path.join(TPS_DIR, CATEGORY + '.train'), 'wb'))
    pickle.dump(batches_valid, open(os.path.join(TPS_DIR, CATEGORY + '.valid'), 'wb'))
    pickle.dump(batches_test, open(os.path.join(TPS_DIR, CATEGORY + '.test'), 'wb'))

    para_dict['user_num'] = user_num
    para_dict['item_num'] = item_num
    para_dict['max_tip_len'] = max_tip_len
    para_dict['tip_vocab'] = vocabulary_tip
    para_dict['tip_idx_vocab'] = tip_idx_vocab
    para_dict['tip_vocab_size'] = len(vocabulary_tip)
    para_dict['train_length'] = len(y_train)
    para_dict['valid_length'] = len(y_valid)
    para_dict['test_length'] = len(y_test)
    pickle.dump(para_dict, open(os.path.join(TPS_DIR, CATEGORY + '.para'), 'wb'))








