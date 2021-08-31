
import pickle
import os
from DataProcess.Constants import META_TPS_DIR, TPS_DIR, CATEGORY, CATEGORY_NUM

batches_train = pickle.load(open(os.path.join(TPS_DIR, CATEGORY + '.train'), 'rb'))
batches_test = pickle.load(open(os.path.join(TPS_DIR, CATEGORY + '.test'), 'rb'))
meta_infor = pickle.load(open(os.path.join(TPS_DIR, CATEGORY + '.meta_infor'), 'rb'))
para_dict = pickle.load(open(os.path.join(TPS_DIR, CATEGORY + '.para'), 'rb'))
#title_str=pickle.load(open(os.path.join(META_TPS_DIR, CATEGORY + '.title'), 'rb'))

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
para_dict['num_epochs'] = 8
para_dict['save_path'] = TPS_DIR + '/save_path/'
