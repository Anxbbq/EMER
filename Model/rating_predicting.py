import tensorflow as tf

class RatingPrediction():
    def __init__(self,  para_dict):
        self.para_dict = para_dict

    def rating_prediction(self, embeded_uid, embeded_iid, input_y):
        Wu = tf.get_variable('Wru', [self.para_dict['embedding_id_size'], self.para_dict['n_latent']])
        self.u_feas = tf.matmul(embeded_uid, Wu)

        Wi = tf.get_variable('Wri', [self.para_dict['embedding_id_size'], self.para_dict['n_latent']])
        self.i_feas = tf.matmul(embeded_iid, Wi)

        Br = tf.get_variable('Wrb', [self.para_dict['n_latent']])
        feas = tf.sigmoid(self.u_feas + self.i_feas + Br)

        for i in range(0, 1):
            Wr = tf.get_variable('Wr' + str(i), [self.para_dict['n_latent'], self.para_dict['n_latent']])
            Br = tf.get_variable('Wb' + str(i), [self.para_dict['n_latent']])  # name：新变量或现有变量的名称，这个参数是必须的，函数会根据变量名称去创建或者获取变量
            feas = tf.sigmoid(tf.matmul(feas, Wr) + Br)

        Wrr = tf.get_variable('Wrr', [self.para_dict['n_latent'], 1])
        Brr = tf.get_variable('Wbr', [1])
        r = tf.matmul(feas, Wrr) + Brr
        mse = tf.reduce_mean(tf.square(input_y - r))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(input_y - r)))
        mae = tf.reduce_mean(tf.abs(input_y - r))
        return r, feas, mse, rmse, mae







