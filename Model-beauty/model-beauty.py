# encoding:utf-8
import time
import tensorflow as tf
import os
import sys
from load_data import Data
import numpy as np
import math
import multiprocessing
import heapq
import random as rd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'MCLN'
data_path = '../Data/'

'''
#######################################################################
Hyper-parameter settings.
'''

batch_size = 2048
dataset = 'amazon-beauty'
n_layers = 4
decay = 1e-2
lambda_m = 0.3
interval = 20

lr = 0.001
embed_size = 64
epochs = 1000
n_mca = 2

data_generator = Data(path=data_path + dataset, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = batch_size

"""
*********************************************************
Construct MCLN model
"""

class Model(object):

    def __init__(self, data_config, img_feat, text_feat, d1, d2):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.d1 = d1
        self.d2 = d2
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = data_config['lr']
        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']
        self.n_layers = data_config['n_layers']
        self.decay = data_config['decay']
        self.n_mca = n_mca

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.int_items = tf.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()

        self.im_v = tf.matmul(img_feat, self.weights['w1_v'])
        self.um_v = self.weights['user_embedding_v']

        self.im_t = tf.matmul(text_feat, self.weights['w1_t'])
        self.um_t = self.weights['user_embedding_t']

        '''
        ######################################################################################
        generate users/items embeddings
        '''
        self.ua_embeddings, self.ia_embeddings = self._create_norm_embed()
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.int_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.int_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        self.int_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.int_items)

        '''
        ######################################################################################
        generate multimodal embeddings
        '''
        self.u_g_embeddings_v = tf.nn.embedding_lookup(self.um_v, self.users)
        self.pos_i_g_embeddings_v = tf.nn.embedding_lookup(self.im_v, self.pos_items)
        self.neg_i_g_embeddings_v = tf.nn.embedding_lookup(self.im_v, self.neg_items)
        self.int_i_g_embeddings_v = tf.nn.embedding_lookup(self.im_v, self.int_items)

        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.um_t, self.users)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.im_t, self.pos_items)
        self.neg_i_g_embeddings_t = tf.nn.embedding_lookup(self.im_t, self.neg_items)
        self.int_i_g_embeddings_t = tf.nn.embedding_lookup(self.im_t, self.int_items)

        self.pos_inputs_embeddings = tf.concat([self.pos_i_g_embeddings, self.pos_i_g_embeddings_v, self.pos_i_g_embeddings_t], axis=1)
        self.neg_inputs_embeddings = tf.concat([self.neg_i_g_embeddings, self.neg_i_g_embeddings_v, self.neg_i_g_embeddings_t], axis=1)
        self.int_inputs_embeddings = tf.concat([self.int_i_g_embeddings, self.int_i_g_embeddings_v, self.int_i_g_embeddings_t], axis=1)

        self.pos_outputs_embeddings = self.causal_difference_1(self.pos_inputs_embeddings, self.int_inputs_embeddings)
        self.neg_outputs_embeddings = self.causal_difference_2(self.neg_inputs_embeddings)

        self.pos_i_g_embeddings_m = tf.layers.dense(self.pos_outputs_embeddings, units=self.emb_dim, activation=tf.nn.relu, use_bias=False)
        self.neg_i_g_embeddings_m = tf.layers.dense(self.neg_outputs_embeddings, units=self.emb_dim, activation=tf.nn.relu, use_bias=False)

        self.multiply = tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings, 1) + \
                        lambda_m*tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_v, 1) + \
                        tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_t, 1) + \
                        tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings_m, 1)

        self.loss, self.mf_loss, self.emb_loss = self.create_bpr_loss()

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['user_embedding_v'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_v')
        all_weights['user_embedding_t'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_t')

        all_weights['w1_v'] = tf.Variable(initializer([self.d1, self.emb_dim]), name='w1_v')
        all_weights['w1_t'] = tf.Variable(initializer([self.d2, self.emb_dim]), name='w1_t')

        return all_weights

    def _split_A_hat(self, X):

        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_norm_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def causal_difference_1(self, cd_inputs_embedding, cd_inputs_embedding_int):

        cd_outputs = cd_inputs_embedding
        cd_outputs_int = cd_inputs_embedding_int

        for i in range(self.n_mca):
            cl_outputs = self.counterfactual_learning_layer_1(
                query=cd_outputs,
                key_value=cd_outputs,
                query_int=cd_outputs_int,
                key_value_int=cd_outputs_int,
                activation=None,
                name='cd_cl' + str(i))

            cd_outputs = self.feed_forward_layer(
                cl_outputs,
                activation=tf.nn.relu,
                name='cd_dense' + str(i))

        return cd_outputs

    def causal_difference_2(self, cd_inputs_embedding):

        cd_outputs = cd_inputs_embedding

        for i in range(self.n_mca):
           cl_outputs = self.counterfactual_learning_layer_2(
                query=cd_outputs,
                key_value=cd_outputs,
                activation=None,
                name='cd_cl' + str(i))
           cd_outputs = self.feed_forward_layer(
                cl_outputs,
                activation=tf.nn.relu,
                name='cd_dense' + str(i))

        return cd_outputs

    # multimodal_counterfactual_learning_layer
    def counterfactual_learning_layer_1(self, query, key_value, query_int, key_value_int, activation=None, name=None):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            V = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='V')
            K = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='K')
            Q = tf.layers.dense(query, units=3 * self.emb_dim, activation=activation, use_bias=False, name='Q')
            K_int = tf.layers.dense(key_value_int, units=3 * self.emb_dim, activation=activation, use_bias=False, name='K_int')
            Q_int = tf.layers.dense(query_int, units=3 * self.emb_dim, activation=activation, use_bias=False, name='Q_int')

            score = tf.matmul(Q, tf.transpose(K)) / np.sqrt(3 * self.emb_dim)
            score_int = tf.matmul(Q_int, tf.transpose(K_int)) / np.sqrt(3 * self.emb_dim)

            score = score - score_int
            softmax = tf.nn.softmax(score, axis=1)
            attention = tf.matmul(softmax, V)

            counterfactual_learning = tf.layers.dense(attention, units=3 * self.emb_dim, activation=activation, use_bias=False, name='linear')

            counterfactual_learning += query
            counterfactual_learning = tf.contrib.layers.layer_norm(counterfactual_learning, begin_norm_axis=1)

            return counterfactual_learning

    def counterfactual_learning_layer_2(self, query, key_value, activation=None, name=None):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            V = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='V')
            K = tf.layers.dense(key_value, units=3 * self.emb_dim, activation=activation, use_bias=False, name='K')
            Q = tf.layers.dense(query, units=3 * self.emb_dim, activation=activation, use_bias=False, name='Q')

            score = tf.matmul(Q, tf.transpose(K)) / np.sqrt(3 * self.emb_dim)
            softmax = tf.nn.softmax(score, axis=1)
            attention = tf.matmul(softmax, V)

            counterfactual_learning = tf.layers.dense(attention, units=3 * self.emb_dim, activation=activation, use_bias=False, name='linear')

            counterfactual_learning += query
            counterfactual_learning = tf.contrib.layers.layer_norm(counterfactual_learning, begin_norm_axis=1)

            return counterfactual_learning

    # feed-forward layer
    def feed_forward_layer(self, inputs, activation=None, name=None):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            inner_layer = tf.layers.dense(inputs, units=3 * 4 * self.emb_dim, activation=activation)
            dense = tf.layers.dense(inner_layer, units=3 * self.emb_dim, activation=activation)

            dense += inputs
            dense = tf.contrib.layers.layer_norm(dense, begin_norm_axis=1)

        return dense

    def create_bpr_loss(self):

        pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)

        pos_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings_v), axis=1)
        neg_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings_v), axis=1)

        pos_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings_t), axis=1)
        neg_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings_t), axis=1)

        pos_scores_m = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings_m), axis=1)
        neg_scores_m = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings_m), axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer_mf_v = tf.nn.l2_loss(self.pos_i_g_embeddings_v) + tf.nn.l2_loss(self.neg_i_g_embeddings_v)
        regularizer_mf_t = tf.nn.l2_loss(self.pos_i_g_embeddings_t) + tf.nn.l2_loss(self.neg_i_g_embeddings_t)
        regularizer_mf_m = tf.nn.l2_loss(self.pos_i_g_embeddings_m) + tf.nn.l2_loss(self.neg_i_g_embeddings_m)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_v - neg_scores_v))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_t - neg_scores_t))) + \
                  tf.reduce_mean(tf.nn.softplus(-(pos_scores_m - neg_scores_m)))

        emb_loss = self.decay * (regularizer_mf + regularizer_mf_t + regularizer_mf_v + regularizer_mf_m) / self.batch_size

        loss = mf_loss + emb_loss
        self.user_embed = tf.nn.l2_normalize(self.u_g_embeddings_pre, axis=1)
        self.item_embed, self.item_embed_v, self.item_embed_t = tf.nn.l2_normalize(self.pos_i_g_embeddings_pre, axis=1), \
                                                                tf.nn.l2_normalize(self.pos_i_g_embeddings_v, axis=1), \
                                                                tf.nn.l2_normalize(self.pos_i_g_embeddings_t, axis=1)

        return loss, mf_loss, emb_loss

    def _evaluate(self, sess):
        test_num = 0
        hit_num_5 = 0
        hit_num_10 = 0
        hit_num_20 = 0
        total_ndcg_5 = 0.
        total_ndcg_10 = 0.
        total_ndcg_20 = 0.

        for user, item in data_generator.test_data():
            feed = {
                self.users: user,
                self.pos_items: item,
                self.int_items: item
            }
            predict_value = sess.run([self.multiply], feed_dict=feed)
            predict_value = predict_value[0]

            ranklist_5 = heapq.nlargest(5, zip(predict_value, item), key=lambda x: x[0])
            ranklist_10 = heapq.nlargest(10, zip(predict_value, item), key=lambda x: x[0])
            ranklist_20 = heapq.nlargest(20, zip(predict_value, item), key=lambda x: x[0])

            # if predict_value[-1] >= ranklist[-1][0]:
            #     hit_num += 1.

            test_num += 1

            for i, val in enumerate(ranklist_5):
                if val[1] == item[99]:
                    hit_num_5 += 1.
                    total_ndcg_5 += math.log(2) / math.log(i + 2)

            for i, val in enumerate(ranklist_10):
                if val[1] == item[99]:
                    hit_num_10 += 1.
                    total_ndcg_10 += math.log(2) / math.log(i + 2)

            for i, val in enumerate(ranklist_20):
                if val[1] == item[99]:
                    hit_num_20 += 1.
                    total_ndcg_20 += math.log(2) / math.log(i + 2)

        # print(predict_value.shape)
        hr_5 = hit_num_5 / test_num
        hr_10 = hit_num_10 / test_num
        hr_20 = hit_num_20 / test_num
        # loss = total_loss / (100 * test_num)
        ndcg_5 = total_ndcg_5 / test_num
        ndcg_10 = total_ndcg_10 / test_num
        ndcg_20 = total_ndcg_20 / test_num

        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if not os.path.exists('Log/'):
        os.mkdir('Log/')
    file = open('Log/ours-{}-result-{}-decay={}-layer=4.txt'.format(time.time(), dataset, decay), 'a')

    cores = multiprocessing.cpu_count() // 3

    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['decay'] = decay
    config['n_layers'] = n_layers
    config['embed_size'] = embed_size
    config['lr'] = lr
    config['batch_size'] = batch_size

    file.write('dataset={}|n_layers={}|decay={}|lr={}\n'.format(dataset, n_layers, decay, lr))

    """
    ################################################################################
    Generate the Laplacian matrix.
    """
    norm_left, norm_3, norm_4, norm_5 = data_generator.get_adj_mat()

    config['norm_adj'] = norm_4

    print('shape of adjacency', norm_4.shape)

    t0 = time.time()

    model = Model(data_config=config, img_feat=data_generator.imageFeatMatrix, text_feat=data_generator.textFeatMatrix, d1=4096, d2=300)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    ################################################################################
    Train.
    """

    for epoch in range(epochs):
        t1 = time.time()
        loss, mf_loss, mf_loss_m, emb_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items, int_items = data_generator.sample_u()

            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items,
                           model.int_items: int_items
                           })
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % interval != 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time.time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time.time()
        hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = model._evaluate(sess)

        t3 = time.time()

        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],hit@5=[%.5f],hit@10=[%.5f],hit@20=[%.5f],ndcg@5=[%.5f],ndcg@10=[%.5f],ndcg@20=[%.5f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20)
        print(perf_str)
        file.write(perf_str + '\n')

    file.close()

