'''
Tensorflow implementation of Localized Factorization Machines

'''
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from time import time
import argparse

import LoadData as DATA
import common
import params

# I use this structure to store the statistics of each dataset
current = params.frappe_l


def parse_args():
    parser = argparse.ArgumentParser(description="Run RaFM.")
    parser.add_argument('--buckets', nargs='?', default='buckets/',
                        help='Input data path.')
    parser.add_argument('--checkpointDir', default='checkpoint/',
                        help='checkpoint')
    parser.add_argument('--summaryDir', default='summary/',
                        help='summary')
    parser.add_argument('--dataset', nargs='?', default=current[0],
                        help='Choose a dataset.')
    parser.add_argument('--continuous', type=int, default=0,
                        help='whether to continue training from existing checkpoint')
    parser.add_argument('--save_epoch', type=int, default=0,
                        help='Epochs between two checkpoint. Set as 0 to disable saving')
    parser.add_argument('--epoch', type=int, default=current[1],
                        help='Number of epochs.')
    parser.add_argument('--iter', type=int, default=current[2],
                        help='Iterations per epoch')
    # iterations per epoch = number_of_train_samples / batch_size
    parser.add_argument('--batch_size', type=int, default=current[3],
                        help='Batch size.')
    parser.add_argument('--embedding_dim', nargs='?', default='[64, 256]',
                        help='Dimensionality of embedding vectors.')
    parser.add_argument('--critical_num', nargs='?', default='[64]',
                        help='Heuristic: the number of samples determining the dimensionality of embeddings')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate of free part')
    parser.add_argument('--dependent_lr_coef', nargs='?', default='[1.0]',
                        help='coefficient multiplied to the SGD of dependent part')
    parser.add_argument('--num_field', type=int, default=current[4][4],
                        help='number of nonzero features')
    parser.add_argument('--regularization_factor', nargs='?', default='[5e-6, 5e-6]',
                        help='Regularization factor of each bilinear part.')
    parser.add_argument('--loss_type', nargs='?', default=current[5],
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--seed', type=int, default=2017,
                        help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                        help='whether to log')

    return parser.parse_args()


class RaFM(BaseEstimator, TransformerMixin):
    def __init__(self, num_feature, num_field, save_file,
                 is_continuous, save_epoch,
                 embedding_dim, feature_table,
                 loss_type, reg_factor,
                 optimizer_type, learning_rate, dependent_lr_coef,
                 epoch, iterations, batch_size,
                 verbose,
                 random_seed=2016, is_sparse=True, is_lookup=True,
                 suffix=''):
        """
        :param num_feature: No. of features in the input data
        :param num_field: No. of nonzero features in the input data
        :param save_file: file path to save
        :param is_continuous: whether to load an existing checkpoint
        :param save_epoch: epochs between two checkpoint. Set as 0 to disable storing.
        :param embedding_dim: rank of embedding vectors (In RaFM, it is a list)
        :param feature_table: a table containing the #occurring of each feature
        :param loss_type: square loss or log loss
        :param reg_factor: list of regularization factors of each rank
        :param optimizer_type: AdamOptimizer, AdagradOptimizer
        :param learning_rate: learning rate of free part
        :param dependent_lr_coef: list of learning rate multipliers of dependent part (i.e. lr_dependent = lr_free * coef)
        :param epoch: No. of epoch
        :param iterations: No. of iterations per epoch
        :param batch_size: Batch size
        :param verbose: whether to print log
        :param random_seed:
        :param is_sparse: is the input data sparse (We only test the "True" option)
        :param is_lookup: is the input data a lookup format (i.e. values of each feature is 0/1) (We only test the "True" option)
        """
        # bind params to class
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.num_field = num_field
        self.is_continuous = is_continuous
        self.save_epoch = save_epoch
        self.feature_table = feature_table
        self.dependent_lr_coef = dependent_lr_coef
        self.feature_flag = np.minimum(1, feature_table).astype(np.float32)
        self.save_file = save_file
        self.loss_type = loss_type
        self.num_features = num_feature
        self.lambda_bilinear = reg_factor
        self.epoch = epoch
        self.iterations = iterations
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.is_sparse = is_sparse
        self.is_lookup = is_lookup
        self.suffix = suffix

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # Input data.
            if self.is_lookup:
                self.train_features = tf.placeholder(tf.int32, shape=[None, self.num_field])  # None * num_features
            elif self.is_sparse:
                self.train_features = tf.sparse_placeholder(tf.float32,
                                                            shape=[None, self.num_features])  # None * num_features
            else:
                self.train_features = tf.placeholder(tf.float32, shape=[None, self.num_features])  # None * num_features
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1

            # Variables.
            self.weights = self._initialize_weights()
            self.weights_feature = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'feature')

            # Model.

            ###################################################################################
            # bilinear embedding
            self.sample_embedding = []  # the embeddings of each FM
            self.sample_flag = []  # flags of whether a specific column exists in each FM.
            self.sample_embedding.append(
                tf.nn.embedding_lookup(self.weights['feature_bilinear_0'], self.train_features))
            # the first FM contains all nonzero columns
            self.sample_flag.append(tf.ones([self.num_field], dtype=tf.float32))
            self.sample_flag[0] = self.sample_flag[0][tf.newaxis, :, tf.newaxis]

            # setting flags and weights for each FM
            for k in range(self.feature_table.shape[1]):
                # the k+1-th FM
                tmp_zero = tf.zeros([1, self.embedding_dim[k + 1]], dtype=tf.float32)
                cur_weight_table = tf.concat((tmp_zero, self.weights['feature_bilinear_%d' % (k + 1)]), axis=0)
                cur_ids = tf.nn.embedding_lookup(self.feature_table[:, k], self.train_features)
                self.sample_embedding.append(tf.nn.embedding_lookup(cur_weight_table, cur_ids))
                self.sample_flag.append(tf.nn.embedding_lookup(self.feature_flag[:, k], self.train_features))
                self.sample_flag[-1] = self.sample_flag[-1][:, :, tf.newaxis]

            self.bilinear = []  # the bilinear parts of each FM

            # core of RaFM: multiple embeddings
            base = tf.zeros_like(self.train_labels, dtype=np.float32)
            for k in range(self.feature_table.shape[1]):
                free_part = self.sample_embedding[k] * (self.sample_flag[k] - self.sample_flag[k + 1])
                dependent_part = self.sample_embedding[k] * self.sample_flag[k + 1]
                # Note the stop_gradient here!
                low_output = common.get_bilinear_embedding_from_feature(tf.stop_gradient(free_part) + dependent_part)
                low_output = tf.reduce_sum(low_output, axis=1, keep_dims=True)
                self.bilinear.append(tf.add_n([tf.stop_gradient(base), low_output]))
                low_interaction = common.get_bilinear_embedding_from_feature(dependent_part + free_part)
                low_interaction = tf.reduce_sum(low_interaction, axis=1, keep_dims=True)
                correction = common.get_bilinear_embedding_from_feature(dependent_part)
                correction = tf.reduce_sum(correction, axis=1, keep_dims=True)
                base = tf.add_n([base, -correction, low_interaction])

            final_high_interaction = common.get_bilinear_embedding_from_feature(self.sample_embedding[-1])
            final_high_interaction = tf.reduce_sum(final_high_interaction, axis=1, keep_dims=True)
            self.bilinear.append(tf.add_n([base, final_high_interaction]))

            # linear embedding
            self.weights_linear_reshape = self.weights['feature_linear']
            self.linear = common.get_linear_embedding(self.train_features, self.weights_linear_reshape, self.is_sparse,
                                                      True)
            self.linear = self.linear[:, tf.newaxis]

            # bias
            self.weights_bias_reshape = self.weights['bias']
            self.bias = tf.ones_like(self.train_labels, dtype=np.float32) * self.weights_bias_reshape

            # out[k]: \mathcal{B}_{1, k+1} in our paper
            self.out = []
            for k in range(self.feature_table.shape[1] + 1):
                self.out.append(tf.add_n([self.bilinear[k], self.linear, self.bias]))

            # The loss function, which uses different update rules for free variables and dependent variables
            self.loss = 0
            if self.loss_type == 'square_loss':
                # free variables
                self.loss += tf.nn.l2_loss(tf.subtract(self.train_labels, self.out[-1]))
                # loss of dependent variables. We use stop_gradient to mimic the update rule of dependent parts
                for k in range(self.feature_table.shape[1]):
                    self.loss += self.dependent_lr_coef[k] * tf.nn.l2_loss(
                        tf.subtract(self.out[k], tf.stop_gradient(self.out[k + 1])))
            elif self.loss_type == 'log_loss':
                for k in range(len(self.out)):
                    self.out[k] = tf.sigmoid(self.out[k])
                # free variables
                self.loss += tf.losses.log_loss(self.train_labels, self.out[-1], weights=1.0, epsilon=1e-07, scope=None)
                # loss of dependent variables. We use stop_gradient to mimic the update rule of dependent parts
                for k in range(self.feature_table.shape[1]):
                    loss = self.dependent_lr_coef[k] * tf.losses.log_loss(tf.stop_gradient(self.out[k + 1]),
                                                                          self.out[k], weights=1.0, epsilon=1e-07,
                                                                          scope=None)
                    self.loss += loss

            self.reg_loss = 0  # L2 regularization of each embedding
            for k in range(self.feature_table.shape[1] + 1):
                if self.lambda_bilinear[k] > 0:
                    self.reg_loss += tf.contrib.layers.l2_regularizer(self.lambda_bilinear[k])(
                        self.weights['feature_bilinear_%d' % k])

            self.loss += self.reg_loss

            self.optimizer = common.get_optimizer(self.optimizer_type, self.learning_rate,
                                                  self.loss, None)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            if self.is_continuous == 1:
                self.saver.restore(self.sess, self.save_file + self.suffix)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        """
        feature_bilinear: interaction term, [features_M, K]
        feature_linear: linear term, [features_M, 1]
        bias: constant term, [1, 1]
        :return:
        """
        all_weights = dict()
        with tf.variable_scope('feature'):
            all_weights['feature_bilinear_0'] = tf.Variable(
                tf.random_normal([self.num_features, self.embedding_dim[0]], 0.0, 0.01),
                name='feature_bilinear_0')  # num_features * D_0
            for k in range(self.feature_table.shape[1]):
                all_weights['feature_bilinear_%d' % (k + 1)] = tf.Variable(
                    tf.random_normal([np.max(self.feature_table[:, k]), self.embedding_dim[k + 1]], 0.0, 0.01),
                    name='feature_bilinear_%d' % (k + 1), dtype=tf.float32)  # num_features * D_{k+1}
            all_weights['feature_linear'] = tf.Variable(
                tf.random_uniform([self.num_features], 0.0, 0.0),
                name='feature_linear')  # num_features
            all_weights['bias'] = tf.Variable(tf.random_uniform([1]), name='bias')  # 1

        return all_weights

    def gradient_descent(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def train(self, data):  # fit a dataset
        validation_min = 100000
        # to prevent from OOM problem
        evaluate_batch_size = int(min(1048576 / 16, data.test_num, data.validation_num))
        evaluate_num = int(min(data.test_num, data.validation_num) / evaluate_batch_size) + 1
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            loss_train, auc_train, loss_low_train = self.evaluate(
                data.get_random_block_from_data(evaluate_batch_size, 'train'))
            loss_valid = np.zeros([evaluate_num])
            auc_valid = np.zeros([evaluate_num])
            loss_low_valid = np.zeros([evaluate_num])
            loss_test = np.zeros([evaluate_num])
            auc_test = np.zeros([evaluate_num])
            loss_low_test = np.zeros([evaluate_num])
            for i in range(evaluate_num):
                loss_valid[i], auc_valid[i], loss_low_valid[i] = self.evaluate(
                    data.get_random_block_from_data(evaluate_batch_size, 'validation'))
                loss_test[i], auc_test[i], loss_low_test[i] = self.evaluate(
                    data.get_random_block_from_data(evaluate_batch_size, 'test'))
            loss_valid = np.mean(loss_valid)
            auc_valid = np.mean(auc_valid)
            loss_low_valid = np.mean(loss_low_valid)
            loss_test = np.mean(loss_test)
            auc_test = np.mean(auc_test)
            loss_low_test = np.mean(loss_low_test)
            print(
                "Init: \t train=(%.4f, %.4f, %.4f), validation=(%.4f, %.4f, %.4f), test=(%.4f, %.4f, %.4f) [%.1f s]" % (
                    loss_train, auc_train, loss_low_train,
                    loss_valid, auc_valid, loss_low_valid,
                    loss_test, auc_test, loss_low_test,
                    time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            for i in range(self.iterations):
                # generate a batch
                batch_xs = data.get_random_block_from_data(self.batch_size, 'train')
                # Fit training
                self.gradient_descent(batch_xs)
            t2 = time()

            # output validation
            loss_train, auc_train, loss_low_train = self.evaluate(
                data.get_random_block_from_data(evaluate_batch_size, 'train'))
            loss_valid = np.zeros([evaluate_num])
            auc_valid = np.zeros([evaluate_num])
            loss_low_valid = np.zeros([evaluate_num])
            loss_test = np.zeros([evaluate_num])
            auc_test = np.zeros([evaluate_num])
            loss_low_test = np.zeros([evaluate_num])
            for i in range(evaluate_num):
                loss_valid[i], auc_valid[i], loss_low_valid[i] = self.evaluate(
                    data.get_random_block_from_data(evaluate_batch_size, 'validation'))
                loss_test[i], auc_test[i], loss_low_test[i] = self.evaluate(
                    data.get_random_block_from_data(evaluate_batch_size, 'test'))
            loss_valid = np.mean(loss_valid)
            auc_valid = np.mean(auc_valid)
            loss_low_valid = np.mean(loss_low_valid)
            loss_test = np.mean(loss_test)
            auc_test = np.mean(auc_test)
            loss_low_test = np.mean(loss_low_test)

            if self.verbose > 0:
                print(
                    "Epoch %d [%.1f s]\ttrain=(%.4f, %.4f, %.4f), validation=(%.4f, %.4f, %.4f), test=(%.4f, %.4f, %.4f) [%.1f s]"
                    % (
                        epoch + 1, t2 - t1, loss_train, auc_train, loss_low_train, loss_valid, auc_valid,
                        loss_low_valid,
                        loss_test,
                        auc_test, loss_low_test, time() - t2))

            if self.save_epoch > 0 and (epoch + 1) % self.save_epoch == 0:
                self.saver.save(self.sess, self.save_file + self.suffix)
                if self.verbose > 0:
                    print("Checkpoint %d saved." % (epoch + 1))

    def evaluate(self, data):  # evaluate the results for an input set
        # num_example = data['Y'].shape[0]
        num_example = data['Y'].shape[0]
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y']}

        # predictions of RaFM and the first FM (i.e. the FM with the lowest dimension)
        predictions, predictions_low = self.sess.run((self.out[-1], self.out[0]), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_pred_low = np.reshape(predictions_low, (num_example,))

        y_true = np.reshape(data['Y'], (num_example,))

        if self.loss_type == 'square_loss':
            # bounding the value
            predictions_dependent = np.maximum(y_pred,
                                                 np.ones(num_example) * min(y_true))
            predictions_dependent = np.minimum(predictions_dependent,
                                                 np.ones(num_example) * max(y_true))
            predictions_dependent_low = np.maximum(y_pred_low,
                                                     np.ones(num_example) * min(y_true))
            predictions_dependent_low = np.minimum(predictions_dependent_low,
                                                     np.ones(num_example) * max(y_true))
            loss = math.sqrt(mean_squared_error(y_true, predictions_dependent))
            auc = 0
            loss_low = math.sqrt(mean_squared_error(y_true, predictions_dependent_low))

        elif self.loss_type == 'log_loss':
            y_pred = np.minimum(y_pred, 0.9999)
            y_pred = np.maximum(y_pred, 0.0001)
            y_pred_low = np.minimum(y_pred_low, 0.9999)
            y_pred_low = np.maximum(y_pred_low, 0.0001)
            loss = log_loss(y_true, y_pred)  # I haven't checked the log_loss
            auc = roc_auc_score(y_true, y_pred)
            loss_low = log_loss(y_true, y_pred_low)

            # the code for computing accuracy if needed
            # y_pred[y_pred > 0.499] = 1
            # y_pred[y_pred < 0.5] = 0
            # y_pred = y_pred.astype(dtype=np.int32)
            # y_true = y_true.astype(dtype=np.int32)
            # acc = 1 - np.sum(y_pred == y_true) / (num_example * 1.0)

        else:
            loss = 0
            auc = 0
            loss_low = 0

        return loss, auc, loss_low


if __name__ == '__main__':
    is_sparse = True
    args = parse_args()

    # get statistics
    data_stat = np.zeros([current[4][3], 4], np.int32)
    for line in tf.gfile.Open(os.path.join(args.buckets, args.dataset, args.dataset + '.stat'), 'r'):
        strs = line.strip().split(' ')
        data_stat[int(strs[0]), :] = list(map(int, strs))

    critical_num = eval(args.critical_num)

    feature_table = np.zeros([data_stat.shape[0], len(critical_num)], dtype=np.int)
    cur = np.zeros([args.num_field], dtype=np.int32)
    mapped_pos = np.ones([len(critical_num)], dtype=np.int32)
    for i in range(data_stat.shape[0]):
        for j in range(len(critical_num)):
            if data_stat[i, 2] > critical_num[j]:
                feature_table[i, j] = mapped_pos[j]
                mapped_pos[j] += 1

    data = DATA.LoadData(args.buckets, args.dataset, args.loss_type, is_sparse=is_sparse, loading=current[4])

    if args.verbose > 0:
        print(
            "RaFM: dataset=%s, embedding_dim=%s, critical_num=%s, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, dependent_lr_coef=%s, reg=%s, optimizer=%s"
            % (
                args.dataset, args.embedding_dim, args.critical_num, args.loss_type, args.epoch, args.batch_size,
                args.lr,
                args.dependent_lr_coef,
                args.regularization_factor, args.optimizer))

    save_file = os.path.join(args.checkpointDir, args.dataset, args.dataset)
    # Training
    t1 = time()
    model = RaFM(data.num_feature, args.num_field, save_file,
                 args.continuous, args.save_epoch,
                 eval(args.embedding_dim), feature_table,
                 args.loss_type, eval(args.regularization_factor),
                 args.optimizer, args.lr, eval(args.dependent_lr_coef),
                 args.epoch, args.iter, args.batch_size,
                 args.verbose,
                 random_seed=args.seed, is_sparse=is_sparse, is_lookup=True,
                 suffix='.rafm')
    model.train(data)
