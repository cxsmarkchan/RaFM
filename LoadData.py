import numpy as np
import tensorflow as tf
from time import time
import common


class LoadData(object):
    '''
    Loading data from input files.
    '''
    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type, is_sparse=True, loading=None):
        self.path = path + dataset + "/"
        self.is_sparse = is_sparse

        if loading is not None:
            self.is_lookup = True
            self.train_num = loading[0]
            self.validation_num = loading[1]
            self.test_num = loading[2]
            self.num_feature = loading[3]
            self.nonzero_num = loading[4]

            self.train_file = self.path + dataset + ".train"
            self.validation_file = self.path + dataset + ".validation"
            self.test_file = self.path + dataset + ".test"
            self.train_data = self.construct_data_from_file(self.train_file, self.train_num)
            self.validation_data = self.construct_data_from_file(self.validation_file, self.validation_num)
            self.test_data = self.construct_data_from_file(self.test_file, self.test_num)

        else:
            self.is_lookup = False
            self.train_file = self.path + dataset + ".train.libfm"
            self.validation_file = self.path + dataset + ".validation.libfm"
            self.test_file = self.path + dataset + ".test.libfm"
            if self.is_sparse:
                self.num_feature = 0
                self.train_num = 0
                self.validation_num = 0
                self.test_num = 0
                self.train_data, self.validation_data, self.test_data = self.construct_data(loss_type)
            else:
                self.num_feature = self.map_features()
                self.train_data, self.validation_data, self.test_data = self.construct_data(loss_type)

    def get_block_from_data(self, start_index, batch_size, type):  # generate a random block of training data
        if type == 'train':
            data = self.train_data
        elif type == 'validation':
            data = self.validation_data
        else:
            data = self.test_data

        if self.is_lookup:
            return {
                'X': data['X'][start_index:start_index + batch_size, :],
                'Y': data['Y'][start_index:start_index + batch_size, :]
            }
        else:
            if self.is_sparse:
                return {
                    'X': common.sparse_concat(data['X_sparse_list'][start_index:start_index + batch_size], self.num_feature),
                    'Y': data['Y'][start_index:start_index + batch_size, np.newaxis]
                }
            else:
                return {
                    'X': data['X'][start_index:start_index + batch_size, :],
                    'Y': data['Y'][start_index:start_index + batch_size, np.newaxis]
                }

    def get_random_block_from_data(self, batch_size, type):  # generate a random block of training data
        if type == 'train':
            data = self.train_data
        elif type == 'validation':
            data = self.validation_data
        else:
            data = self.test_data

        if self.is_lookup:
            start_index = np.random.randint(0, data['Y'].shape[0] - batch_size + 1)
        else:
            start_index = np.random.randint(0, data['Y'].shape[0] - batch_size + 1)

        return self.get_block_from_data(start_index, batch_size, type)

    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        features_train, self.train_num = self.read_features(self.train_file)
        features_validation, self.validation_num = self.read_features(self.validation_file)
        features_test, self.test_num = self.read_features(self.test_file)
        return max([features_train, features_validation, features_test])

    def read_features(self, file):  # read a feature file
        num_features = 0
        num = 0
        for line in tf.gfile.Open(file, 'r'):
            items = line.strip().split(' ')
            num = num + 1
            for item in items[1:]:
                feature_id = int(item.strip().split(':')[0])
                if num_features < feature_id + 1:
                    num_features = feature_id + 1

        return num_features, num

    def construct_data_from_file(self, file_name, example_num):
        t1 = time()

        data_arr = np.zeros([example_num, self.nonzero_num], dtype=np.int32)
        data_y = np.zeros([example_num, 1], dtype=np.float32)
        i = 0
        print(file_name + '.libfm')
        if tf.gfile.Exists(file_name + '.libfm'):
            for line in tf.gfile.Open(file_name + '.libfm', 'r'):
                items = line.strip().split(' ')
                data_y[i, 0] = float(items[0])
                data_arr[i, :] = list(map(int, items[1:]))
                # data_arr[i, :] = map(int, items)

                i = i + 1
                if i % 1000000 == 0:
                    t2 = time()
                    print('%.1fs: %dM' % (t2 - t1, i / 1000000))
                    t1 = time()
        else:
            num = 0
            while tf.gfile.Exists(file_name + '.%d.libfm' % num):
                print(file_name + '.%d.libfm' % num)
                for line in tf.gfile.Open(file_name + '.%d.libfm' % num, 'r'):
                    items = line.strip().split(' ')
                    data_y[i, 0] = float(items[0])
                    data_arr[i, :] = list(map(int, items[1:]))

                    i = i + 1
                    if i % 1000000 == 0:
                        t2 = time()
                        print('%.1fs: %dM' % (t2 - t1, i / 1000000))
                        t1 = time()
                num += 1
        return {'X': data_arr, 'Y': data_y}

    def construct_data(self, loss_type):
        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.train_file, self.train_num)
        if loss_type == 'log_loss':
            train_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            train_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of training:", len(Y_))
        self.train_num = len(Y_)

        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.validation_file, self.validation_num)
        if loss_type == 'log_loss':
            valid_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            valid_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of validation:", len(Y_))
        self.validation_num = len(Y_)

        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.test_file, self.test_num)
        if loss_type == 'log_loss':
            test_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            test_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of test:", len(Y_))
        self.test_num = len(Y_)

        return train_data, valid_data, test_data

    def read_data(self, file, data_num):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        if not self.is_sparse:
            X_ = np.zeros([data_num, self.num_feature], dtype=np.float32)
        else:
            X_ = None
        # Y_ = np.zeros([data_num], dtype=np.float32)
        Y_ = []
        Y_for_logloss = []
        X_sparse_list = []
        # Y_for_logloss = np.zeros([data_num], dtype=np.float32)
        i = 0
        for line in tf.gfile.Open(file, 'r'):
            indices = []
            values = []
            items = line.strip().split(' ')
            Y_.append(1.0 * float(items[0]))

            if float(items[0]) > 0:  # > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append(v)

            for item in items[1:]:
                key_value_pair = item.strip().split(':')
                key = int(key_value_pair[0])
                if key >= self.num_feature:
                    self.num_feature = key + 1
                value = float(key_value_pair[1])
                if not self.is_sparse:
                    X_[i, key] = float(value)
                indices.append(key)
                values.append(int(value))
            X_sparse_list.append({'indices': indices, 'values': values})

            i = i + 1
            if i % 1000000 == 0:
                print('Data Loaded: %dk' % (i / 1000))

        X_sparse = None

        Y_ = np.array(Y_, dtype=np.float32)
        Y_for_logloss = np.array(Y_for_logloss, dtype=np.float32)

        return X_, Y_, Y_for_logloss, X_sparse_list, X_sparse

    def construct_dataset(self, X_, Y_, X_sparse_list, X_sparse):
        Data_Dic = {}
        Data_Dic['Y'] = Y_
        Data_Dic['X'] = X_
        Data_Dic['X_sparse_list'] = X_sparse_list
        Data_Dic['X_sparse'] = X_sparse
        return Data_Dic


