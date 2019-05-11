import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def sparse_concat(input_list, num_features):
    data_shape = (len(input_list), num_features)
    indices = []
    values = []

    i = 0
    for item in input_list:
        values.extend(item['values'])
        item_indices = i * np.ones([len(item['indices']), 2], dtype=np.int32)
        item_indices[:, 1] = item['indices']
        indices.extend(item_indices.tolist())
        i = i + 1

    return tf.SparseTensorValue(indices, values, data_shape)


def get_optimizer(optimizer_type, learning_rate, obj, var_list=None):
    # Optimizer.
    if optimizer_type == 'AdamOptimizer':
        Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-8)
    elif optimizer_type == 'AdagradOptimizer':
        Optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                              initial_accumulator_value=1e-8)
    elif optimizer_type == 'GradientDescentOptimizer':
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'MomentumOptimizer':
        Optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)

    return Optimizer.minimize(obj, var_list=var_list)


def get_bilinear_embedding_from_feature(feature_embedding):
    '''
    get bilinear embedding when there exists different dims of factors.
    lookup table is needed
    :param x: None * num_features
    :param weights: num_features * num_factor
    :param is_sparse:
    :return:
    '''

    # orignal part
    summed_features_emb = tf.reduce_sum(feature_embedding, 1)
    squared_sum_features_emb = tf.reduce_sum(tf.square(feature_embedding), 1)
    summed_square_features_emb = tf.square(summed_features_emb)  # None * num_factor
    # bilinear part
    embedding = 0.5 * tf.subtract(summed_square_features_emb, squared_sum_features_emb)  # None * num_factor

    return embedding


def get_bilinear_embedding(x, weights, is_sparse, is_lookup=False):
    '''

    :param x: None * num_features
    :param weights: num_features * num_factor
    :param is_sparse:
    :return:
    '''
    if is_lookup:
        nonzero_embeddings = tf.nn.embedding_lookup(weights, x)
        summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)
        squared_sum_features_emb = tf.reduce_sum(tf.square(nonzero_embeddings), 1)
    elif is_sparse:
        summed_features_emb = tf.sparse_tensor_dense_matmul(x, weights)  # None * num_factor
        squared_sum_features_emb = tf.sparse_tensor_dense_matmul(tf.square(x), tf.square(weights))
    else:
        summed_features_emb = tf.matmul(x, weights)  # None * num_factor
        squared_sum_features_emb = tf.matmul(tf.square(x), tf.square(weights))

    # get the element-multiplication
    summed_square_features_emb = tf.square(summed_features_emb)  # None * num_factor

    # bilinear part
    embedding = 0.5 * tf.subtract(summed_square_features_emb, squared_sum_features_emb)  # None * num_factor

    return embedding


def get_linear_embedding(x, weights, is_sparse, is_lookup=False):
    '''
    :param x: None * num_features
    :param weights: num_features * num_factor
    :param is_sparse:
    :return:
    '''

    if is_lookup:
        linear = tf.nn.embedding_lookup(weights, x)
        linear = tf.reduce_sum(linear, 1)
    elif is_sparse:
        linear = tf.sparse_tensor_dense_matmul(x, weights)
    else:
        linear = tf.matmul(x, weights)

    return linear


