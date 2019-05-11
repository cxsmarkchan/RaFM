'''
Tensorflow implementation of Localized Factorization Machines

'''
import os
import argparse
import LoadData as DATA
import numpy as np
import tensorflow as tf
from time import time
import params

current = params.frappe_l


def parse_args():
    parser = argparse.ArgumentParser(description="Data Statistics.")
    parser.add_argument('--buckets', nargs='?', default='buckets/',
                        help='Running data path - for online storage system.')
    parser.add_argument('--checkpointDir', default='checkpoint/',
                        help='checkpoint')
    parser.add_argument('--summaryDir', default='summary/',
                        help='summary')
    parser.add_argument('--path', nargs='?', default='running_data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=current[0],
                        help='Choose a dataset.')
    parser.add_argument('--num_class', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = DATA.LoadData(args.buckets, args.dataset, 'log_loss', is_sparse=True,
                         loading=current[4])
    all_data = np.concatenate((data.train_data['X'], data.validation_data['X'], data.test_data['X']), axis=0)
    all_y = np.concatenate((data.train_data['Y'], data.validation_data['Y'], data.test_data['Y']), axis=0)

    stat = np.zeros([np.max(all_data, axis=(0, 1)) + 1, 4], dtype=np.int32)
    stat[:, 0] = np.array(range(stat.shape[0]))

    print('handling...')
    prev = 0
    t1 = time()
    for i in range(all_data.shape[0]):
        if (i + 1) % 1000000 == 0:
            t2 = time()
            print('%.1fs: %dM' % (t2 - t1, (i + 1) / 1000000))
            t1 = time()
        stat[all_data[i, :], 1] = np.array(range(all_data.shape[1]))
        stat[all_data[i, :], 2] += 1
        if all_y[i, 0] > 0.1:
            stat[all_data[i, :], 3] += 1

    with tf.gfile.Open(os.path.join(args.buckets, args.dataset, args.dataset + '.stat'), 'w') as fp:
        for i in range(stat.shape[0]):
            fp.write('%d %d %d %d\n' % (i, stat[i, 1], stat[i, 2], stat[i, 3]))

    print('\nstatistics:all')
    prev = 0
    num_instance = stat[:, 2]
    for num in [2 ** n for n in range(1, 30)]:
        count_sum = np.count_nonzero(num_instance[num_instance <= num])
        count = count_sum - prev
        prev = count_sum
        print('%d: %d, %d' % (num, count, count_sum))

    print('\nstatistics:pos')
    prev = 0
    num_instance = stat[:, 3]
    for num in [2 ** n for n in range(1, 30)]:
        count_sum = np.count_nonzero(num_instance[num_instance <= num])
        count = count_sum - prev
        prev = count_sum
        print('%d: %d, %d' % (num, count, count_sum))
