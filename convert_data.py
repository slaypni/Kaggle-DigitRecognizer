import csv
import argparse

import numpy as np
import lmdb
import caffe

train_csv_path = './data/train.csv'
train_lmdb_path = './build/train_lmdb'
test_lmdb_path = './build/test_lmdb'

parser = argparse.ArgumentParser()
parser.add_argument('-k')
args = parser.parse_args()
k = None if args.k == None else int(args.k)

with open(train_csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    train_env = lmdb.open(train_lmdb_path, map_size=1099511627776)
    test_env = lmdb.open(test_lmdb_path, map_size=1099511627776)
    with train_env.begin(write=True) as train_txn, test_env.begin(write=True) as test_txn:
        for i, row in enumerate(reader):
            y = int(row[0])
            x = np.array(np.reshape(map(int, row[1:]), (1, 28, 28)), dtype=np.uint8)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = x.shape[0]
            datum.height = x.shape[1]
            datum.width = x.shape[2]
            datum.data = x.tobytes()
            datum.label = y
            key = '{:08}'.format(i)
            txn = train_txn if k == None or (i % k) >= 1 else test_txn
            txn.put(key.encode('ascii'), datum.SerializeToString())
