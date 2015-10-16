from __future__ import print_function
import csv

import numpy as np
import caffe

test_csv_path = 'data/test.csv'
model_path = 'lenet.prototxt'
pretrained_path = 'build/lenet_iter_10000.caffemodel'

caffe.set_mode_cpu()
clf = caffe.Classifier(model_path, pretrained_path, image_dims=(28, 28))

print('ImageId,Label')
with open(test_csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    X = np.array([np.reshape([float(v) / 255 for v in row], (28, 28, 1)) for row in reader])
    for i, y in enumerate(clf.predict(X, oversample=False)):
        print(i+1, np.argmax(y), sep=',')
