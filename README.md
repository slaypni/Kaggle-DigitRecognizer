Caffe LeNet for Kaggle "Digit Recognizer"
=========================================

Caffe needs to be installed.

```
mkdir build
python convert_data.py -k 10
sh train_lenet.sh
python predict.py > build/result.csv
```
