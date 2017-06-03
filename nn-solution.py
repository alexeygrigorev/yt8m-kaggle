
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 
import tensorflow as tf

from glob import glob
from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.utils import shuffle


# In[14]:

import keras.backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import Callback

from keras_tqdm import TQDMCallback as KerasTQDMCallback
# from keras_tqdm import TQDMNotebookCallback as KerasTQDMCallback


# reading the data

label_names = pd.read_csv('data/label_names.csv', index_col='label_id').label_name.to_dict()

train_recs = sorted(glob('data/video_level/train*.tfrecord'))
val_recs = sorted(glob('data/video_level/validate*.tfrecord'))
test_recs = sorted(glob('data/video_level/test*.tfrecord'))


def read_data(files_list):
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    for f in files_list:
        for example in tf.python_io.tf_record_iterator(f):
            tf_example = tf.train.Example.FromString(example)
            features = tf_example.features.feature

            id = features['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
            label = np.array(features['labels'].int64_list.value, dtype='uint16')
            rgb = np.array(features['mean_rgb'].float_list.value, dtype='float32')
            audio = np.array(features['mean_audio'].float_list.value, dtype='float32')
            vid_ids.append(id)
            labels.append(label)
            mean_rgb.append(rgb)
            mean_audio.append(audio)

    return vid_ids, labels, mean_rgb, mean_audio

def read_file(fine_name):
    return read_data([file_name])

def to_dense_lab(labs):
    res = np.zeros(output_dim, dtype='uint8')
    res[labs] = 1
    return res

def to_dense_labs(list_labs):
    return np.array([to_dense_lab(l) for l in list_labs])


# training the scaler

train_ids, train_labs, train_rgb, train_audio = read_data(train_recs[:10])
X_10_concat = np.hstack([train_rgb, train_audio])
scaler = StandardScaler().fit(X_10_concat)


# validation set

val_ids, val_labs, val_rgb, val_audio = read_data(val_recs[:100])

X_100_val = np.hstack([val_rgb, val_audio])
X_100_val = scaler.transform(X_100_val)

val_labs_dense = to_dense_labs(val_labs)


# prepare the model 

class WatchlistCallback(Callback):
    def __init__(self, watchlist):
        super(Callback, self).__init__()
        self.X, self.y = watchlist

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X, verbose=0)
        ll = log_loss(self.y, y_pred)
        print('epoch no %d, logloss=%.4f' % (epoch, ll))

watchlist = WatchlistCallback(watchlist=(X_100_val, val_labs_dense))


def prepare_batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def image_gen(file_list, n=2, seed=0):
    i = seed + 0

    while True:
        files = shuffle(file_list, random_state=i)

        batches = prepare_batches(files, n)

        for batch in batches:
            _, labs, rgb, audio = read_data(batch)
            X = np.hstack([rgb, audio])
            X = scaler.transform(X)
            y = to_dense_labs(labs)

            yield X, y

        i = i + 1

input_dim = X_100_val.shape[1]
output_dim = len(label_names)


# train the model 

model = Sequential()

model.add(Dense(input_dim=input_dim, units=5500, kernel_initializer='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(units=output_dim, kernel_initializer='glorot_uniform')) 
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))



gen = image_gen(train_recs, n=8, seed=2)
model.fit_generator(gen, steps_per_epoch=32, epochs=270, verbose=0, 
                    callbacks=[watchlist, KerasTQDMCallback()])

K.set_value(model.optimizer.lr, 0.001)
model.fit_generator(gen, steps_per_epoch=32, epochs=32, verbose=0, 
                    callbacks=[watchlist, KerasTQDMCallback()])

K.set_value(model.optimizer.lr, 0.0001)
model.fit_generator(gen, steps_per_epoch=32, epochs=32, verbose=0, 
                    callbacks=[watchlist, KerasTQDMCallback()])

model.save_weights('ololo.bin')


# prepare submission

def prepare_pred_row(prow):
    classes = (-prow).argsort()[:20]
    scores = prow[classes]
    return ' '.join(['%d %0.3f' % (c, s) for (c, s) in zip(classes, scores)])

with open('subm.csv', 'w') as f:
    f.write('VideoId,LabelConfidencePairs\n')

    for fn in tqdm(test_recs):
        ids, _, rgb, audio = read_file(fn)
        X = np.hstack([rgb, audio])
        X = scaler.transform(X)

        pred = model.predict(X)

        for id, prow in zip(ids, pred):
            lab_conf = prepare_pred_row(prow)
            f.write('%s,%s\n' % (id, lab_conf))

import os
os.system('gzip subm.csv')

