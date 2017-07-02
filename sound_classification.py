
# coding: utf-8

# In[ ]:

import glob
import os
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime


# In[86]:

# データ読み込み
import urbansound8k_loader as dataset

#parent_dir = 'UrbanSound8K/audio/'
#tr_features, tr_labels = dataset.load_urbansound8k(parent_dir, ['fold1', 'fold2'])
#ts_features, ts_labels = dataset.load_urbansound8k(parent_dir, ['fold3'])

get_ipython().magic('time tr_features, tr_labels, ts_features, ts_labels = dataset.load_from_npy_files()')

if (len(tr_features) !=  len(tr_labels)):
    print('WARN: invalid # of training data. features=' + str(len(tr_features)) + ', labels=' + str(len(tr_labels)))
elif (len(ts_features) !=  len(ts_labels)):
    print('WARN: invalid # of tast data. features=' + str(len(ts_features)) + ', labels=' + str(len(ts_labels)))
elif (len(tr_features) == 0 || len(ts_features) == 0):
    print('WARN: no data.')
else: 
    print('loaded successfully. # of train data=' + str(len(tr_features)) + ', # of test data=' + str(len(ts_features)))


# In[ ]:




# In[64]:

training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = 10
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01
log_dir = './log'


# In[65]:

# モデル構築、学習、評価 by keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import keras.callbacks
import keras.backend.tensorflow_backend as KTF

log_filepath = './keras/log'
batch_size = 128
old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(n_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    cbks = [tb_cb]
    
    print ('fit start: ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    history = model.fit(tr_features, tr_labels,
                        batch_size=batch_size,
                        epochs=training_epochs,
                        verbose=1,
                        callbacks=cbks, 
                        validation_data=(ts_features, ts_labels))

    print ('evaluate start: ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    score = model.evaluate(ts_features, ts_labels, verbose=0)
    print('Test accuracy:', score[1])
    print ('finished: ', datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

KTF.set_session(old_session)


# In[71]:

type(tr_features)


# In[80]:

np.save('tr_features.npy', tr_features)
np.save('tr_labels.npy', tr_labels)
np.save('ts_features.npy', ts_features)
np.save('ts_labels.npy', ts_labels)


# In[ ]:




# In[ ]:



