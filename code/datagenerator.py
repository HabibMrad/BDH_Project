# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:40:26 2018

@author: Srini
"""

from pandas import Series
import pandas as pd
import numpy as np
from keras_contrib.applications.densenet import DenseNetImageNet121
from keras.layers import Dense
from keras.models import Model
from keras.utils import Sequence
import keras
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
ia.seed(1)
import keras.backend as K
from keras.optimizers import Adam
import os


train_val = pd.read_pickle('train_val_filtered.pkl')
test = pd.read_pickle('test_filtered.pkl')

files = os.listdir('../data/trial_images/')

#########################
#%%

#partition = {'train': files[:21], 'validation': files[21:]}
partition = {'train': files}

labels = Series(train_val.array_label.values,index=train_val.img_filename).to_dict()

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-15, 15)),  # random rotate image
    iaa.Affine(scale=(0.8, 1.1)),  # randomly scale the image
], random_order=True)

#%%

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size= 4, dim=(224,224), n_channels=3,
                 n_classes=14, current_state = 'train',shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels # labels dictionary
        self.list_IDs = list_IDs #partition['train'] or partition['validation'], type = list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.current_state = current_state
        self.len = int(np.floor(len(self.list_IDs) / self.batch_size))
        self.on_epoch_end()
        print("for DataGenerator", current_state, "total rows are:", len(self.list_IDs), ", len is", self.len)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, batch_no):
        'Generate one batch of data'
        # Generate indexes of the batch
        print('batch number: {}'.format(batch_no))
        ind = self.indexes[batch_no*self.batch_size:(batch_no+1)*self.batch_size]
        print('batch indexes: {}'.format(ind))
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in ind]
        print('temp image ids: {}'.format(list_IDs_temp))
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        print('indexes: {}'.format(self.indexes))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('shuffled indexes: {}'.format(self.indexes))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

           path = os.path.join('../data/trial_images/', ID)
           print('Image ID: {}'.format(ID))
           img = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
           X[i, :, :, :] = img
           y[i, :] = labels[ID]

        if self.current_state == 'train':
            print('augmenting images')
            x_augmented = seq.augment_images(X)
        else:
            x_augmented = X

        return x_augmented, y


#%%

def build_model():
    """

    Returns: a model with specified weights

    """
    # define the model, use pre-trained weights for image_net
    base_model = DenseNetImageNet121(input_shape=(224, 224, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')

    x = base_model.output
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def unweighted_binary_crossentropy(y_true, y_pred):
    """
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns: the sum of binary cross entropy loss across all the classes

    """
    return K.sum(K.binary_crossentropy(y_true, y_pred))

#%%

initial_lr = 0.001
initial_lr = 0.001
resized_height = 224
resized_width = 224
num_channel = 3
num_classes = 14
epochs = 2 #200

model_multi_gpu = build_model()

model_multi_gpu.compile(optimizer=Adam(lr=initial_lr), loss=unweighted_binary_crossentropy)

model_multi_gpu.fit_generator(generator=DataGenerator(partition['train'],labels, current_state='train'),epochs=3, verbose=2, workers=1, shuffle=False)



