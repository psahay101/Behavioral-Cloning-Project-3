#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:53:34 2018

@author: psahay101
"""



import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
#from keras.callbacks import EarlyStopping

from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random


def create_samples(data, batch_size):

  while True:

    SIZE = len(data)
    data.sample(frac = 1)

    for start in range(0, SIZE, batch_size):
      images, measurements = [], []

      for i in range(start, start + batch_size):
        if i < SIZE:
            
          posit, check = ['left', 'center', 'right'], [.25, 0, -.25]
          serial, r = data.index[i], random.choice([0, 1, 2])  
          measurementi = data['steering'][serial] + check[r] 
          path = PATH + data[posit[r]][serial][1:]
          if r == 1: path = PATH + data[posit[r]][serial]
          imagei = imread(path)  
          if random.random() > 0.5:
              imagei =np.fliplr(imagei)
              measurementi=-measurementi
          measurements.append(measurementi)
          images.append(imagei)

      yield np.array(images), np.array(measurements)

# creating the model (NVIDIA)

model = Sequential()

model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()
model.compile(optimizer = "adam", loss = "mse")

# Preparation from training 


NUM_EPOCHS = 10
BATCH_SIZE = 64

CSV_FILE = "driving_log.csv"
PATH = "data/"

DATA = pd.read_csv(PATH + CSV_FILE, usecols = [0, 1, 2, 3])

TRAINING_DATA, VALIDATION_DATA = train_test_split(DATA, test_size = 0.15)

TOTAL_VALID_DATA = len(VALIDATION_DATA)
TOTAL_TRAIN_DATA = len(TRAINING_DATA)

# Training begins here 
print('Training started...')


gen_validation = create_samples(VALIDATION_DATA, batch_size = BATCH_SIZE)
gen_training = create_samples(TRAINING_DATA, batch_size = BATCH_SIZE)

history_object = model.fit_generator(gen_training,
                 samples_per_epoch = TOTAL_TRAIN_DATA,
                 validation_data = gen_validation,
                 nb_val_samples = TOTAL_VALID_DATA,
                 nb_epoch = NUM_EPOCHS,
                 #callbacks = [early_stopping],
                 verbose = 1)

# Saving the model 
print('Saving the final model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Saved.")
