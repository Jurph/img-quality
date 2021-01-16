# JR's first neural net classifier!
# This classifier tries to determine whether a crop-and-chop program, 
# like the one used to build FFHQ or CelebA-HQ, is delivering high quality
# faces or weird anomalies. 
# 
# Substantially cribbed from https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa 

import os
import numpy as np
import tensorflow as tf
from itertools import cycle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from scipy import interpolate
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'I:\tng\neural-net'
subdir = ['train', 'valid']
classes = ['goodface', 'notface', 'badface'] # Three categories 

# Directory with our training "face" pictures
train_face_dir = os.path.join('I:\tng\neural-net\train\face')

# Directory with our training "not face" pictures
train_notface_dir = os.path.join('I:\tng\neural-net\train\notface')

# Directory with our validation "face" pictures
valid_face_dir = os.path.join('I:\tng\neural-net\valid\face')

# Directory with our validation "not face" pictures
valid_notface_dir = os.path.join('I:\tng\neural-net\valid\notface')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'I:/tng/neural-net/train/',  # This is the source directory for training images
        classes = ['goodface', 'notface', 'badface'],
        target_size=(1024, 1024),  # All images will be resized to (was) 256x256
        batch_size=7, # was 120 // a 2GB video card can handle at least 16
        # Use binary labels
        class_mode='categorical', shuffle=True)

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'I:/tng/neural-net/valid/',  # This is the source directory for training images
        classes = ['goodface', 'notface', 'badface'], 
        target_size=(1024, 1024),  # All images will be resized to (was) 256x256
        batch_size=7, # was 19 // a 2GB video card can handle at least 16
        # Use binary labels
        class_mode='categorical',
        shuffle=True)

# Original model is Flatten(256x256x3), Dense(64 relu), Dense(1, sigmoid)
model = Sequential()

flatten0 = Flatten(input_shape = (256, 256, 3))
conv1 = Convolution2D(filters = 4, kernel_size = (3, 3), input_shape = (1024, 1024, 3), activation = 'relu')
pool1 = MaxPooling2D(pool_size=(2,2))
conv2 = Convolution2D(filters = 8, kernel_size = (3, 3), input_shape = (512, 512, 3), activation = 'relu')
pool2 = MaxPooling2D(pool_size=(2,2))
conv3 = Convolution2D(filters = 16, kernel_size = (3, 3), input_shape = (256, 256, 3), activation = 'relu')
conv3a = Convolution2D(filters = 32, kernel_size = (3, 3), activation='relu')
pool3 = MaxPooling2D(pool_size=(2,2))
conv4 = Convolution2D(filters = 32, kernel_size = (3, 3), input_shape = (128, 128, 3), activation = 'relu')
conv4a = Convolution2D(filters = 32, kernel_size = (3, 3), activation = 'relu')
pool4 = MaxPooling2D(pool_size=(2,2))
conv5 = Convolution2D(filters = 64, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu')
conv5a = Convolution2D(filters = 64, kernel_size = (3, 3), activation = 'relu')
pool5 = MaxPooling2D(pool_size=(2,2))
conv6 = Convolution2D(filters = 128, kernel_size = (3, 3), input_shape = (32, 32, 3), activation = 'relu')
conv6a = Convolution2D(filters = 128, kernel_size = (3, 3), activation = 'relu')
pool6 = MaxPooling2D(pool_size=(2,2))
conv7 = Convolution2D(filters = 128, kernel_size = (3, 3), input_shape = (16, 16, 3), activation = 'relu')
conv7a = Convolution2D(filters = 128, kernel_size = (3, 3), activation = 'relu')
pool7 = MaxPooling2D(pool_size=(2,2))
flatten1 = Flatten()
dense0 = Dense(72, activation=tf.nn.relu)
dense1 = Dense(32, activation=tf.nn.relu)
dense2 = Dense(128, activation=tf.nn.relu)
output = Dense(3, activation=tf.nn.sigmoid)

# Original recipe (gets >85% on three categories)
# structure = [flatten0, dense0, output]

# JR's attempt (not bad!)
# structure = [conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5, pool5, conv6, pool6, conv7, pool7, flatten1, dense1, output]

# Based on VGG concept
structure = [conv1, pool1, conv2, pool2, conv3, conv3a, pool3, conv4, conv4a, conv4a, pool4, conv5, conv5a, conv5a, pool5, conv6, conv6a, conv6a, conv6a, pool6, conv7, conv7a, conv7a, pool7, flatten1, dense2, dense2, output]

for layer in structure:
        model.add(layer)
model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      steps_per_epoch=19,  
      epochs=100,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=11)

model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")