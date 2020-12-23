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
classes = ['face', 'notface']

# Directory with our training dandelion pictures
train_face_dir = os.path.join('I:\tng\neural-net\train\face')

# Directory with our training grass pictures
train_notface_dir = os.path.join('I:\tng\neural-net\train\notface')

# Directory with our validation dandelion pictures
valid_face_dir = os.path.join('I:\tng\neural-net\valid\face')

# Directory with our validation grass pictures
valid_notface_dir = os.path.join('I:\tng\neural-net\valid\notface')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'I:/tng/neural-net/train/',  # This is the source directory for training images
        classes = ['face', 'notface'],
        target_size=(256, 256),  # All images will be resized to 256x256
        batch_size=120,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'I:/tng/neural-net/valid/',  # This is the source directory for training images
        classes = ['face', 'notface'], 
        target_size=(256, 256),  # All images will be resized to 256x256
        batch_size=19,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (256,256,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      steps_per_epoch=16,  
      epochs=20,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)

model.summary()