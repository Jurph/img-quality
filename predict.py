#!/usr/bin/python3
# Iterates through 'dirname' and produces a file where each row is:
# "filename : brisque score"
# BRISQUE is an image quality metric (0 is better)
# Read more at https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

import os
import glob 
import torch 
import PIL.Image 
import numpy as np
import tensorflow as tf
from pathlib import Path
from piq import brisque, psnr
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing import image

# TODO: use itertools to make a named tuple with filename, BRISQUE score, and classifier output 

# Setup file I/O
outfile = 'output.txt'
dirname = r'I:/tng/normalized/'
extensions = ['png', 'jpg', 'jpeg']
files = []
[files.extend(glob.glob(dirname + '*.' + e)) for e in extensions]
print("Found {} image files in {}".format(len(files), dirname))

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model = loaded_model
model.summary()

# Iterate through files 
with open(outfile, 'a') as o:
    scores = []
    confidences = []
    counter = 0
    for file in files:
        counter += 1

        # Score BRISQUE image quality metric
        img = PIL.Image.open(file)
        trans = transforms.ToTensor()
        # For high-RAM GPUs, you can uncomment the "to.cuda" and get more performance
        gpu_img = trans(img)  #.to('cuda')
        brisk = brisque(gpu_img) 
        scores.append(float(brisk))
 
        # Run classifier 
        img = image.load_img(file, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        faceconfidence = 1.000 - float(classes[0])
        confidences.append(faceconfidence)
        if faceconfidence > 0.5:
            category = "face"
            pass
        else:
            category = "not a face"
            pass
        
        # Write scoring data
        text = "{:0>6d} of {:0>6d} : {:>24} : {:>8} : {:>8} : {}\n".format(counter, len(files), file, brisk, faceconfidence, category)
        print(text)
        o.write(text)
#        if not len(scores) % 10:
#            ranked = sorted(scores)
#            lowscore = ranked[0]
#            hiscore = ranked[-1]
#            middle = ranked[int(len(scores)/2)]
#            print("MIN: {} | MEDIAN: {} | MAX: {}".format(lowscore, middle, hiscore))

# Findings (on my data): 
# For strongly-suspected "good" images, median is ~35 but ranges as high as 116
# 90th percentile is near 55 with a long tail. 
# For strongly-suspected "bad" images, median is ~70 but ranges as low as 14
# 90th percentile is near 80; distribution may be bimodal with categories for "not a face" and "face of very low quality"