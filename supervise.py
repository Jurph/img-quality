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
goodfile = 'goodexamples.txt'
badfile = 'badexamples.txt'
dirname = r'I:/tng/normalized/'
extensions = ['png', 'jpg', 'jpeg']
files = []
[files.extend(glob.glob(dirname + '*.' + e)) for e in extensions]
print("Found {} image files in {}".format(len(files), dirname))

# Iterate through files 
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
    if brisk > 70:
        print("Found bad file {} with BRISQUE = {}".format(file, brisk))
        with open(badfile, 'a') as b:
            b.write("{}\n".format(file))
    elif brisk >= 20:
        pass
    elif brisk < 20:
        print("Found good file {} with BRISQUE = {}".format(file, brisk))
        with open(goodfile, 'a') as g:
            g.write("{}\n".format(file))
    else:
        pass 

    # Run classifier 
#    img = image.load_img(file, target_size=(256, 256))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    images = np.vstack([x])
#    classes = model.predict(images, batch_size=10)
#    faceconfidence = 1.000 - float(classes[0])
#    confidences.append(faceconfidence)
#    if faceconfidence > 0.5:
#        category = "face"
#        pass
#    else:
#        category = "not a face"
#        pass
    
    # Write scoring data
    # text = "{:0>6d} of {:0>6d} : {:>24} : {:>8} : {:>8} : {}\n".format(counter, len(files), file, brisk, faceconfidence, category)
    # print(text)
    # o.write(text)
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