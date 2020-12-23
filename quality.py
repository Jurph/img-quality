#!/usr/bin/python3
# Iterates through 'dirname' and produces a file where each row is:
# "filename : brisque score"
# BRISQUE is an image quality metric (0 is better)
# Read more at https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

import os
import cv2
import glob 
import torch
import PIL.Image 
from piq import brisque, psnr
from pathlib import Path
import torchvision.transforms as transforms

# Setup file I/O
outfile = 'output.txt'
dirname = r'I:/tng/neural-net/train/face/'
extensions = ['png', 'jpg', 'jpeg']
files = []
[files.extend(glob.glob(dirname + '*.' + e)) for e in extensions]
print("Found {} image files in {}".format(len(files), dirname))

# Iterate through files 
with open(outfile, 'a') as o:
    scores = []
    for file in files:
        img = PIL.Image.open(file)
        trans = transforms.ToTensor()
        gpu_img = trans(img).to('cuda')
        brisk = brisque(gpu_img) 
        scores.append(float(brisk))
        text = "{:>24} : {:>8}\n".format(file, brisk)
        print(text)
        o.write(text)
        if not len(scores) % 10:
            ranked = sorted(scores)
            lowscore = ranked[0]
            hiscore = ranked[-1]
            middle = ranked[int(len(scores)/2)]
            print("MIN: {} | MEDIAN: {} | MAX: {}".format(lowscore, middle, hiscore))

# Findings (on my data): 
# For strongly-suspected "good" images, median is ~35 but ranges as high as 116
# 90th percentile is near 55 with a long tail. 
# For strongly-suspected "bad" images, median is ~70 but ranges as low as 14
# 90th percentile is near 80; distribution may be bimodal with categories for "not a face" and "face of very low quality"