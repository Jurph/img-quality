#!/usr/bin/python3

import os
import cv2
import glob 
import torch
import PIL.Image 
from piq import brisque
from pathlib import Path
import torchvision.transforms as transforms
# import imquality.brisque as brisque

# Activate CUDA 
cuda = torch.device('cuda')

# Setup file I/O
outfile = 'output.txt'
dirname = r'I:/tng/normalized/'
extensions = ['png', 'jpg', 'jpeg']
files = []
[files.extend(glob.glob(dirname + '*.' + e)) for e in extensions]
print("Found {} image files in {}".format(len(files), dirname))

with open(outfile, 'a') as o:
    for file in files:
        img = PIL.Image.open(file)
        trans = transforms.ToTensor()
        score = brisque(trans(img))
        print("{:>24} : {}".format(file, score))
        text = str(file) + ',' + str(float(score)) + "\n"
        o.write(text)
