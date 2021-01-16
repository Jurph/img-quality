#! usr/bin/python
# do stuff to a set of files named in a txt files

# Import modules 
import os
import sys
import shutil 
import argparse 

infile = "badexamples.txt"
inpath = "I:\\tng\\normalized\\"
outpath = "I:\\tng\\neural-net\\train\\badface\\" 

fileHandler = open(infile, "r")
while True:
	line = fileHandler.readline()
	if not line:
		break
	(dirname, filename) = os.path.split(line)
	print("DIR: {}   |  FILE: {}".format(dirname, filename))
	(shortname, extension) = os.path.splitext(filename)
	src = str(inpath+filename).strip()
	dest = outpath
	try:
		# print("NOP: would have moved {} to {}".format(src, dest))
		shutil.copy2(src,dest)
	except OSError:
		print("Skipping {}".format(filename))
		
