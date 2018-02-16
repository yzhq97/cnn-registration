from __future__ import print_function

import numpy as np
import cv2
import CNNR
import matplotlib.pyplot as plt

datadir = '../data/Objects/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_path = datadir + 'car3.jpeg'
IY_path = datadir + 'car4.jpeg'

reg = CNNR.CNN()
reg.register(IX_path, IY_path)


    
