from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import cv2
import REG
import matplotlib.pyplot as plt
from utils.utils import *

#datadir = '../data/Objects/'
datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/TEST/'
#datadir = '/Users/yzhq/Code/data/Objects/'
IX_path = datadir + 'e1.JPG'
IY_path = datadir + 'e2.JPG'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)
shape_arr = np.array(IX.shape[:2])
center = shape_arr / 2.0

reg = REG.CNN()
xtime = time.time()
X, Y, T = reg.register(IX, IY)
print(time.time()-xtime)

# im = plt.imread(IX_path)
# plt.imshow(im)
# plt.scatter(X[:, 1], X[:, 0], s=20, marker='o')
# plt.scatter(T[:, 1], T[:, 0], s=20, marker='x', linewidths=0.5)
# plt.show()

registered = tps_warp(Y, T, IY, IX.shape)
cb = checkboard(IX, registered)
plt.imshow(cb)
plt.show()




