from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import cv2
import Registration
import matplotlib.pyplot as plt
from utils.utils import *

datadir = '../../data/RemoteSense/TEST/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/TEST/'
#datadir = '/Users/yzhq/Code/data/Objects/'
IX_path = datadir + 'ttt1.jpg'
IY_path = datadir + 'ttt2.jpg'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)
shape_arr = np.array(IX.shape[:2])
center = shape_arr / 2.0

reg = Registration.CNN6()
xtime = time.time()
X, Y, T = reg.register(IX, IY)
print(time.time()-xtime)

registered = tps_warp(Y, T, IY, IX.shape)
cb = checkboard(IX, registered)
plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
plt.scatter(X[:, 1], X[:, 0], s=10, marker='o')
plt.scatter(T[:, 1], T[:, 0], s=10, marker='x', linewidths=0.1)
plt.show()




