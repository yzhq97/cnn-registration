from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import cv2

#datadir = '../../data/RemoteSense/TEST/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/TEST/'
datadir = '/Users/yzhq/Code/data/'
IX_path = datadir + 'JSJ3.jpg'
IY_path = datadir + 'JSJ4.jpg'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)
shape_arr = np.array(IX.shape[:2])
center = shape_arr / 2.0

for i in range(4):
    name = Registration.names[i]
    print(name)
    reg = Registration.get_reg_by_name(name)
    reg.init_thres = 1.1

    start_time = time.time()
    X, Y, T = reg.register(IX, IY)
    print("total %s seconds" % (time.time() - start_time))

    try:
        start_time = time.time()
        registered = tps_warp(Y, T, IY, IX.shape)
        cb = checkboard(IX, registered, 11)
        print("warp and checkboard cost %s seconds" % (time.time() - start_time))
        plt.subplot(221 + i)
        plt.title(name)
        plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
        plt.scatter(X[:, 1], X[:, 0], s=10, marker='o')
        plt.scatter(T[:, 1], T[:, 0], s=10, marker='x', linewidths=0.1)
    except:
        plt.subplot(221 + i)
        plt.title(name+'(TPS failed)')
        plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
        plt.scatter(X[:, 1], X[:, 0], s=10, marker='o')
        plt.scatter(T[:, 1], T[:, 0], s=10, marker='x', linewidths=0.1)

    print()

plt.show()






    
