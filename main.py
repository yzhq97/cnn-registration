from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import REG
import matplotlib.pyplot as plt
from utils.utils import *

#datadir = '../data/Objects/'
datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/TEST/'
#datadir = '/Users/yzhq/Code/data/Objects/'
IX_path = datadir + 'r1.JPG'
IY_path = datadir + 'r2.JPG'

IX = plt.imread(IX_path)
IY = plt.imread(IY_path)
shape_arr = np.array(IX.shape[:2])
center = shape_arr / 2.0

for i in range(6):
    name = REG.names[i]
    print(name)
    reg = REG.get_reg_by_name(name)

    start_time = time.time()
    X, Y, T = reg.register(IX, IY)
    print("total %s seconds" % (time.time() - start_time))

    start_time = time.time()
    registered = tps_warp(Y, T, IY, IX.shape)
    cb = checkboard(IX, registered)
    print("warp and checkboard cost %s seconds" % (time.time() - start_time))

    #plt.gca().invert_yaxis()
    plt.subplot(231 + i)
    plt.title(name)
    plt.imshow(cb)
    plt.scatter(X[:, 1], X[:, 0], s=20, marker='o')
    plt.scatter(T[:, 1], T[:, 0], s=20, marker='x', linewidths=0.5)

    print()

plt.show()






    
