from __future__ import print_function

import cv2
import CNNR
import matplotlib.pyplot as plt

datadir = '../data/RemoteSense/ANGLE/68/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_path = datadir + '2.jpg'
IY_path = datadir + '7.jpg'

reg = CNNR.CNN()

X, Y, T = reg.register(IX_path, IY_path)
plt.gca().invert_yaxis()
plt.scatter(X[:, 0], X[:, 1])
#plt.scatter(Y[:, 0], Y[:, 1])
plt.scatter(T[:, 0], T[:, 1])
plt.show()



    
