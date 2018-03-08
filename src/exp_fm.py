from __future__ import print_function
import time
import numpy as np
import Registration
from utils.utils import *
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

datadir = '/Users/yzhq/Code/data/ottawa/'
matoutdir = datadir + 'matout/'
name1 = '1a'
name2 = '1b'
ext = '.jpg'
IX_path = datadir + name1 + ext
IY_path = datadir + name2 + ext

IX = cv2.imread(IX_path)
# IX = cv2.resize(IX, (600, 800))
IY = cv2.imread(IY_path)
# IY = cv2.resize(IY, (600, 800))
shape = IX.shape[:2]
shape_arr = np.array(shape)
center = shape_arr / 2.0
window_width = shape[1]
window_height = shape[0]

print('initializing')
reg = Registration.CNN7()
reg.init_thres = 1.15

print('registering')
X, Y, T = reg.register(IX, IY)

print('generating warped image')
registered = tps_warp(Y, T, IY, IX.shape)
print('generating checkboard image')
cb = checkboard(IX, registered, 11)

plt.imshow(cv2.cvtColor(cb, cv2.COLOR_RGB2BGR))
plt.show()

PD = pairwise_distance(X, T)
C, _ = match(PD)
X, Y, T = X[C[:, 0]], Y[C[:, 1]], T[C[:, 1]]

outfile = name1 + '_' + name2 + '.mat'
sio.savemat(matoutdir+outfile, {'X':X, 'Y':Y, 'T':T})
print('saved to ' + outfile)