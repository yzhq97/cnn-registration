from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import REG
import matplotlib.pyplot as plt
from utils.thin_plate_spline import ThinPlateSpline2


#datadir = '../data/Objects/'
datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/baoshan/'
IX_path = datadir + '1.jpg'
IY_path = datadir + '1.jpg'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)
shape_arr = np.array(IY.shape[:2])
center = shape_arr / 2.0

reg = REG.CNN()
Y, T = reg.register(IX, IY)

im = plt.imread(IY_path)
#plt.gca().invert_yaxis()
plt.imshow(im)
plt.scatter(Y[:, 1], Y[:, 0])
plt.scatter(T[:, 1], T[:, 0])
plt.show()

print('generating warped image')

Y = (Y - center) / shape_arr;
T = (T - center) / shape_arr;

Y = tf.constant(np.expand_dims(Y, axis=0), dtype=np.float32)
T = tf.constant(np.expand_dims(T, axis=0), dtype=np.float32)

img = tf.constant(np.expand_dims(IY, axis=0), dtype=np.float32)
img = ThinPlateSpline2(img, Y, T, shape_arr.tolist())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img1 = sess.run(img)
img1 = np.uint8(img1[0])
cv2.imwrite('registered.png', img1)

print('finished')






    
