from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from VGG16mo import VGG16mo
from utils.shape_context import ShapeContext
from utils.utils import *

height = 224
width = 224

datadir = '../data/RemoteSense/ANGLE/68/'
# datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_name = '2.jpg'
IY_name = '7.jpg'

IX_image = Image.open(datadir+IX_name)
IY_image = Image.open(datadir+IX_name)

IX_image = IX_image.resize((width, height))
IY_image = IY_image.resize((width, height))

IX = np.asarray(IX_image, dtype='float32')
IY = np.asarray(IY_image, dtype='float32')
IX = np.expand_dims(IX, axis=0)
IY = np.expand_dims(IY, axis=0)
input = np.concatenate((IX, IY), axis=0)

with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: input}
    vgg = VGG16mo()
    vgg.build(images)
    D3, D4, D5 = sess.run([
        vgg.kconv3, vgg.kconv4, vgg.conv5_1
    ], feed_dict=feed_dict)

seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
X = np.array(seq, dtype='float32') * 32 + 16
Y = np.array(seq, dtype='float32') * 32 + 16
N = X.shape[0]
M = X.shape[0]
assert M==N

DX3 = D3[0, seq[:, 0], seq[:, 1]]
DY3 = D3[1, seq[:, 0], seq[:, 1]]
DX4 = D4[0, seq[:, 0], seq[:, 1]]
DY4 = D4[1, seq[:, 0], seq[:, 1]]
DX5 = D5[0, seq[:, 0], seq[:, 1]]
DY5 = D5[1, seq[:, 0], seq[:, 1]]

T = Y.copy()
GRB = gaussian_radial_basis(Y)
sigma2 = init_sigma2(X, Y)
Q = 0
iter = 1










