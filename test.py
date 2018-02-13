from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from VGG16mo import VGG16mo
from utils.shape_context import ShapeContext
from utils.utils import *
import matplotlib.pyplot as plt

height = 224
width = 224

datadir = '../data/RemoteSense/ANGLE/68/'
# datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_name = '2.jpg'
IY_name = '13.jpg'

IX_image = Image.open(datadir+IX_name)
IY_image = Image.open(datadir+IY_name)

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
    D1, D2, D3, D4 = sess.run([
        vgg.kconv3_1, vgg.kconv3_3, vgg.kconv4_3, vgg.conv5_1
    ], feed_dict=feed_dict)

seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
X = np.array(seq, dtype='float32') * 32 + 16
Y = np.array(seq, dtype='float32') * 32 + 16
N = X.shape[0]
M = X.shape[0]
assert M==N

D = np.concatenate([D1, D2, D3, D4], axis=3)
DX = D[0, seq[:, 0], seq[:, 1]]
DY = D[1, seq[:, 0], seq[:, 1]]

C, q = match(DX, DY)
x = np.arange(1.0, 1.4, 0.001)
y = np.zeros_like(x)
for i in range(x.shape[0]):
    y[i] = np.where(q>=x[i])[0].shape[0]

plt.plot(x, y)
plt.show()

