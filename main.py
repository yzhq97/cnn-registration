from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

from VGG16mo import VGG16mo

height = 224
width = 224

datadir = '../data/RemoteSense/ANGLE/68/'
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
    conv3, conv4, conv5 = sess.run([
        vgg.kconv3, vgg.kconv4, vgg.conv5_3
    ], feed_dict=feed_dict)

seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
X = np.array(seq, dtype='float32') * 32 + 16
Y = np.array(seq, dtype='float32') * 32 + 16
N = X.shape[0]

DX3 = conv3[0, seq[:, 0], seq[:, 1]]
DY3 = conv3[1, seq[:, 0], seq[:, 1]]
DX4 = conv4[0, seq[:, 0], seq[:, 1]]
DY4 = conv4[1, seq[:, 0], seq[:, 1]]
DX5 = conv5[0, seq[:, 0], seq[:, 1]]
DY5 = conv5[1, seq[:, 0], seq[:, 1]]