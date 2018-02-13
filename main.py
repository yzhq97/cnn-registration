from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from VGG16mo import VGG16mo
from utils.hungarian import Hungarian
from utils.utils import *

height = 224
width = 224

tolerance = 1e-3
freq = 5
tau_0 = 1.3
delta = 0.025
epsilon = 0.4
omega = 0.5
lambd = 3.0

#datadir = '../data/RemoteSense/ANGLE/68/'
datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_name = '2.jpg'
IY_name = '7.jpg'

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
PD = pairwise_distance(DX, DY)
C_all, quality = match(PD)

T = Y.copy()
GRB = gaussian_radial_basis(Y)
A = np.zeros([M, 2])
sigma2 = init_sigma2(X, Y)
tau = tau_0

Hung = Hungarian()
Pm = None

Q = 0
dQ = float('Inf')
iter = 1

while iter <= 100 and abs(dQ) > tolerance and sigma2 > 1e-4:
    T_old = T.copy()
    Q_old = Q

    if (iter-1) % freq == 0:
        # refine
        C = C_all[np.where(quality >= tau)]
        L = np.zeros_like(PD)
        L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
        L = L/L.max()
        L[np.where(L==0.0)] = 1.0

        C = Hung.calculate(L)
        Pm = np.ones_like(PD) * (1.0-epsilon)/N
        Pm[C[:, 0], C[:, 1]] = 1.0
        Pm = Pm / np.sum(Pm, axis=0)

        tau = tau - delta

    Po, P1, Np, tmp, Q = compute(X, Y, T_old, Pm, sigma2, omega)
    Q = Q + lambd / 2 * np.trace(np.dot(np.dot(A.transpose(), GRB), A))

    
