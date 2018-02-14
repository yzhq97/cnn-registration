from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from VGG16mo import VGG16mo
from utils.utils import *
import cv2
from lap import lapjv
import matplotlib.pyplot as plt
#from LAPJV import lap

height = 224
width = 224

tolerance = 1e-2
freq = 5
tau_0 = 5.0
delta = 0.05
epsilon = 0.4
omega = 0.5
beta = 2.0
lambd = 0.5

datadir = '../data/Objects/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_name = 'car3.jpeg'
IY_name = 'car4.jpeg'

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

X = X / 224.0
Y = Y / 224.0

D = np.concatenate([D3, D4], axis=3)
DX = D[0, seq[:, 0], seq[:, 1]]
DY = D[1, seq[:, 0], seq[:, 1]]
PD = pairwise_distance(DX, DY)
C_all, quality = match(PD)
while np.where(quality>=tau_0)[0].shape[0] < 5: tau_0 -= 0.01

T = Y.copy()
GRB = gaussian_radial_basis(Y, beta)
A = np.zeros([M, 2])
sigma2 = init_sigma2(X, Y)
tau = tau_0

Pm = None

Q = 0
dQ = float('Inf')
iter = 1

while iter <= 100 and abs(dQ) > tolerance and sigma2 > 1e-4:
    T_old = T.copy()
    Q_old = Q

    # refine
    if (iter-1) % freq == 0:
        C = C_all[np.where(quality >= tau)]
        L = np.zeros_like(PD)
        L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
        L = L/L.max()
        L[np.where(L==0.0)] = 1.0

        C = lapjv(L)[1]
        Pm = np.ones_like(PD) * (1.0-epsilon)/N
        Pm[np.arange(C.shape[0]), C] = 1.0
        Pm = Pm / np.sum(Pm, axis=0)

        tau = tau - delta
        if tau < 1.1: tau = 1.1

        # plt.scatter(T[:, 0], T[:, 1])
        # plt.show()

    # compute minimization
    Po, P1, Np, tmp, Q = compute(X, Y, T_old, Pm, sigma2, omega)
    Q = Q + lambd / 2 * np.trace(np.dot(np.dot(A.transpose(), GRB), A))

    # update variables
    dP = np.diag(P1)
    t1 = np.dot(dP, GRB) + lambd * sigma2 * np.eye(M)
    t2 = np.dot(Po, X) - np.dot(dP, Y)
    A = np.dot(np.linalg.inv(t1), t2)
    sigma2 = tmp / (2.0*Np)
    omega = 1 - (Np/N)
    if omega > 0.99: omega = 0.99
    if omega < 0.01: omega = 0.01
    T = Y + np.dot(GRB, A)
    lambd = lambd * 0.95
    if lambd < 0.1: lambd = 0.1

    dQ = Q - Q_old
    iter = iter + 1

    print(iter, Q, tau)

plt.scatter(T[:, 0], T[:, 1])
plt.show()





    
