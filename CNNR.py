from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from VGG16mo import VGG16mo
from utils.utils import *
import cv2
from lap import lapjv
import matplotlib.pyplot as plt

class CNNR(object):
    def __init__(self):
        self.height = 224
        self.width = 224

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.tolerance = 1e-2
        self.freq = 5
        self.tau_0 = 5.0
        self.delta = 0.05
        self.epsilon = 0.4
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.SIFT = cv2.xfeatures2d.SIFT_create()

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.VGG = VGG16mo()
        self.VGG.build(self.cnnph)

    def register(self, path1, path2):
        # SIFT
        IX = cv2.resize(cv2.imread(path1), (self.width, self.height))
        IY = cv2.resize(cv2.imread(path2), (self.width, self.height))
        IX_gray = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
        IY_gray = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

        XS0, DXS0 = self.SIFT.detectAndCompute(IX_gray, None)
        YS0, DYS0 = self.SIFT.detectAndCompute(IY_gray, None)
        XS0 = np.array([kp.pt for kp in XS0])
        YS0 = np.array([kp.pt for kp in YS0])

        DXS0 = (DXS0 - np.mean(DXS0)) / np.std(DXS0)
        DYS0 = (DYS0 - np.mean(DYS0)) / np.std(DYS0)

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)

        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D1, D2, D3, D4 = sess.run([
                self.VGG.kconv3_1, self.VGG.kconv3_3, self.VGG.kconv4_3, self.VGG.conv5_1
            ], feed_dict=feed_dict)

        D4 = (D4 - np.mean(D4)) / np.std(D4)

        # generate combined feature

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        X = np.array(seq, dtype='float32') * 32 + 16
        Y = np.array(seq, dtype='float32') * 32 + 16
        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        XS = np.zeros([14*14, 128])
        Xmap = [[] for _ in range(14*14)]
        for i in range(len(XS0)):
            Xmap[int(XS0[i, 0]/32.0)*14 + int(XS0[i, 1]/32.0)].append(i)
        for i in range(14*14):
            for j in range(len(Xmap[i])):







        # normalize

        X = X / 224.0
        Y = Y / 224.0

        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, self.beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)
        tau = 5.0
        while np.where(quality >= tau)[0].shape[0] < 5: tau -= 0.01

        Pm = None

        Q = 0
        dQ = float('Inf')
        iter = 1

        # registration process
        while iter <= 100 and abs(dQ) > self.tolerance and sigma2 > 1e-4:
            T_old = T.copy()
            Q_old = Q

            # refine
            if (iter - 1) % self.freq == 0:
                C = C_all[np.where(quality >= tau)]
                L = np.zeros_like(PD)
                L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
                L = L / L.max()
                L[np.where(L == 0.0)] = 1.0

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - self.epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - self.delta
                if tau < 1.1: tau = 1.1

                # plt.scatter(T[:, 0], T[:, 1])
                # plt.show()

            # compute minimization
            Po, P1, Np, tmp, Q = compute(X, Y, T_old, Pm, sigma2, omega)
            Q = Q + self.lambd / 2 * np.trace(np.dot(np.dot(A.transpose(), GRB), A))

            # update variables
            dP = np.diag(P1)
            t1 = np.dot(dP, GRB) + self.lambd * sigma2 * np.eye(M)
            t2 = np.dot(Po, X) - np.dot(dP, Y)
            A = np.dot(np.linalg.inv(t1), t2)
            sigma2 = tmp / (2.0 * Np)
            omega = 1 - (Np / N)
            if omega > 0.99: omega = 0.99
            if omega < 0.01: omega = 0.01
            T = Y + np.dot(GRB, A)
            lambd = self.lambd * 0.95
            if lambd < 0.1: self.lambd = 0.1

            dQ = Q - Q_old
            iter = iter + 1

            print(iter, Q, tau)
