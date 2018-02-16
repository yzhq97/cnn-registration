from __future__ import print_function

import numpy as np
import tensorflow as tf
from VGG16mo import VGG16mo
from utils.utils import *
import cv2
from lap import lapjv
import matplotlib.pyplot as plt

class CNN(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224, 224])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 50

        self.tolerance = 1e-2
        self.freq = 5
        self.delta = 0.05
        self.epsilon = 0.4
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.SIFT = cv2.xfeatures2d.SIFT_create()

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.VGG = VGG16mo()
        self.VGG.build(self.cnnph)

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance = 1e-2
        freq = self.freq = 5
        delta = self.delta = 0.05
        epsilon = self.epsilon = 0.4
        omega = self.omega = 0.5
        beta = self.beta = 2.0
        lambd = self.lambd = 0.5

        # resize image
        IX = cv2.resize(IX, (self.width, self.height))
        scale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IY = cv2.resize(IY, (self.width, self.height))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D1, D2, D3, D4 = sess.run([
                self.VGG.kconv3_1, self.VGG.kconv3_3, self.VGG.kconv4_3, self.VGG.conv5_1
            ], feed_dict=feed_dict)

        D1 = (D1 - np.mean(D1)) / np.std(D1)
        D2 = (D2 - np.mean(D2)) / np.std(D2)
        D3 = (D3 - np.mean(D3)) / np.std(D3)
        D4 = (D4 - np.mean(D4)) / np.std(D4)

        D = np.concatenate([D3, D4], axis=3)

        # generate combined feature

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 16 + 8
        Y = np.array(seq, dtype='float32') * 16 + 8
        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # normalize

        X = X / 224.0
        Y = Y / 224.0

        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)
        tau = 5.0
        while np.where(quality >= tau)[0].shape[0] < 5: tau -= 0.01

        Pm = None

        Q = 0
        dQ = float('Inf')
        itr = 1

        # registration process
        while itr < self.max_itr and abs(dQ) > tolerance and sigma2 > 1e-4:
            T_old = T.copy()
            Q_old = Q

            # refine
            if (itr - 1) % freq == 0:
                C = C_all[np.where(quality >= tau)]
                L = np.zeros_like(PD)
                L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
                L = L / L.max()
                L[np.where(L == 0.0)] = 1.0

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
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
            sigma2 = tmp / (2.0 * Np)
            omega = 1 - (Np / N)
            if omega > 0.99: omega = 0.99
            if omega < 0.01: omega = 0.01
            T = Y + np.dot(GRB, A)
            lambd = lambd * 0.95
            if lambd < 0.1: lambd = 0.1

            dQ = Q - Q_old
            itr = itr + 1

            print(itr, Q, tau)

        return Y * scale * 224.0, T * scale * 224.0


class SIFT(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224, 224])

        self.max_itr = 50

        self.tolerance = 1e-2
        self.freq = 5
        self.delta = 0.05
        self.epsilon = 0.4
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.SIFT = cv2.xfeatures2d.SIFT_create()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance = 1e-2
        freq = self.freq = 5
        delta = self.delta = 0.05
        epsilon = self.epsilon = 0.4
        omega = self.omega = 0.5
        beta = self.beta = 2.0
        lambd = self.lambd = 0.5

        # SIFT
        IX = cv2.resize(IX, (self.width, self.height))
        scale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IY = cv2.resize(IY, (self.width, self.height))
        IX_gray = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
        IY_gray = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

        XS, DXS = self.SIFT.detectAndCompute(IX_gray, None)
        YS, DYS = self.SIFT.detectAndCompute(IY_gray, None)
        XS = np.array([kp.pt for kp in XS])
        YS = np.array([kp.pt for kp in YS])

        # select points
        PD = pairwise_distance(DXS, DYS)
        C_all, quality = match(PD)
        C = C_all[np.where(quality >= 1.5)]
        X = XS[C[:, 0], :]
        Y = YS[C[:, 1], :]
        DX = DXS[C[:, 0], :]
        DY = DYS[C[:, 1], :]

        N = X.shape[0]
        M = Y.shape[0]
        assert M==N

        # prepare feature
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        # normalize

        X = X / 224.0
        Y = Y / 224.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)
        tau = 5.0
        while np.where(quality >= tau)[0].shape[0] < 5: tau -= 0.01

        Pm = None

        Q = 0
        dQ = float('Inf')
        itr = 1

        # registration process
        while itr < self.max_itr and abs(dQ) > tolerance and sigma2 > 1e-4:
            T_old = T.copy()
            Q_old = Q

            # refine
            if (itr - 1) % freq == 0:
                C = C_all[np.where(quality >= tau)]
                L = np.zeros_like(PD)
                L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
                L = L / L.max()
                L[np.where(L == 0.0)] = 1.0

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < 1.25: tau = 1.25

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
            sigma2 = tmp / (2.0 * Np)
            omega = 1 - (Np / N)
            if omega > 0.99: omega = 0.99
            if omega < 0.01: omega = 0.01
            T = Y + np.dot(GRB, A)
            lambd = lambd * 0.95
            if lambd < 0.1: lambd = 0.1

            dQ = Q - Q_old
            itr = itr + 1

            print(itr, Q, tau)

        return Y * scale * 224.0, T * scale * 224.0

class Combined(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224, 224])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 50

        self.tolerance = 1e-2
        self.freq = 5
        self.delta = 0.05
        self.epsilon = 0.4
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.SIFT = cv2.xfeatures2d.SIFT_create()

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.VGG = VGG16mo()
        self.VGG.build(self.cnnph)

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance = 1e-2
        freq = self.freq = 5
        delta = self.delta = 0.05
        epsilon = self.epsilon = 0.4
        omega = self.omega = 0.5
        beta = self.beta = 2.0
        lambd = self.lambd = 0.5

        # SIFT
        IX = cv2.resize(IX, (self.width, self.height))
        scale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IY = cv2.resize(IY, (self.width, self.height))
        IX_gray = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
        IY_gray = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

        XS, DXS = self.SIFT.detectAndCompute(IX_gray, None)
        YS, DYS = self.SIFT.detectAndCompute(IY_gray, None)
        XS = np.array([kp.pt for kp in XS])
        YS = np.array([kp.pt for kp in YS])

        DXS = (DXS - np.mean(DXS)) / np.std(DXS)
        DYS = (DYS - np.mean(DYS)) / np.std(DYS)

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
        DX4 = D4[0, seq[:, 0], seq[:, 1]]
        DY4 = D4[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 16 + 8
        Y = np.array(seq, dtype='float32') * 16 + 8
        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        X, DXS = convert_feature(X, XS, DXS)
        Y, DYS = convert_feature(Y, YS, DYS)

        DX = np.concatenate([self.cnn_weight * DX4, self.sift_weight * DXS], axis=1)
        DY = np.concatenate([self.cnn_weight * DY4, self.sift_weight * DYS], axis=1)

        # normalize

        X = X / 224.0
        Y = Y / 224.0

        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)
        tau = 5.0
        while np.where(quality >= tau)[0].shape[0] < 5: tau -= 0.01

        Pm = None

        Q = 0
        dQ = float('Inf')
        itr = 1

        # registration process
        while itr < self.max_itr and abs(dQ) > tolerance and sigma2 > 1e-4:
            T_old = T.copy()
            Q_old = Q

            # refine
            if (itr - 1) % freq == 0:
                C = C_all[np.where(quality >= tau)]
                L = np.zeros_like(PD)
                L[C[:, 0], C[:, 1]] = PD[C[:, 0], C[:, 1]]
                L = L / L.max()
                L[np.where(L == 0.0)] = 1.0

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
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
            sigma2 = tmp / (2.0 * Np)
            omega = 1 - (Np / N)
            if omega > 0.99: omega = 0.99
            if omega < 0.01: omega = 0.01
            T = Y + np.dot(GRB, A)
            lambd = lambd * 0.95
            if lambd < 0.1: lambd = 0.1

            dQ = Q - Q_old
            itr = itr + 1

            print(itr, Q, tau)

        return Y * scale * 224.0, T * scale * 224.0
