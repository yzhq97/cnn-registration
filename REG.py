from __future__ import print_function
import time
import gc

import numpy as np
import tensorflow as tf
from VGG16mo import VGG16mo
from utils.utils import *
import cv2
from lap import lapjv
from utils.shape_context import ShapeContext
import matplotlib.pyplot as plt

class CNN(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.pool4
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 16.0 + 8.0
        Y = np.array(seq, dtype='float32') * 16.0 + 8.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        print('mean quality: %f' % np.mean(quality))

        tau_max = np.max(quality)
        print(tau_max)
        while np.where(quality >= tau_max)[0].shape[0] <= 65: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 33: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


class SIFT(object):
    def __init__(self):
        self.max_itr = 250

        self.tolerance = 1e-3
        self.freq = 5
        self.epsilon = 0.7
        self.omega = 0.3
        self.beta = 2.0
        self.lambd = 0.5

        self.SIFT = cv2.xfeatures2d.SIFT_create()
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # SIFT
        start_time = time.time()
        IX_gray = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
        IY_gray = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

        XS0, DXS0 = self.SIFT.detectAndCompute(IX_gray, None)
        YS0, DYS0 = self.SIFT.detectAndCompute(IY_gray, None)
        XSPD = [(XS0[i], DXS0[i]) for i in range(len(XS0)) if XS0[i].response > 0.025]
        YSPD = [(YS0[i], DYS0[i]) for i in range(len(YS0)) if YS0[i].response > 0.025]
        XSPD.sort(key=lambda p: p[0].response, reverse=True)
        YSPD.sort(key=lambda p: p[0].response, reverse=True)
        XSPD = XSPD[:800]
        YSPD = YSPD[:800]

        XS = [t[0].pt for t in XSPD]
        YS = [t[0].pt for t in YSPD]
        DXS = [t[1] for t in XSPD]
        DYS = [t[1] for t in YSPD]

        XS, YS, DXS, DYS = np.array(XS), np.array(YS), np.array(DXS), np.array(DYS)
        XS = np.array([XS[:, 1], XS[:, 0]]).transpose()
        YS = np.array([YS[:, 1], YS[:, 0]]).transpose()

        # select points
        PD = pairwise_distance(DXS, DYS)
        C_all, quality = match(PD)
        C = C_all[np.where(quality >= 1.25)]
        X, Y = XS[C[:, 1], :], YS[C[:, 0], :]
        DX, DY = DXS[C[:, 1], :], DYS[C[:, 0], :]

        #normalize
        X_shape = np.array(IX_gray.shape[:2], dtype='float32')
        Y_shape = np.array(IY_gray.shape[:2], dtype='float32')
        X_center = 0.5 * X_shape
        Y_center = 0.5 * Y_shape
        X = (X - X_center) / X_shape
        Y = (Y - Y_center) / Y_shape

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = Y.shape[0]
        assert M==N

        # prepare feature
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau = 2.5
        while tau > 1 and np.where(quality >= tau)[0].shape[0] < 0.25 * quality.shape[0]: tau -= 0.1
        delta = (tau - 1.0) / 10.0

        # registration process
        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < 1: tau = 1

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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return X * X_shape + X_center, Y * Y_shape + Y_center, T * X_shape + X_center


class CNN1(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.conv5_1
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 16.0 + 8.0
        Y = np.array(seq, dtype='float32') * 16.0 + 8.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        print('mean quality: %f' % np.mean(quality))

        tau_max = np.max(quality)
        print('tau_max: %f' % tau_max)
        while np.where(quality >= tau_max)[0].shape[0] <= 48: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 16: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


class CNN2(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.7
        self.omega = 0.3
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.pool3
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(28) for j in range(28)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 8.0 + 4.0
        Y = np.array(seq, dtype='float32') * 8.0 + 4.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        print('mean quality: %f' % np.mean(quality))

        tau_max = np.max(quality)
        print(tau_max)
        while np.where(quality >= tau_max)[0].shape[0] <= 261: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 130: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            #print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


class CNN3(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature
        start_time = time.time()

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.pool5_1
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(7) for j in range(7)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 32.0 + 16.0
        Y = np.array(seq, dtype='float32') * 32.0 + 16.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        print('mean quality: %f' % np.mean(quality))

        tau_max = np.max(quality)
        print(tau_max)
        while np.where(quality >= tau_max)[0].shape[0] <= 16: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 8: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


class CNN4(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.pool4
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 16.0 + 8.0
        Y = np.array(seq, dtype='float32') * 16.0 + 8.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 48: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 16: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


class CNN5(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature
        start_time = time.time()

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D = sess.run([
                self.vgg.pool5
            ], feed_dict=feed_dict)[0]

        # generate combined feature

        seq = np.array([[i, j] for i in range(7) for j in range(7)], dtype='int32')
        DX = D[0, seq[:, 0], seq[:, 1]]
        DY = D[1, seq[:, 0], seq[:, 1]]

        X = np.array(seq, dtype='float32') * 32.0 + 16.0
        Y = np.array(seq, dtype='float32') * 32.0 + 16.0

        # normalize

        DX = DX / np.std(DX)
        DY = DY / np.std(DY)

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)

        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 20: tau_max -= 0.001

        C = C_all[np.where(quality >= tau_max)]
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        DX, DY = DX[C[:, 1]], DY[C[:, 0]]

        SCX = self.SC.compute(X)

        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        PD = pairwise_distance(DX, DY)
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 8: tau_max -= 0.001
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale

class CNN7(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5
        self.init_thres = 1.15

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def register(self, IX, IY):

        # set parameters
        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = 1.0 * np.array(IX.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY.shape[:2]) / self.shape
        IX = cv2.resize(IX, (self.height, self.width))
        IY = cv2.resize(IY, (self.height, self.width))

        # CNN feature

        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D1, D2, D3 = sess.run([
                self.vgg.pool3, self.vgg.pool4, self.vgg.pool5_1
            ], feed_dict=feed_dict)

        DX1, DY1 = np.reshape(D1[0], [-1, 256]), np.reshape(D1[1], [-1, 256])
        DX2, DY2 = np.reshape(D2[0], [-1, 512]), np.reshape(D2[1], [-1, 512])
        DX3, DY3 = np.reshape(D3[0], [-1, 512]), np.reshape(D3[1], [-1, 512])

        del D1, D2, D3

        PD1 = pairwise_distance(DX1, DY1)
        PD2 = pd_expand(pairwise_distance(DX2, DY2), 2)
        PD3 = pd_expand(pairwise_distance(DX3, DY3), 4)
        PD = PD1 + PD2 + PD3

        del DX1, DY1, DX2, DY2, DX3, DY3, PD1, PD2, PD3

        seq = np.array([[i, j] for i in range(28) for j in range(28)], dtype='int32')

        X = np.array(seq, dtype='float32') * 8.0 + 4.0
        Y = np.array(seq, dtype='float32') * 8.0 + 4.0

        # normalize

        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        C_all, quality = match(PD)

        # tau_max = np.max(quality)
        # print(tau_max)
        # while np.where(quality >= tau_max)[0].shape[0] <= 128: tau_max -= 0.01

        C = C_all[np.where(quality >= self.init_thres)]
        cnt = C.shape[0]
        print(cnt)
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        PD = PD[np.repeat(np.reshape(C[:, 1], [cnt, 1]), cnt, axis=1),
                np.repeat(np.reshape(C[:, 0], [1, cnt]), cnt, axis=0)]


        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        # feature match precalc
        C_all, quality = match(PD)
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 0.5 * cnt: tau_max -= 0.01
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        SCX = self.SC.compute(X)
        T = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        A = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

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
                Lt = PD[C[:, 0], C[:, 1]]
                maxLt = np.max(Lt)
                if maxLt > 0: Lt = Lt / maxLt
                L = np.ones([M, N])
                L[C[:, 0], C[:, 1]] = Lt

                SCT = self.SC.compute(T)
                SC_cost = self.SC.cost(SCT, SCX)
                L = L * SC_cost

                C = lapjv(L)[1]
                Pm = np.ones_like(PD) * (1.0 - epsilon) / N
                Pm[np.arange(C.shape[0]), C] = 1.0
                Pm = Pm / np.sum(Pm, axis=0)

                tau = tau - delta
                if tau < tau_min: tau = tau_min

                # plt.gca().invert_yaxis()
                # plt.scatter(T[:, 1], T[:, 0])
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

            # print(itr, Q, tau)

        print('finish: itr %d, Q %d, tau %d' % (itr, Q, tau))
        return ((X*224.0)+112.0)*Xscale, ((Y*224.0)+112.0)*Yscale, ((T*224.0)+112.0)*Xscale


def get_reg_by_name(name):
    if name == 'pool3':
        return CNN2()
    if name == 'pool4':
        return CNN()
    if name == 'conv5_1':
        return CNN1()
    if name == 'pool5_1':
        return CNN3()
    if name == 'pool5':
        return CNN5()
    if name == 'cnn_combined':
        return CNN7()
    if name == 'sift':
        return SIFT()

names = ['pool3', 'pool4', 'pool5_1', 'cnn_combined', 'sift']