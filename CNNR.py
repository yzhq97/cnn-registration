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

        # generate combined feature

        # normalize

        seq = np.array([[i, j] for i in range(14) for j in range(14)], dtype='int32')
        X = np.array(seq, dtype='float32') * 32 + 16
        Y = np.array(seq, dtype='float32') * 32 + 16
        N = X.shape[0]
        M = X.shape[0]
        assert M == N

        X = X / 224.0
        Y = Y / 224.0

