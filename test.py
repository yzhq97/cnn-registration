from __future__ import print_function

import numpy as np
from PIL import Image
from utils.utils import *
import cv2
from lap import lapjv
import matplotlib.pyplot as plt

height = 224
width = 224

tolerance = 1e-3
freq = 5
tau_0 = 1.5
delta = 0.025
epsilon = 0.4
omega = 0.5
beta = 2.0
lambd = 3.0

datadir = '../data/RemoteSense/ANGLE/68/'
# datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_name = '2.jpg'
IY_name = '7.jpg'

IX = cv2.imread(datadir+IX_name)
IY = cv2.imread(datadir+IY_name)
IX = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
IY = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

SIFT = cv2.xfeatures2d.SIFT_create()
X, DX = SIFT.detectAndCompute(IX, None)
Y, DY = SIFT.detectAndCompute(IY, None)
X = np.array([kp.pt for kp in X])
Y = np.array([kp.pt for kp in Y])

PD = pairwise_distance(DX, DY)
C_all, quality = match(PD)
while np.where(quality >= tau_0)[0].shape[0] < 5: tau_0 -= 0.01

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
    if (iter - 1) % freq == 0:
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

        plt.scatter(T[:, 0], T[:, 1])
        plt.show()

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
    iter = iter + 1

    print(Q, tau)








