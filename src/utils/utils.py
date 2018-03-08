import numpy as np
from math import exp, log
from scipy.interpolate import Rbf

def gaussian_kernel(size, sigma=1.2):
    seq = np.array([[i, j] for i in range(size) for j in range(size)], dtype='int32')
    points = np.array(seq, dtype='float32') + 0.5
    center = np.array([0.5 * size, 0.5 * size])
    d = np.linalg.norm(points-center, axis=1)
    kern1d = 1.0/(sigma * (2*np.pi)**0.5) * np.exp(-1.0 * np.power(d, 2) / (2.0 * sigma**2))
    kern = kern1d.reshape([size, size]) / np.sum(kern1d)
    return kern

def pairwise_distance(X, Y):
    assert len(X.shape) == len(Y.shape)
    N = X.shape[0]
    M = Y.shape[0]
    # D = len(X.shape)
    res = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            res[i][j] = np.linalg.norm(X[j] - Y[i])
    return res
    # return np.linalg.norm(
    #     np.repeat(np.expand_dims(X, axis=0), M, axis=0) -
    #     np.repeat(np.expand_dims(Y, axis=1), N, axis=1),
    #     axis=D)

def gaussian_radial_basis(X, beta=2.0):
    PD = pairwise_distance(X, X)
    return np.exp(-0.5 * np.power(PD/beta, 2))

def init_sigma2(X, Y):
    N = float(X.shape[0])
    M = float(Y.shape[0])
    t1 = M * np.trace(np.dot(np.transpose(X), X))
    t2 = N * np.trace(np.dot(np.transpose(Y), Y))
    t3 = 2.0 * np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0)))
    return (t1 + t2 -t3)/(M*N*2.0)

def match(PD):
    seq = np.arange(PD.shape[0])
    amin1 = np.argmin(PD, axis=1)
    C = np.array([seq, amin1]).transpose()
    min1 = PD[seq, amin1]
    mask = np.zeros_like(PD)
    mask[seq, amin1] = 1
    masked = np.ma.masked_array(PD, mask)
    min2 = np.amin(masked, axis=1)
    return C, np.array(min2/min1)

def match_max(PD):
    seq = np.arange(PD.shape[0])
    amax1 = np.argmin(PD, axis=1)
    C = np.array([seq, amax1]).transpose()
    return C

def compute(X, Y, T_old, Pm, sigma2, omega):
    N = X.shape[0]
    M = Y.shape[0]
    T = T_old

    Te = np.repeat(np.expand_dims(T, axis=1), N, axis=1)
    Xe = np.repeat(np.expand_dims(T, axis=0), M, axis=0)
    Pmxn = (1-omega) * Pm * np.exp(
        -(1 / (2 * sigma2)) * np.sum(np.power(Xe-Te, 2), axis=2) )

    Pxn = np.sum(Pmxn, axis=0) + omega/N
    Po = Pmxn / np.repeat(np.expand_dims(Pxn, axis=0), M, axis=0)

    Np = np.dot(np.dot(np.ones([1, M]), Po), np.ones([N, 1]))[0, 0]
    P1 = np.squeeze(np.dot(Po, np.ones([N, 1])))
    Px = np.diag(np.squeeze(np.dot(Po.transpose(), np.ones([M, 1]))))
    Py = np.diag(P1)
    t1 = np.trace(np.dot(np.dot(X.transpose(), Px), X))
    t2 = np.trace(np.dot(np.dot(T.transpose(), Po), X))
    t3 = np.trace(np.dot(np.dot(T.transpose(), Py), T))
    tmp =  t1 - 2.0*t2 + t3
    Q = Np * log(sigma2) + tmp/(2.0*sigma2)
    return Po, P1, Np, tmp, Q

def pd_expand(PD, k):
    N0 = np.int(np.sqrt(PD.shape[0]))
    N1 = k*N0
    L0, L1 = N0**2, N1**2
    Cmat = np.kron(np.arange(L0).reshape([N0, N0]), np.ones([k, k], dtype='int32'))
    i = np.repeat(Cmat.reshape([L1, 1]), L1, axis=1)
    j = np.repeat(Cmat.reshape([1, L1]), L1, axis=0)
    return PD[i, j]

def tps_warp(Y, T, Y_image, out_shape):
    Y_height, Y_width = Y_image.shape[:2]
    T_height, T_width = out_shape[:2]

    i_func = Rbf(T[:, 0], T[:, 1], Y[:, 0], function='thin-plate')
    j_func = Rbf(T[:, 0], T[:, 1], Y[:, 1], function='thin-plate')

    iT, jT = np.mgrid[:T_height, :T_width]
    iT = iT.flatten()
    jT = jT.flatten()
    iY = np.int_(i_func(iT, jT))
    jY = np.int_(j_func(iT, jT))

    keep = np.logical_and(iY>=0, jY>=0)
    keep = np.logical_and(keep, iY<Y_height)
    keep = np.logical_and(keep, jY<Y_width)
    iY, jY, iT, jT = iY[keep], jY[keep], iT[keep], jT[keep]

    out_image = np.zeros(out_shape, dtype='uint8')
    out_image[iT, jT, :] = Y_image[iY, jY, :]

    return out_image

def checkboard(I1, I2, n=7):
    assert I1.shape == I2.shape
    height, width, channels = I1.shape
    hi, wi = height/n, width/n
    outshape = (hi*n, wi*n, channels)

    out_image = np.zeros(outshape, dtype='uint8')
    for i in range(n):
        h = hi * i
        h1 = h + hi
        for j in range(n):
            w = wi * j
            w1 = w + wi
            if (i-j)%2 == 0:
                out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]

    return out_image



