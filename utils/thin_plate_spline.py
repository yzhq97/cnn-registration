import numpy as np
import scipy as sp

class ThinPlateSpline2D(object):
    def __init__(self):
        pass

    def U(self, r):
        rsq = r**2.0
        if rsq == 0.0: return 0.0
        val = rsq*np.log(rsq)
        if np.isnan(val): return 0.0
        return val

    def fit(self, xs, ys, hs):
        self.xs = xs = np.array(xs).ravel()
        self.ys = ys = np.array(ys).ravel()
        self.hs = hs = np.array(hs).ravel()
        N = len(xs)
        P = np.vstack([np.ones_like(xs), xs, ys]).T
        K = np.zeros((N, N), np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                K[j, i] = K[i, j] = self.U(np.linalg.norm(P[i] - P[j]))
        L = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((P.shape[1], P.shape[1]))]),
            ])
        Y = np.hstack([hs, np.zeros(3)]).T
        # L * (W | a0 a1 a2) = Y
        W_a0_a1_a2 = sp.linalg.solve(L, Y)
        self.W = W_a0_a1_a2[:N]
        self.a_s = W_a0_a1_a2[N:]
        self.P = P
        self.L = L

    def interpolate(self, x_or_xs, y_or_ys):
        a0, a1, a2 = self.a_s[0], self.a_s[1], self.a_s[2]
        x_or_xs = np.array(x_or_xs).ravel().astype(np.float32)
        y_or_ys = np.array(y_or_ys).ravel().astype(np.float32)
        res = a0 + a1*x_or_xs + a2*y_or_ys
        N = len(self.xs)
        P_input = np.vstack([np.ones_like(x_or_xs), x_or_xs, y_or_ys]).T
        for i in range(N):
            if len(P_input) > 1:
                for j, row in enumerate(P_input):
                    res[j] += self.W[i]*self.U(np.linalg.norm(self.P[i] - row))
            else:
                res += self.W[i]*self.U(np.linalg.norm(self.P[i] - P_input))
        return res