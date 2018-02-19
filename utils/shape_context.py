from numpy import *
import math

seterr(all='ignore')


def logspace(d1, d2, n):
    sp = [(10 ** (d1 + k * (d2 - d1) / (n - 1))) for k in xrange(0, n - 1)]
    sp.append(10 ** d2)
    return sp


def euclid_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def get_angle(p1, p2):
    """Return angle in radians"""
    return math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))


class ShapeContext(object):
    HUNGURIAN = 1

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nbins = nbins_theta * nbins_r

    def _dist2(self, x, c):
        result = zeros((len(x), len(c)))
        for i in xrange(len(x)):
            for j in xrange(len(c)):
                result[i, j] = euclid_distance(x[i], c[j])
        return result

    def _get_angles(self, x):
        result = zeros((len(x), len(x)))
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                result[i, j] = get_angle(x[i], x[j])
        return result

    def get_mean(self, matrix):
        """ This is not working. Should delete this and make something better"""
        h, w = matrix.shape
        mean_vector = matrix.mean(1)
        mean = mean_vector.mean()

        return mean

    def compute(self, points, r=None):
        r_array = self._dist2(points, points)
        mean_dist = r_array.mean()
        r_array_n = r_array / mean_dist

        r_bin_edges = logspace(log10(self.r_inner), log10(self.r_outer), self.nbins_r)

        r_array_q = zeros((len(points), len(points)), dtype=int)
        for m in xrange(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        theta_array = self._get_angles(points)
        # 2Pi shifted
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        theta_array_q = int_(1 + floor(theta_array_2 / (2 * math.pi / self.nbins_theta)))
        # norming by mass(mean) angle v.0.1 ############################################
        # By Andrey Nikishaev
        # theta_array_delta = theta_array - theta_array.mean()
        # theta_array_delta_2 = theta_array_delta + 2*math.pi * (theta_array_delta < 0)
        # theta_array_q = 1 + floor(theta_array_delta_2 /(2 * math.pi / self.nbins_theta))
        ################################################################################

        BH = zeros((len(points), self.nbins))
        for i in xrange(len(points)):
            sn = zeros((self.nbins_r, self.nbins_theta))
            for j in xrange(len(points)):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            BH[i] = sn.reshape(self.nbins)

        return BH

    def _cost(self, hi, hj):
        cost = 0
        for k in xrange(self.nbins):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k]) ** 2) / (hi[k] + hj[k])

        return cost * 0.5

    def cost(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = zeros((p, p2))
        for i in xrange(p):
            for j in xrange(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)

        return C

        return result