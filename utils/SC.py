from numpy import *
from scipy.interpolate import Rbf,InterpolatedUnivariateSpline,interp1d
import math
# Hungurian algorithm implementation
import munkres
from utils import get_points_from_img,get_elements,bookenstain
import time
import heapq
import cv
import sys

seterr(all='ignore')

def logspace(d1, d2, n):
    sp =  [( 10 **(d1 + k * (d2-d1)/(n - 1)))   for k in xrange(0, n -1)]
    sp.append(10 ** d2)
    return sp
    
def euclid_distance(p1,p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )
    
    
def get_angle(p1,p2):
    """Return angle in radians"""
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))
    
    
class SC(object):

    HUNGURIAN = 1

    def __init__(self,nbins_r=5,nbins_theta=12,r_inner=0.1250,r_outer=2.0):
        self.nbins_r        = nbins_r
        self.nbins_theta    = nbins_theta
        self.r_inner        = r_inner
        self.r_outer        = r_outer
        self.nbins          = nbins_theta*nbins_r


    def _dist2(self, x, c):
        result = zeros((len(x), len(c)))
        for i in xrange(len(x)):
            for j in xrange(len(c)):
                result[i,j] = euclid_distance(x[i],c[j])
        return result
        
        
    def _get_angles(self, x):
        result = zeros((len(x), len(x)))
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                result[i,j] = get_angle(x[i],x[j])
        return result
        
    
    def get_mean(self,matrix):
        """ This is not working. Should delete this and make something better"""
        h,w = matrix.shape
        mean_vector = matrix.mean(1)
        mean = mean_vector.mean()
        
        return mean

        
    def compute(self,points,r=None):
        t = time.time()
        r_array = self._dist2(points,points)
        mean_dist = r_array.mean()
        r_array_n = r_array / mean_dist
        
        r_bin_edges = logspace(log10(self.r_inner),log10(self.r_outer),self.nbins_r)  

        r_array_q = zeros((len(points),len(points)), dtype=int)
        for m in xrange(self.nbins_r):
           r_array_q +=  (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0
        
        theta_array = self._get_angles(points)
        # 2Pi shifted
        theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
        theta_array_q = 1 + floor(theta_array_2 /(2 * math.pi / self.nbins_theta))
        # norming by mass(mean) angle v.0.1 ############################################
        # By Andrey Nikishaev
        #theta_array_delta = theta_array - theta_array.mean()
        #theta_array_delta_2 = theta_array_delta + 2*math.pi * (theta_array_delta < 0)
        #theta_array_q = 1 + floor(theta_array_delta_2 /(2 * math.pi / self.nbins_theta))
        ################################################################################

        
        BH = zeros((len(points),self.nbins))
        for i in xrange(len(points)):
            sn = zeros((self.nbins_r, self.nbins_theta))
            for j in xrange(len(points)):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            BH[i] = sn.reshape(self.nbins)
            
        print 'PROFILE TOTAL COST: ' + str(time.time()-t)     
            
        return BH        
        
        
    def _cost(self,hi,hj):
        cost = 0
        for k in xrange(self.nbins):
            if (hi[k] + hj[k]):
                cost += ( (hi[k] - hj[k])**2 ) / ( hi[k] + hj[k] )
            
        return cost*0.5
        
    
    def cost(self,P,Q,qlength=None):
        p,_ = P.shape
        p2,_ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = zeros((p,p2))
        for i in xrange(p):
            for j in xrange(p2):
                C[i,j] = self._cost(Q[j]/d,P[i]/p)    
        
        return C
        
    def __hungurian_method(self,C):
        t = time.time()
        m = munkres.Munkres()
        indexes = m.compute(C.tolist())
        total = 0
        for row, column in indexes:
            value = C[row][column]
            total += value
        print 'PROFILE HUNGURIAN ALGORITHM: ' + str(time.time()-t)     

        return total,indexes


    def quick_diff(self,P,Qs,method=HUNGURIAN):
        """
            Samplered fast shape context
        """
        res = []
        
        p,_ = P.shape
        q,_ = Qs.shape
        for i in xrange(p):
            for j in xrange(q):
                heapq.heappush(res,(self._cost(P[i],Qs[j]),i) )
        
        data = zeros((q,self.nbins))
        for i in xrange(q):
            data[i] = P[heapq.heappop(res)[1]]
       
        return self.diff(data,Qs)
        
        
    def diff(self,P,Q,qlength=None,method=HUNGURIAN):
        """
            if Q is generalized shape context then it compute shape match.
            
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
                
        """
        result = None
        C = self.cost(P,Q,qlength)

        if method == self.HUNGURIAN:
            result = self.__hungurian_method(C)
        else:
            raise Exception('No such optimization method.')
            
        return result
            

    def get_contextes(self,BH,r=5):
        res = zeros((r,self.nbins))
        used = []
        sums = []
        
        # get r shape contexts with maximum number(-BH[i]) or minimum(+BH[i]) of connected elements
        # this gives same result for same query
        for i in xrange(len(BH)):
            heapq.heappush(sums,(BH[i].sum(),i))
            
        for i in xrange(r):
            _,l = heapq.heappop(sums)
            res[i] = BH[l]
            used.append(l)
            
        del sums     

        
        return res,used

    def interpolate(self,P1,P2):
        t = time.time()
        assert len(P1)==len(P2),'Shapes has different number of points'
        x = [0]*len(P1)
        xs = [0]*len(P1)
        y = [0]*len(P1)
        ys = [0]*len(P1)
        for i in xrange(len(P1)):
            x[i]  = P1[i][0]
            xs[i] = P2[i][0]
            y[i]  = P1[i][1]
            ys[i] = P2[i][1]    
            
        def U(r):
            res = r**2 * log(r**2)
            res[r == 0] = 0
            return res
            
        SM=0.01      
        # not working without smoothenes, because of singular matrix
        fx = Rbf(x, xs, function=U,smooth=SM)
        fy = Rbf(y, ys, function=U,smooth=SM)
               
        cx,cy,E,affcost,L = bookenstain(P1,P2,15)
        
        print 'PROFILE TPS INTERPOLATION: ' + str(time.time()-t)
        
        return fx,fy,E,float(affcost)
    
