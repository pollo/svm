from kernels import *

import numpy as np
from cvxopt.solvers import qp
from cvxopt.base import matrix

class SVM(object):
    def __init__(self, kernel_type='linear', C=1.0,
                 degree=3.0, sigma=0.7, k=1.0, coef0=0.0):
        if kernel_type=='linear':
            self.kernel = linear_kernel()
        elif kernel_type=='polynomial':
            self.kernel = polynomial_kernel(degree)
        elif kernel_type=='rbf':
            self.kernel = radial_kernel(sigma)
        elif kernel_type=='sigmoid':
            self.kernel = sigmoid_kernel(k, coef0)
        else:
            raise ValueError('Kernel '+kernel_type+' not available')

        self.epsilon = 10**-5
        self.C = C

    def train(self, samples):
        return self._solve_optimization(samples)

    def indicator(self, sample):
        return sum([sv[0]*sv[1][1]*self.kernel(sample,sv[1][0])
                    for sv in self.support_vector])

    def predict(self, sample):
        if self.indicator(sample)>0:
            return 1
        else:
            return -1

    def _build_P(self, samples):
        return [[si[1]*sj[1]*self.kernel(si[0],sj[0]) for sj in samples]
                for si in samples]

    def _solve_optimization(self, samples):
        P = self._build_P(samples)
        q = [-1.0] * len(samples)
        h = [0.0] * len(samples) + [self.C] * len(samples)
        G = np.concatenate((np.identity(len(samples)) * -1,
                            np.identity(len(samples))))
        optimized = qp(matrix(P), matrix(q), matrix(G), matrix(h))
        if optimized['status'] == 'optimal':
            alphas = list(optimized['x'])

            self.support_vector = [(alpha, samples[i])
                                   for i,alpha in enumerate(alphas)
                                   if alpha>self.epsilon]
            return True
        else:
            print "No valid separating hyperplane found"
            return False

