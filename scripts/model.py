import numpy as np
from scipy.interpolate import interp1d

def gauss(x, mu, sig, n):
    return n*np.exp(-(x-mu)**2/(2*sig**2))

def convolve_lsf(sig, lsf):
    a = 2*np.sqrt(2*np.log(2))
    return np.sqrt(sig**2+(lsf/a)**2)

class GaussSingleLine():
    def __init__(self, lam0, lsf):
        self.lam0 = lam0
        self.lsf = lsf  

    def model(self, x, z, sig, n):
        return gauss(x, self.lam0*(1. + z), 
            convolve_lsf(sig, self.lsf)/2.998e5*self.lam0*(1. + z), n)

class O3_1comp():
    def __init__(self, lam0, lsf):
        self.lam0 = lam0
        self.lsf = lsf

    def model(self, x, z1, sig1, n1):
        # note that n1 parameter is the amp. of red component (5008A)
        n1_blue = n1/3.
        g1_blue = gauss(x, self.lam0[0]*(1. + z1), 
            convolve_lsf(sig1, self.lsf[0])/2.998e5*self.lam0[0]*(1. + z1),
            n1_blue)
        g1_red = gauss(x, self.lam0[1]*(1. + z1), 
            convolve_lsf(sig1, self.lsf[1])/2.998e5*self.lam0[1]*(1. + z1), 
            n1)
        return g1_blue + g1_red

    def model_display(self, x, z1, sig1, n1):
        return self.model(x, z1, sig1, n1)