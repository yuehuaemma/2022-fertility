import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import math
import pickle

from numpy import exp, log
from math import erfc, sqrt
from numba import jit, njit

extpar, _, _, _ = pickle.load(open('../vars/params.pickle', 'rb'))

sigma, cfloor, \
rpd, rps, rpu, rpb, rmd, rmb, \
betas, betap, betam, \
colfrac,\
deltavec, \
sums, sump, summ = extpar


@njit
def getrhovec(par):
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
#     rhovec = np.array([alpha0*n**alpha1 for n in range(4)])
    rhovec = np.array([alpha0*(1-exp(-alpha1*n)) for n in range(4)])
    return rhovec


@njit
def u(c):
    
    return (c+cfloor)**(1-sigma)/(1-sigma) - cfloor**(1-sigma)/(1-sigma)

@njit
def du(c):
    
    return (c+cfloor)**(-sigma)

@njit
def invu(lam):
    return lam**(-1/sigma)

@njit
def Scf(ic, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    return phic0 + phic1*ic

@njit
def dScf(ic, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    return phic1

@njit
def Ssf(isval, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    return phis0 + phis1*isval

@njit
def dSsf(isval, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    return phis1

@njit
def f(ic, isval, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    return (a*(ic + Scf(ic, polpar))**b + (1-a)*(isval+Ssf(isval, polpar))**b)**(d/b)

@njit
def fc(ic, isval, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    tmp = a*(ic + Scf(ic, polpar))**b + (1-a)*(isval+Ssf(isval, polpar))**b
    
    return d*tmp**(d/b-1)*a*(ic + Scf(ic, polpar))**(b-1)*(1+dScf(ic, polpar)) 

@njit
def fs(ic, isval, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    tmp = a*(ic + Scf(ic, polpar))**b + (1-a)*(isval+Ssf(isval, polpar))**b
    
    return d*tmp**(d/b-1)*(1-a)*(isval + Ssf(isval, polpar))**(b-1)*(1+dSsf(isval, polpar)) 


@njit
def rapf(ap, E, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    lt = 0
    
    if E == 1:
        if ap >= 0:
            rap = (1+rpd)*ap
        elif ap >= - L:
            rap = (1+rps)*ap
            lt = 1
        else:
            rap = -(1+rps)*L + (1+rpb)*(ap+L)
            lt = 2
    elif E == 0:
        if ap >= 0:
            rap = (1+rpd)*ap
        else:
            rap = (1+rpb)*ap
            lt = 2
    
    return rap, lt

@njit
def rpf(ap, E, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    if E == 1:
        if ap >= 0:
            r = rpd
        elif ap >= - L:
            r = rps
        else:
            r = rpb
    elif E == 0:
        if ap >= 0:
            r = rpd
        else:
            r = rpb

    return r

@njit
def ramf(am):
    
    if am >= 0:
        ram = (1+rmd)*am
    else:
        ram = (1+rmb)*am
    
    return ram

@njit
def rmf(am):
    
    if am >= 0:
        r = rmd
    else:
        r = rmb
    
    return r