import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import math
import pickle

from numpy import exp, log
from math import erfc, sqrt
from numba import jit, njit

@njit
def std_norm_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))

@njit
def tauchen(x, n, rho, sig, half_step):
    
    P = np.zeros((n, n))
    
    for i in range(n):
        P[i, 0] = std_norm_cdf((x[0] - rho * x[i] + half_step) / sig)
        P[i, n - 1] = 1 -             std_norm_cdf((x[n - 1] - rho * x[i] - half_step) / sig)
        for j in range(1, n - 1):
            z = x[j] - rho * x[i]
            P[i, j] = (std_norm_cdf((z + half_step) / sig) -
                       std_norm_cdf((z - half_step) / sig))
            
    return P


@njit("f8(f8, f8, f8[:], f8[:], f8[:, :])")
def interp2d_lin(x, y, xgrid, ygrid, zgrid):
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    findygrid = y < ygrid
    
    if findygrid[-1] == 0:
        y1i = len(ygrid)-1
        y0i = y1i - 1
    elif findygrid[0] == 1:
        y0i = 0
        y1i = y0i + 1
    else:
        y1i = np.argmax(findygrid)
        y0i = y1i - 1
    y1 = ygrid[y1i]
    y0 = ygrid[y0i]
        
    z = (x1-x)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[x0i, y0i] +         (x-x0)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[x1i, y0i] +         (x1-x)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[x0i, y1i] +         (x-x0)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[x1i, y1i]
    
    return z


# In[19]:


@njit("f8(f8, f8, f8[:], f8[:], f8[:, :])")
def interp2d_exp(x, y, xgrid, ygrid, zgrid):
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    findygrid = y < ygrid
    
    if findygrid[-1] == 0:
        y1i = len(ygrid)-1
        y0i = y1i - 1
    elif findygrid[0] == 1:
        y0i = 0
        y1i = y0i + 1
    else:
        y1i = np.argmax(findygrid)
        y0i = y1i - 1
    y1 = ygrid[y1i]
    y0 = ygrid[y0i]
        
    z = exp((x1-x)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[x0i, y0i]) +         (x-x0)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[x1i, y0i]) +         (x1-x)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[x0i, y1i]) +         (x-x0)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[x1i, y1i]))
    
    return z


# In[20]:


@njit("f8(f8, f8, f8, f8[:], f8[:], f8[:], f8[:, :, :])")
def interp3d_lin(w, x, y, wgrid, xgrid, ygrid, zgrid):
    
    findwgrid = w < wgrid
    
    if findwgrid[-1] == 0:
        w1i = len(wgrid)-1
        w0i = w1i - 1
    elif findwgrid[0] == 1:
        w0i = 0
        w1i = w0i + 1
    else:
        w1i = np.argmax(findwgrid)
        w0i = w1i - 1
    w1 = wgrid[w1i]
    w0 = wgrid[w0i]
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    findygrid = y < ygrid
    
    if findygrid[-1] == 0:
        y1i = len(ygrid)-1
        y0i = y1i - 1
    elif findygrid[0] == 1:
        y0i = 0
        y1i = y0i + 1
    else:
        y1i = np.argmax(findygrid)
        y0i = y1i - 1
    y1 = ygrid[y1i]
    y0 = ygrid[y0i]
        
    z = (w1-w)/(w1-w0)*(x1-x)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[w0i, x0i, y0i] +         (w-w0)/(w1-w0)*(x1-x)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[w1i, x0i, y0i] +         (w1-w)/(w1-w0)*(x-x0)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[w0i, x1i, y0i] +         (w-w0)/(w1-w0)*(x-x0)/(x1-x0)*(y1-y)/(y1-y0)*zgrid[w1i, x1i, y0i] +         (w1-w)/(w1-w0)*(x1-x)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[w0i, x0i, y1i] +         (w-w0)/(w1-w0)*(x1-x)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[w1i, x0i, y1i] +         (w1-w)/(w1-w0)*(x-x0)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[w0i, x1i, y1i] +         (w-w0)/(w1-w0)*(x-x0)/(x1-x0)*(y-y0)/(y1-y0)*zgrid[w1i, x1i, y1i]
    
    return z


@njit("f8(f8, f8, f8, f8[:], f8[:], f8[:], f8[:, :, :])")
def interp3d_exp(w, x, y, wgrid, xgrid, ygrid, zgrid):
    
    findwgrid = w < wgrid
    
    if findwgrid[-1] == 0:
        w1i = len(wgrid)-1
        w0i = w1i - 1
    elif findwgrid[0] == 1:
        w0i = 0
        w1i = w0i + 1
    else:
        w1i = np.argmax(findwgrid)
        w0i = w1i - 1
    w1 = wgrid[w1i]
    w0 = wgrid[w0i]
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    findygrid = y < ygrid
    
    if findygrid[-1] == 0:
        y1i = len(ygrid)-1
        y0i = y1i - 1
    elif findygrid[0] == 1:
        y0i = 0
        y1i = y0i + 1
    else:
        y1i = np.argmax(findygrid)
        y0i = y1i - 1
    y1 = ygrid[y1i]
    y0 = ygrid[y0i]
        
    z = exp((w1-w)/(w1-w0)*(x1-x)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[w0i, x0i, y0i]) +             (w-w0)/(w1-w0)*(x1-x)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[w1i, x0i, y0i]) +             (w1-w)/(w1-w0)*(x-x0)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[w0i, x1i, y0i]) +             (w-w0)/(w1-w0)*(x-x0)/(x1-x0)*(y1-y)/(y1-y0)*log(zgrid[w1i, x1i, y0i]) +             (w1-w)/(w1-w0)*(x1-x)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[w0i, x0i, y1i]) +             (w-w0)/(w1-w0)*(x1-x)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[w1i, x0i, y1i]) +             (w1-w)/(w1-w0)*(x-x0)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[w0i, x1i, y1i]) +             (w-w0)/(w1-w0)*(x-x0)/(x1-x0)*(y-y0)/(y1-y0)*log(zgrid[w1i, x1i, y1i]))
    
    return z


@njit("f8(f8, f8[:], f8[:])")
def interp_lin(x, xgrid, ygrid):
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    y = (x1-x)/(x1-x0)*ygrid[x0i] +         (x-x0)/(x1-x0)*ygrid[x1i]
    
    if np.isnan(y):
        raise ValueError("interp_lin: y is nan")
    
    return y


@njit("f8(f8, f8[:], f8[:])")
def interp_exp(x, xgrid, ygrid):
    
    findxgrid = x < xgrid
    
    if findxgrid[-1] == 0:
        x1i = len(xgrid)-1
        x0i = x1i - 1
    elif findxgrid[0] == 1:
        x0i = 0
        x1i = x0i + 1
    else:
        x1i = np.argmax(findxgrid)
        x0i = x1i - 1
    x1 = xgrid[x1i]
    x0 = xgrid[x0i]
        
    y = exp((x1-x)/(x1-x0)*log(ygrid[x0i]) +         (x-x0)/(x1-x0)*log(ygrid[x1i]))
    
    if np.isnan(y):
        raise ValueError("interp_exp: y is nan")
    
    return y


# In[24]:


@njit
def getavggrid(somegrid):
    N = len(somegrid)
    avggrid = np.zeros(N-1)
    
    for i in range(N-1):
        avggrid[i] = (somegrid[i] + somegrid[i+1])/2
        
    return avggrid


@njit
def wtd_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    
    wts = weights/np.sum(weights)
    
#     average = np.average(values, weights=weights)
    average = values @ wts
    # Fast and numerically precise:
#     variance = np.average((values-average)**2, weights=weights)
    variance = (values-average)**2 @ wts
    return (average, variance**0.5)


@njit
def wtd_cov(values1, values2, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    
    wts = weights/np.sum(weights)
    
#     average = np.average(values, weights=weights)
    average1 = values1 @ wts
    average2 = values2 @ wts
    # Fast and numerically precise:
#     variance = np.average((values-average)**2, weights=weights)
    cov = ((values1-average1)*(values2-average2)) @ wts
    return cov


@njit
def weighted_quantile(values, quantiles, sample_weight):
    
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    
    return np.interp(quantiles, weighted_quantiles, values)