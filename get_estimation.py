import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import pickle
import pandas as pd

from numpy import exp, log
from math import erfc, sqrt
from numba import jit, njit, typed
from scipy.optimize import minimize

from basic import std_norm_cdf, tauchen, \
                    interp2d_lin, interp2d_exp, interp3d_lin, interp3d_exp, interp_lin, interp_exp, \
                    getavggrid, wtd_avg_and_std, wtd_cov, weighted_quantile

from fcn import getrhovec, \
                u, du, \
                Scf, dScf, Ssf, dSsf, f, fc, fs, \
                rapf, rpf, ramf, rmf

from estimation import estpar_to_estparhat, estparhat_to_estpar, lossf, smalllossf

# _, _, par0, _ = pickle.load(open('../vars/params.pickle', 'rb'))
par0 = pickle.load(open('../vars/par1_est.pickle', 'rb'))
df = pd.read_csv('../vars/datamoms.csv', header = None)
dataM = df[1].to_numpy()
init_hs_coeffs = np.array([-1., 1., 0.])
estparhat = estpar_to_estparhat(par0)


rep = 1
for it in range(rep):
    
    output = minimize(lambda x: lossf(x, dataM, init_hs_coeffs, 0)[0], estparhat, method='Nelder-Mead', \
                                                    tol = 1e-4,\
                                                    options = {'maxfev': 3*len(estparhat)})
    estparhat = output.x
    par1 = estparhat_to_estpar(estparhat)
    res, hs_coeffs = lossf(estparhat, dataM, init_hs_coeffs, 0)

    ## Store variables
    pickle.dump(par1, open('../vars/par1_est.pickle', 'wb'))
    pickle.dump(hs_coeffs, open('../vars/hs_coeffs_est.pickle', 'wb'))