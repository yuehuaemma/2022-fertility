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

from VFI import getgrids
from simulation import getmom1, getmom2, getmom3, getmom4, getmom5, getmom6, getallmoms

extpar, polpar, _, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))
par = pickle.load(open('../vars/par1_est.pickle', 'rb'))
hs_coeffs = pickle.load(open('../vars/hs_coeffs_est.pickle', 'rb'))

Peta, Peps, etagrid, epsgrid, \
icgrid, isgrid, hgrid,\
asgrid, apgrid,\
zsgrid, \
zphatgrid, zphat2grid,\
zmgrid, zmhatgrid = getgrids(par, polpar)

transabmat, \
transpmat, sims,\
tranpabmat, tranpbcmat, tranpacmat,\
tranpmmat, simpb, simpc, \
tranmsmat, simm, \
tranallmat, \
erglam, ergdistall, ergdistsa, ergdistsb, ergdistpa, ergdistpb, ergdistpc, ergdistmb, \
cldistsb, cldistpa, cldistpb, cldistpc, cldistm, mcldistsa, mcldistsb, cmcldistsb, \
negnwdist, mnegnwdist = pickle.load(open('../vars/simulation.pickle', 'rb'))

allmoms = getallmoms(ergdistsb, ergdistpc, \
               cldistsb, cldistpc, mcldistsb, cmcldistsb, \
               negnwdist, \
               sims, simpc, \
               etagrid, par, polpar, hs_coeffs)

pickle.dump(allmoms, open('../vars/allmoms.pickle', 'wb'))
