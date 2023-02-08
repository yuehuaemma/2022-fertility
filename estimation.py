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

from VFI import initdEVs, getgrids, VFIsolve

from simulation import gettransmats, genergdist, \
                        getmom1, getcldist, getmom2, getmom3, \
                        genapqdist, gennegnwdist, gennegnwdistpb,\
                        getmom4, getmom5, getmom6, \
                        getallmoms

extpar, polpar, _, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))
init_dEVsgrid = pickle.load(open('../vars/init_dEVsgrid.pickle', 'rb'))

lb = np.array([1., .2, 0., 1., 1., 1., 0., -2., 0., 0., 0., 1e-4, 0., 0.5, -0.1])
ub = np.array([2.5, 2., 1., 2., 3., 10., 1., 1., 1., 1.1, 3., 1e-1, 1., 1., 0.2])

lb_hs = np.array([-2., 0., -1])
ub_hs = np.array([0., 10., 1])

def estpar_to_estparhat(par):
    
    parhat = log(1/(1-(par-lb)/(ub-lb)) - 1)
    
    return parhat


def estparhat_to_estpar(parhat):
    
    par = (ub-lb)*(1-1/(exp(parhat)+1))+lb
    
    return par


def smalllossf(estparhat_hs, cldistpc, negnwdist, simpc, dataM):
    
    hs_coeffs = (ub_hs-lb_hs)*(1-1/(exp(estparhat_hs)+1))+lb_hs
    
    smallhatM = getmom5(cldistpc, negnwdist, simpc, hs_coeffs)
    smalldataM = dataM[15:19]
    
    I = np.eye(len(smalldataM))
    
    err = ((smallhatM - smalldataM) @ I) @ (smallhatM - smalldataM)
    
    return err


def lossf(parhat, dataM, init_hs_coeffs, config):
    
    par = estparhat_to_estpar(parhat)
    
    Peta, Peps, etagrid, epsgrid, \
    icgrid, isgrid, hgrid,\
    asgrid, apgrid, \
    zsgrid, \
    zphatgrid, zphat2grid,\
    zmgrid, zmhatgrid = getgrids(par, polpar)
    
    dEVsgrid, \
    solveVmgrid,\
    solvetilVpgrid,\
    pngrid,\
    solvetilVsgrid,\
    colgrid = VFIsolve(par, polpar, init_dEVsgrid)
    
    transabmat, \
    transpmat, sims,\
    tranpabmat, tranpbcmat, tranpacmat,\
    tranpmmat, simpb, simpc, \
    tranmsmat, simm, \
    tranallmat = gettransmats(solveVmgrid,\
                     solvetilVpgrid,\
                     pngrid,\
                     solvetilVsgrid,\
                     colgrid,\
                     par, polpar)
    
    erglam, ergdistall, ergdistsa, ergdistsb, ergdistpa, ergdistpb, ergdistpc, ergdistmb \
        = genergdist(tranallmat, transabmat, transpmat, tranpabmat, tranpbcmat, tranpmmat, tranmsmat)

    cldistsb, cldistpa, cldistpb, cldistpc, cldistm, mcldistsa, mcldistsb, cmcldistsb \
        = getcldist(ergdistsb, sims, transpmat, tranpabmat, tranpacmat, tranpmmat, tranmsmat, transabmat)

    negnwdist, mnegnwdist = gennegnwdist(ergdistsb, sims, transpmat, tranpacmat, tranpmmat, tranmsmat, transabmat)

    estparhat_hs = log(1/(1-(init_hs_coeffs-lb_hs)/(ub_hs-lb_hs)) - 1)
    
    output = minimize(smalllossf, estparhat_hs, args = (cldistpc, negnwdist, simpc, dataM), method='Nelder-Mead')
    
    hs_coeffs = (ub_hs-lb_hs)*(1-1/(exp(output.x)+1))+lb_hs
    
    hatM = getallmoms(ergdistsb, ergdistpc, \
               cldistsb, cldistpc, mcldistsb, cmcldistsb, \
               negnwdist, \
               sims, simpc, \
               etagrid, par, polpar, hs_coeffs)
    
    if config == 0:
        tmp = np.ones(len(hatM))
    elif config == 1:
        tmp = np.ones(len(hatM))
        tmp[5:11] = 1e-1
    elif config == 2:
        tmp = 1e-1*np.ones(len(hatM))
        tmp[5:11] = 1
    
    W = np.diag(tmp)
    tmp = dataM
    err = ((hatM - tmp) @ W) @ (hatM - tmp)

    print(err)
    print(par)
    
    return err, hs_coeffs