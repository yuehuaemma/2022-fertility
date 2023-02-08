import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import pickle

from numpy import exp, log
from math import erfc, sqrt
from numba import jit, njit, typed

from basic import std_norm_cdf, tauchen, \
                    interp2d_lin, interp2d_exp, interp3d_lin, interp3d_exp, interp_lin, interp_exp, \
                    getavggrid, wtd_avg_and_std, wtd_cov, weighted_quantile

from fcn import getrhovec, \
                u, du, \
                Scf, dScf, Ssf, dSsf, f, fc, fs, \
                rapf, rpf, ramf, rmf

from simulation import gettransmats, genergdist, \
                        getmom1, getcldist, getmom2, getmom3, \
                        genapqdist, gennegnwdist, gennegnwdistpb,\
                        getmom4, getmom5, getmom6, \
                        getallmoms

extpar, polpar, par0, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))

dEVsgrid, \
solveVmgrid,\
solvetilVpgrid,\
pngrid,\
solvetilVsgrid,\
colgrid = pickle.load(open('../vars/VFI.pickle', 'rb'))

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
                 par0, polpar)

# erglam, ergdistall, ergdistsa, ergdistsb, ergdistpa, ergdistpb, ergdistpc, ergdistmb \
# = genergdist(tranallmat, transabmat, transpmat, tranpabmat, tranpbcmat, tranpmmat, tranmsmat)

# cldistsb, cldistpa, cldistpb, cldistpc, cldistm, mcldistsa, mcldistsb, cmcldistsb \
# = getcldist(ergdistsb, sims, transpmat, tranpabmat, tranpacmat, tranpmmat, tranmsmat, transabmat)

# negnwdist, mnegnwdist = gennegnwdist(ergdistsb, sims, transpmat, tranpacmat, tranpmmat, tranmsmat, transabmat)

# pickle.dump([transabmat, \
#             transpmat, sims,\
#             tranpabmat, tranpbcmat, tranpacmat,\
#             tranpmmat, simpb, simpc, \
#             tranmsmat, simm, \
#             tranallmat, \
#             erglam, ergdistall, ergdistsa, ergdistsb, ergdistpa, ergdistpb, ergdistpc, ergdistmb, \
#             cldistsb, cldistpa, cldistpb, cldistpc, cldistm, mcldistsa, mcldistsb, cmcldistsb, \
#             negnwdist, mnegnwdist\
#             ], open('../vars/simulation.pickle', 'wb'))