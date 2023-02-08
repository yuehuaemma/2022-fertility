import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import pickle

from numpy import exp, log
from math import erfc, sqrt
from numba import jit, njit, typed

from basic import std_norm_cdf, tauchen, \
                    interp2d_lin, interp2d_exp, interp3d_lin, interp3d_exp, interp_lin, interp_exp

from fcn import getrhovec, \
                u, du, \
                Scf, dScf, Ssf, dSsf, f, fc, fs, \
                rapf, rpf, ramf, rmf

from VFI import getgrids, initdEVs, VFIsolve

_, polpar, par0, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))

Peta, Peps, etagrid, epsgrid, \
icgrid, isgrid, hgrid,\
asgrid, apgrid,\
zsgrid, \
zphatgrid, zphat2grid,\
zmgrid, zmhatgrid = getgrids(par0, polpar)

init_dEVsgrid = initdEVs(asgrid, icgrid)
# init_dEVsgrid = pickle.load(open('../vars/init_dEVsgrid.pickle', 'rb'))

dEVsgrid, \
solveVmgrid,\
solvetilVpgrid,\
pngrid,\
solvetilVsgrid,\
colgrid = VFIsolve(par0, polpar, init_dEVsgrid)

pickle.dump([dEVsgrid, \
            solveVmgrid,\
            solvetilVpgrid,\
            pngrid,\
            solvetilVsgrid,\
            colgrid], open('../vars/VFI.pickle', 'wb'))
pickle.dump(dEVsgrid, open('../vars/init_dEVsgrid.pickle', 'wb'))