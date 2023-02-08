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


########## IMPORT PARAMETER VALUES ##########

extpar, _, _, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))

sigma, cfloor, \
rpd, rps, rpu, rpb, rmd, rmb, \
betas, betap, betam, \
colfrac,\
deltavec, \
sums, sump, summ = extpar

bdeta, Neta, bdeta2, Neta2,\
bdeps, Neps, bdeps2, Neps2,\
icmin, icmax1, icmax2, Nic, \
ismin, ismax1, ismax2, Nis, \
Nh, \
zpmax, Nzp, Nzpzero, \
asmax, Nas, \
Nzs, \
Nzphat, Nzphatzero, \
Nzphat2, Nzphat2zero, \
zmmax, Nzm,\
Nzmhat = gridpar


# Parameters related to VFI algorithm
step = 0.5
itmax = 50


########## CONSTRUCT GRIDS & INIT GUESS ##########

@njit
def getgrids(par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    x = np.linspace(-bdeta*seta, bdeta*seta, Neta)
    half_step = bdeta*seta/(Neta-1)
    Peta = tauchen(x, Neta, 0, seta, half_step)
    Peta = Peta[0]
    etagrid = np.array([exp(meta+k*seta) for k in np.linspace(-bdeta, bdeta, Neta)])
    
    x = np.linspace(-bdeps*seps, bdeps*seps, Neps)
    half_step = bdeps*seps/(Neps-1)
    Peps = tauchen(x, Neps, 0, seps, half_step)
    Peps = Peps[0]
    epsgrid = np.array([exp(meps+k*seps) for k in np.linspace(-bdeps, bdeps, Neps)])
    
    icgrid = np.empty(Nic)
    icgrid[:-1] = np.linspace(icmin, icmax1, Nic-1)
    icgrid[-1] = icmax2
    
    isgrid = np.empty(Nis)
    isgrid[:-1] = np.linspace(ismin, ismax1, Nis-1)
    isgrid[-1] = ismax2
    
    hmin = rhowage*exp(meta - seta) + (1-rhowage)*theta*f(icmin, ismin, par, polpar)
    hmax = rhowage*exp(meta + seta) + (1-rhowage)*theta*f(icmax2, ismax2, par, polpar)
    hgrid = np.linspace(hmin, hmax, Nh)
    
    asgrid = np.zeros(Nas)
    asgrid[0] = 0
    asgrid[1:] = exp(np.linspace(log(1e-1), log(asmax), Nas-1))
    
    zpmin,_ = rapf(-ismax2, 0, polpar)
    zpgrid = np.zeros(Nzp)
    zpgrid[:Nzpzero] = -exp(np.linspace(log(-zpmin), log(1e-1), Nzpzero))
    zpgrid[Nzpzero] = 0
    zpgrid[Nzpzero+1:] = exp(np.linspace(log(1e-1), log(zpmax), Nzp-Nzpzero-1))
    
    zsgrid = np.zeros(Nzs)
    zsgrid[0] = 0
    zsgrid[1:] = exp(np.linspace(log(1e-1), log(asmax + exp(meta + 2*seta)), Nzs-1))

    zphatgrid = np.empty((Nzphat, Nh))
    zphat2grid = np.empty((Nzphat2, Nh))
    
    for hi, h in enumerate(hgrid):
        
        zphatmin = - lambc*ybarm/(1+rmb)*h- ybarp*h + 1
        zphatmax = zpmax
        zphatgrid[:Nzphatzero, hi] = -exp(np.linspace(log(-zphatmin), log(1e-1), Nzphatzero))
        zphatgrid[Nzphatzero, hi] = 0
        zphatgrid[Nzphatzero+1:, hi] = exp(np.linspace(log(1e-1), log(zphatmax), Nzphat-Nzphatzero-1))
        
        zphat2min = -lambc*ybarm/(1+rmb)*h + 1e-1
        zphat2max = zpmax + ybarp*h
        zphat2grid[:Nzphat2zero, hi] = -exp(np.linspace(log(-zphat2min), log(1e-1), Nzphat2zero))
        zphat2grid[Nzphat2zero, hi] = 0
        zphat2grid[Nzphat2zero+1:, hi] = exp(np.linspace(log(1e-1), log(zphat2max), Nzphat2-Nzphat2zero-1))        

        
    zmmin = (1-lambc)*hmin
    zmgrid = exp(np.linspace(log(zmmin), log(zmmax), Nzm))
    zmhatgrid = np.linspace(zmgrid[0]+ybarm*exp(meps - seps), zmgrid[-1]+ybarm*exp(meps + seps), Nzmhat)
    
    return Peta, Peps, etagrid, epsgrid, \
            icgrid, isgrid, hgrid,\
            asgrid, zpgrid, \
            zsgrid, \
            zphatgrid, zphat2grid,\
            zmgrid, zmhatgrid


@njit
def initdEVs(asgrid, icgrid):
    
    dEVsgrid = np.zeros((Nas, Nic, 3))
    
    for asi, asval in enumerate(asgrid):
        for ici, ic in enumerate(icgrid):
            dEVsgrid[asi, ici, 0] = 1e-2*(1/(asval + 1) + 1e-2*log(ic+1)/(asval + 1))
            dEVsgrid[asi, ici, 1] = 1e-2*(1/(ic + 1) + 1e-2*log(asval+1)/(ic + 1))
            dEVsgrid[asi, ici, 2] = 1e-2*(log(asval + 1) + log(ic + 1) + 1e-2*log(asval + 1)*log(ic + 1))
                
    return dEVsgrid


########## UPDATING VALUE FUNCTIONS ##########


# Middle-aged Stage

@njit
def res_as(asval, zmhat, ic, n, asgrid, icgrid, dEVsgrid, par):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    rhovec = getrhovec(par)
    rho = rhovec[n]
    
    cm = zmhat - (n/2)*asval
    
    dEVsda = interp2d_exp(asval, ic, asgrid, icgrid, dEVsgrid[:, :, 0])
    
    return rho*betam*dEVsda - (n/2)*du(cm)


@njit
def bisect_as(zmhat, ic, n, asgrid, icgrid, dEVsgrid, par):
   
    x0 = 0
    x1 = zmhat/(n/2)*0.99
    
    y0 = res_as(x0, zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
    y1 = res_as(x1, zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
        return x1
    else:
        
        for iter in range(100):
            x2 = (x0+x1)/2
            y2 = res_as(x2, zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if iter == 99:
            raise ValueError("bisect_as: max iter reached")
        
        return (x0 + x1)/2


@njit
def solveVm(zmhat, ic, n, asgrid, icgrid, dEVsgrid, par):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    rhovec = getrhovec(par)
    rho = rhovec[n]
    
    asval = bisect_as(zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
    cm = zmhat - (n/2)*asval
    
    dEVsda = interp2d_exp(asval, ic, asgrid, icgrid, dEVsgrid[:, :, 0])
    dEVsdi = interp2d_exp(asval, ic, asgrid, icgrid, dEVsgrid[:, :, 1])
    EVs = interp2d_lin(asval, ic, asgrid, icgrid, dEVsgrid[:, :, 2])
    
    deltam = 0
    if asval <= 0:
        deltam = - res_as(asval, zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
    
    dVmdz = du(cm)
    dVmdi = rho*betam*dEVsdi
    Vmval = u(cm) + rho*betam*EVs
    
    return dVmdz, dVmdi, Vmval, deltam, asval, cm


@njit
def makegrid_Vm(zmgrid, zmhatgrid, asgrid, icgrid, dEVsgrid, Peps, epsgrid, par, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    solveVmgrid = np.zeros((Nzmhat, Nic, 4, 3))
    dVmgrid = np.zeros((Nzmhat, Nic, 4, 3))
    dEVmgrid = np.zeros((Nzm, Nic, 4, 3))
    
    tmpvec2 = np.zeros((Neps, 3))
    
    for ici, ic in enumerate(icgrid):
        for n in range(1, 4):
            for zmhati, zmhat in enumerate(zmhatgrid):
                tmpvec = solveVm(zmhat, ic, n, asgrid, icgrid, dEVsgrid, par)
                solveVmgrid[zmhati, ici, n, :] = tmpvec[3:]
                dVmgrid[zmhati, ici, n, :] = tmpvec[:3]
            
            for zmi, zm in enumerate(zmgrid):
                for epsmi, epsm in enumerate(epsgrid):
                    
                    tmpvec2[epsmi, 0] = interp_exp(zm+ybarm*epsm, zmhatgrid, dVmgrid[:, ici, n, 0])
                    tmpvec2[epsmi, 1] = interp_exp(zm+ybarm*epsm, zmhatgrid, dVmgrid[:, ici, n, 1])
                    tmpvec2[epsmi, 2] = interp_lin(zm+ybarm*epsm, zmhatgrid, dVmgrid[:, ici, n, 2])
                    
                dEVmgrid[zmi, ici, n, :] = tmpvec2.T @ Peps
                    
    return solveVmgrid, dVmgrid, dEVmgrid


# Parent Stage

@njit
def res_am(am, zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    ic = (zphat2 - am)/(n/2)
    ram = ramf(am)
    rm = rmf(am)
    zm = ram + ybarm*hp
    
    dEVmdz = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 0])
    dEVmdi = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 1])
    
    return betap*(1+rm)*dEVmdz - betap/(n/2)*dEVmdi


@njit
def bisect_am(zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    x0 = -lambc*ybarm/(1+rmb)*hp
    x1 = zphat2
    
    y0 = res_am(x0, zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
    y1 = res_am(x1, zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
        return x1
    else:
        
        if zphat2 >= 0:
            ic = zphat2/(n/2)
            zm = ybarm*hp

            dEVmdz = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 0])
            dEVmdi = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 1])

            res1 = betap*(1+rmb)*dEVmdz - betap/(n/2)*dEVmdi
            res2 = betap*(1+rmd)*dEVmdz - betap/(n/2)*dEVmdi

            if res1 >= 0 and res2 <= 0:
                return 0
        
        for it in range(100):
            x2 = (x0+x1)/2
            y2 = res_am(x2, zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if it == 99:
            raise ValueError("bisect_am: max iter reached")
        
        return (x0 + x1)/2


@njit
def solvetilVZp(zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    am = bisect_am(zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
    ic = (zphat2 - am)/(n/2)
    ram = ramf(am)
    rm = rmf(am)
    zm = ram + ybarm*hp
    
    dEVmdz = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 0])
    dEVmdi = interp2d_exp(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 1])
    EVmval = interp2d_lin(zm, ic, zmgrid, icgrid, dEVmgrid[:, :, n, 2])
    
    tilVZpval = betap*EVmval
    
    mu = 0
    deltai = 0
    res = res_am(am, zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
    if am <= -lambc*ybarm/(1+rmb)*hp:
        mu = -res
    elif am >= zphat2:
        deltai = res*(n/2)
    
    dtilVzpdz = (betap*dEVmdi + deltai)/(n/2)
    dtilVzpdh = betap*ybarm*dEVmdz + mu*lambc*ybarm/(1+rmb)
    
    return dtilVzpdz, dtilVzpdh, tilVZpval, mu, deltai, am, ic


@njit
def makegrid_tilVZp(zphat2grid, hgrid, zmgrid, icgrid, dEVmgrid, par, polpar):
    
    solvetilVZpgrid = np.zeros((Nzphat2, Nh, 4, 4))
    dtilVZpgrid = np.zeros((Nzphat2, Nh, 4, 3))
    
    for hpi, hp in enumerate(hgrid):
        for zphat2i, zphat2 in enumerate(zphat2grid[:, hpi]):
            for n in range(1, 4):
                tmpvec = solvetilVZp(zphat2, hp, n, zmgrid, icgrid, dEVmgrid, par, polpar)
                solvetilVZpgrid[zphat2i, hpi, n, :] = tmpvec[3:]
                dtilVZpgrid[zphat2i, hpi, n, :] = tmpvec[:3]
                    
    return solvetilVZpgrid, dtilVZpgrid


@njit
def res_zphat2(zphat2, zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    delta = deltavec[n]
    
    hp = hgrid[hpi]
    
    cp = (zphat + ybarp*hp - zphat2)/(1+delta)
    dtilVzpdz = interp_exp(zphat2, zphat2grid[:, hpi], dtilVZpgrid[:, hpi, n, 0])
    
    return dtilVzpdz - du(cp)/(1+delta)


@njit
def bisect_zphat2(zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    delta = deltavec[n]
    
    hp = hgrid[hpi]
    
    x0 = - lambc*ybarm/(1+rmb)*hp + 1e-2
    x1 = zphat + ybarp*hp - 1e-2
    
    y0 = res_zphat2(x0, zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, polpar)
    y1 = res_zphat2(x1, zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
        return x1
    else:
        
        for it in range(100):
            x2 = (x0+x1)/2
            y2 = res_zphat2(x2, zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if it == 99:
            raise ValueError("bisect_zphat2: max iter reached")
        
        return (x0 + x1)/2


@njit
def solvetilVp(zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, solvetilVZpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    delta = deltavec[n]
    
    hp = hgrid[hpi]
    
    zphat2 = bisect_zphat2(zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, par, polpar)
    cp = (zphat + ybarp*hp - zphat2)/(1+delta)
    
    dtilVpdz = du(cp)/(1+delta)
    
    dtilVzpdz = interp_exp(zphat2, zphat2grid[:, hpi], dtilVZpgrid[:, hpi, n, 0])
    dtilVzpdh = interp_exp(zphat2, zphat2grid[:, hpi], dtilVZpgrid[:, hpi, n, 1])
    tilVzpval = interp_lin(zphat2, zphat2grid[:, hpi], dtilVZpgrid[:, hpi, n, 2])
    
    tilVpval = u(cp) + tilVzpval
    
    mu = 0
    res = res_zphat2(zphat2, zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, polpar)
    if zphat2 <= -lambc*ybarm/(1+rmb)*hp + 1e-2:
        mu = - res
    dtilVpdh = dtilVzpdh + du(cp)/(1+delta)*ybarp + mu*lambc*ybarm/(1+rmb)
    
    mu2 = interp_lin(zphat2, zphat2grid[:, hpi], solvetilVZpgrid[:, hpi, n, 0])
    deltai = interp_lin(zphat2, zphat2grid[:, hpi], solvetilVZpgrid[:, hpi, n, 1]) 
    am = interp_lin(zphat2, zphat2grid[:, hpi], solvetilVZpgrid[:, hpi, n, 2]) 
    ic = interp_lin(zphat2, zphat2grid[:, hpi], solvetilVZpgrid[:, hpi, n, 3]) 
    
    return dtilVpdz, dtilVpdh, tilVpval, mu+mu2, mu+deltai, am, ic, cp


@njit
def dEVm_nk(zm, Peps, epsgrid, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    tmpvec = np.zeros((2, Neps))
    
    for epsmi, epsm in enumerate(epsgrid):
        zmhat = zm + ybarm*epsm
        tmpvec[0, epsmi] = du(zmhat)
        tmpvec[1, epsmi] = u(zmhat)
        
    dEVm = tmpvec[0] @ Peps
    EVmval = tmpvec[1] @ Peps
    
    return dEVm, EVmval


@njit
def res_am_nk(am, zphat, hp, Peps, epsgrid, par, polpar):

    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    cp = zphat + ybarp*hp - am
    ram = ramf(am)
    rm = rmf(am)
    
    zm = ram + ybarm*hp
    
    dEVm, EVmval = dEVm_nk(zm, Peps, epsgrid, polpar)
    
    res = betap*(1+rm)*dEVm - du(cp)
    
    return res


@njit
def bisect_am_nk(zphat, hp, Peps, epsgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    x0 = -lambc*ybarm/(1+rmb)*hp
    x1 = zphat + ybarp*hp - 1e-2
    
    y0 = res_am_nk(x0, zphat, hp, Peps, epsgrid, par, polpar)
    y1 = res_am_nk(x1, zphat, hp, Peps, epsgrid, par, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
#         print("Reached minimal c!")
#         raise ValueError("Reached minimal c!")
        return x1
    else:
        
        cp = zphat + ybarp*hp
        zm = ybarm*hp

        dEVm, EVmval = dEVm_nk(zm, Peps, epsgrid, polpar)

        res1 = betap*(1+rmd)*dEVm - du(cp)
        res2 = betap*(1+rmb)*dEVm - du(cp)
        
        if res1 >= 0 and res2 <= 0:
            return 0
        
        for iter in range(100):
            x2 = (x0+x1)/2
            y2 = res_am_nk(x2, zphat, hp, Peps, epsgrid, par, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if iter == 99:
            raise ValueError("bisect_am_nk: max iter reached")
        
        return (x0 + x1)/2
    

@njit
def solvetilVp_nk(zphat, hp, Peps, epsgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    am = bisect_am_nk(zphat, hp, Peps, epsgrid, par, polpar)
    
    cp = zphat + ybarp*hp - am
    ram = ramf(am)
    rm = rmf(am)
    
    zm = ram + ybarm*hp
    
    dEVm, EVmval = dEVm_nk(zm, Peps, epsgrid, polpar)
    tilVpval = u(cp) + betap*EVmval
    
    dtilVpdz = du(cp)
    
    res = res_am_nk(am, zphat, hp, Peps, epsgrid, par, polpar)
    mu = 0
    if am <= -lambc*ybarm/(1+rmb)*hp:
        mu = -res
        
    dtilVpdh = betap*ybarm*dEVm + du(cp)*ybarp + mu*lambc*ybarm/(1+rmb)
    
    return dtilVpdz, dtilVpdh, tilVpval, mu, 0., am, 0., cp


@njit
def makegrid_tilVp(zphatgrid, zphat2grid, hgrid, dtilVZpgrid, solvetilVZpgrid, Peps, epsgrid, par, polpar):
    
    solvetilVpgrid = np.zeros((Nzphat, Nh, 4, 5))
    dtilVpgrid = np.zeros((Nzphat, Nh, 4, 3))
    
    for hpi, hp in enumerate(hgrid):
        for zphati, zphat in enumerate(zphatgrid[:, hpi]):

            ### nokids
            tmpvec = solvetilVp_nk(zphat, hp, Peps, epsgrid, par, polpar)
            solvetilVpgrid[zphati, hpi, 0, :] = tmpvec[3:]
            dtilVpgrid[zphati, hpi, 0, :] = tmpvec[:3]

            for n in range(1, 4):
                tmpvec = solvetilVp(zphat, hpi, n, zphat2grid, hgrid, dtilVZpgrid, solvetilVZpgrid, par, polpar)
                solvetilVpgrid[zphati, hpi, n, :] = tmpvec[3:]
                dtilVpgrid[zphati, hpi, n, :] = tmpvec[:3]
    
    return solvetilVpgrid, dtilVpgrid


@njit
def makegrid_Vp(dtilVpgrid, par):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    pngrid = np.zeros((Nzphat, Nh, 4))
    dVpgrid = np.zeros((Nzphat, Nh, 3))
    
    tmpvec = np.zeros((3, 4))
    tmppn = np.zeros(4)
    
    for hpi in range(Nh):
        for zphati in range(Nzphat):
            for n in range(4):
                tmpvec[:, n] = dtilVpgrid[zphati, hpi, n, :]
            
            tmpvec[2, 0] = tmpvec[2, 0] + nuhat

            for n in range(4):
                pngrid[zphati, hpi, n] = exp(tmpvec[2, n]/sign)/np.sum(exp(tmpvec[2, :]/sign))

            tmppn[:] = pngrid[zphati, hpi, :]

            dVpgrid[zphati, hpi, 0] = tmppn @ tmpvec[0, :]
            dVpgrid[zphati, hpi, 1] = tmppn @ tmpvec[1, :]
            dVpgrid[zphati, hpi, 2] = sign*(np.euler_gamma + log(np.sum(exp(tmpvec[2, :]/sign))))

    return pngrid, dVpgrid


@njit
def makegrid_EVp(zpgrid, zphatgrid, epsgrid, dVpgrid, Peps, polpar):
    
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    dEVpgrid = np.zeros((Nzp, Nh, 3))
    
    tmpvec = np.zeros((Neps, 3))
    
    for hpi in range(Nh):
        for zpi, zp in enumerate(zpgrid):
            for epspi, epsp in enumerate(epsgrid):         
                
                zphat = zp + ybarp*epsp
                
                tmpvec[epspi, 0] = interp_exp(zphat, zphatgrid[:, hpi], dVpgrid[:, hpi, 0])
                tmpvec[epspi, 1] = interp_exp(zphat, zphatgrid[:, hpi], dVpgrid[:, hpi, 1])
                tmpvec[epspi, 2] = interp_lin(zphat, zphatgrid[:, hpi], dVpgrid[:, hpi, 2])
            
            dEVpgrid[zpi, hpi, :] = tmpvec.T @ Peps
                
    return dEVpgrid


# Student Stage

@njit
def res_is(isval, zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    ap = zs - isval
    rap, tl = rapf(ap, E, polpar)
    rp = rpf(ap, E, polpar)
    hp = rhowage*ys + (1-rhowage)*theta*f(ic, isval, par, polpar)
    
    dEVpdz = interp2d_exp(rap, hp, zpgrid, hgrid, dEVpgrid[:, :, 0])
    dEVpdh = interp2d_exp(rap, hp, zpgrid, hgrid, dEVpgrid[:, :, 1])
    
    res = betas*(1-rhowage)*theta*fs(ic, isval, par, polpar)*dEVpdh \
            - betas*(1+rp)*dEVpdz
    
    return res


@njit
def bisect_is(zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    x0 = 0
    x1 = ismax2
    
    y0 = res_is(x0, zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)
    y1 = res_is(x1, zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
        return x1
    else:
        
        hp = rhowage*ys + (1-rhowage)*theta*f(ic, zs, par, polpar)

        dEVpdz = interp_exp(hp, hgrid, dEVpgrid[Nzpzero, :, 0])
        dEVpdh = interp_exp(hp, hgrid, dEVpgrid[Nzpzero, :, 1])

        res1 = betas*(1-rhowage)*theta*fs(ic, zs, par, polpar)*dEVpdh \
                - betas*(1+rpd)*dEVpdz
        res2 = betas*(1-rhowage)*theta*fs(ic, zs, par, polpar)*dEVpdh*(1+rps)/(1+rpd) \
                - betas*(1+rpd)*dEVpdz
        
        if res1 >= 0 and res2 <= 0:
            return 0
        
        for iter in range(100):
            x2 = (x0+x1)/2
            y2 = res_is(x2, zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if iter == 99:
            raise ValueError("bisect_is: max iter reached")
        
        return (x0 + x1)/2
    

@njit
def solveVZs(zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    isval = bisect_is(zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)
    
    ap = zs - isval
    rp = rpf(ap, E, polpar)
    hp = rhowage*ys + (1-rhowage)*theta*f(ic, isval, par, polpar)
    
    dEVpdz = interp2d_exp(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 0])
    dEVpdh = interp2d_exp(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 1])
    EVpval = interp2d_lin(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 2])
    
    VZsval = betas*EVpval
    
    res = res_is(isval, zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)
    delta = 0.
    mu = 0.
    
    if isval <= 0:
        delta = -res
    if isval >= ismax2:
        mu = res
        
    dVZsdz = betas*theta*(1-rhowage)*fs(ic, isval, par, polpar)*dEVpdh + delta + mu
    dVZsdi = betas*theta*(1-rhowage)*fc(ic, isval, par, polpar)*dEVpdh
    
    return dVZsdz, dVZsdi, VZsval, delta, ap, isval


@njit
def makegrid_VZs(zsgrid, etagrid, icgrid, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    solveVZsgrid = np.zeros((Nzs, Neta, Nic, 2, 3))
    dVZsgrid = np.zeros((Nzs, Neta, Nic, 2, 3))
    
    for zsi, zs in enumerate(zsgrid):
        for ysi, ys in enumerate(etagrid):
            for ici, ic in enumerate(icgrid):
                for E in range(2):

                    tmpvec = solveVZs(zs, ys, ic, E, zpgrid, hgrid, dEVpgrid, par, polpar)

                    solveVZsgrid[zsi, ysi, ici, E, :] = tmpvec[3:]
                    dVZsgrid[zsi, ysi, ici, E, :] = tmpvec[:3]
                    
    return solveVZsgrid, dVZsgrid


@njit
def res_zs(zs, asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    E = (asval <= asbar)
    ys = etagrid[ysi]
    cs = (summ/sums)*asval + colfrac*ys - zs
    
    dVZsdz = interp_exp(zs, zsgrid, dVZsgrid[:, ysi, ici, int(E), 0])
    
    res = dVZsdz - du(cs)
    
    return res


@njit
def bisect_zs(asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar):
    
    ys = etagrid[ysi]
    
    x0 = 0.
    x1 = (summ/sums)*asval + colfrac*ys  - 1e-2
    
    y0 = res_zs(x0, asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar)
    y1 = res_zs(x1, asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
#         print("Reached minimal c!")
        return x1
    else:
        
        for iter in range(100):
            x2 = (x0+x1)/2
            y2 = res_zs(x2, asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if iter == 99:
            raise ValueError("bisect_zs: max iter reached")
        
        return (x0 + x1)/2


@njit
def solvetilVs(asval, ysi, ici, etagrid, zsgrid, dVZsgrid, solveVZsgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    E = (asval <= asbar)
    ys = etagrid[ysi]
    
    zs = bisect_zs(asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar)
    cs = (summ/sums)*asval + colfrac*ys - zs
    
    dVZsdz = interp_exp(zs, zsgrid, dVZsgrid[:, ysi, ici, int(E), 0])
    dVZsdi = interp_exp(zs, zsgrid, dVZsgrid[:, ysi, ici, int(E), 1])
    VZsval = interp_lin(zs, zsgrid, dVZsgrid[:, ysi, ici, int(E), 2])
    
    tilVsval = u(cs) + VZsval
    
    dtilVsda = du(cs)
    dtilVsdi = dVZsdi
    
    res = res_zs(zs, asval, ysi, ici, etagrid, zsgrid, dVZsgrid, par, polpar)
    
    mu = 0
    
    if zs <= 0:
        mu = -res
        
    delta = interp_lin(zs, zsgrid, solveVZsgrid[:, ysi, ici, int(E), 0])
    ap = interp_lin(zs, zsgrid, solveVZsgrid[:, ysi, ici, int(E), 1])
    isval = interp_lin(zs, zsgrid, solveVZsgrid[:, ysi, ici, int(E), 2])
    
    return dtilVsda, dtilVsdi, tilVsval, mu, delta, ap, isval, cs


@njit
def res_ap_nc(ap, asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    cs = (summ/sums)*asval + ys - ap

    rap, tl = rapf(ap, asval <= asbar, polpar)
    rp = rpf(ap, asval <= asbar, polpar)
    hp = rhowage*ys + (1-rhowage)*theta*f(ic, 0, par, polpar)
    hpi = np.argmin((hp - hgrid)**2)
    zp = rap + ybarp*hp
    
    dEVpda = interp2d_exp(rap, hp, zpgrid, hgrid, dEVpgrid[:, :, 0])
    
    res = betas*dEVpda - du(cs)
    
    return res


@njit
def bisect_ap_nc(asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    x0 = 0.
    x1 = (summ/sums)*asval + ys - 1e-2
    
    y0 = res_ap_nc(x0, asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
    y1 = res_ap_nc(x1, asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
    
    if y0 <= 0:
        return x0
    elif y1 >= 0:
#         print("Reached minimal c!")
        return x1
    else:
        
        for iter in range(100):
            x2 = (x0+x1)/2
            y2 = res_ap_nc(x2, asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
            
            if y2*y0 >= 0:
                x0 = x2
                y0 = y2
            else:
                x1 = x2
                y1 = y2
            
            if np.abs(x0 - x1) < 1e-6:
                break
        
        if iter == 99:
            raise ValueError("bisect_zs: max iter reached")
        
        return (x0 + x1)/2
    

@njit
def solvetilVs_nc(asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    ap = bisect_ap_nc(asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
    cs = (summ/sums)*asval + ys - ap
    rap, tl = rapf(ap, asval <= asbar, polpar)
    rp = rpf(ap, asval <= asbar, polpar)
    hp = rhowage*ys + (1-rhowage)*theta*f(ic, 0, par, polpar)
    hpi = np.argmin((hp - hgrid)**2)
    zp = rap + ybarp*hp
    
    dEVpda = interp2d_exp(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 0])
    dEVpdh = interp2d_exp(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 1])
    EVpval = interp2d_lin(ap, hp, zpgrid, hgrid, dEVpgrid[:, :, 2])
    
    tilVsval = u(cs) + betas*EVpval
    
    dtilVsda = (summ/sums)*du(cs)
    dtilVsdi = betas*(1-rhowage)*theta*fc(ic, 0, par, polpar)*dEVpdh
    
    res = res_ap_nc(ap, asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
    
    mu = 0
    
    if ap <= 0:
        mu = -res
    
    return dtilVsda, dtilVsdi, tilVsval, mu, 0., ap, 0., cs


@njit
def makegrid_Vs(asgrid, etagrid, icgrid, zpgrid, hgrid, zsgrid, dEVpgrid, dVZsgrid, solveVZsgrid, \
                Peta, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    
    solvetilVsgrid = np.zeros((Nas, Neta, Nic, 5, 2))
    dtilVsgrid = np.zeros((Nas, Neta, Nic, 3, 2))
    colgrid = np.zeros((Nas, Neta, Nic))
    dVsgrid = np.zeros((Nas, Neta, Nic, 3))
    dEVsgrid = np.zeros((Nas, Nic, 3))
    
    tmpvec = np.zeros((8, 2))
    tmpvec2 = np.zeros((Neta, 3))
    tmpvec3 = np.zeros((3, 2))
    
    for asi, asval in enumerate(asgrid):
        for ici, ic in enumerate(icgrid):
            for ysi, ys in enumerate(etagrid):
            
                
                tmpvec[:, 0] = solvetilVs_nc(asval, ys, ic, zpgrid, hgrid, dEVpgrid, par, polpar)
                tmpvec[:, 1] = solvetilVs(asval, ysi, ici, etagrid, zsgrid, dVZsgrid, solveVZsgrid, par, polpar)
                
                dtilVsgrid[asi, ysi, ici, :, :] = tmpvec[:3]
                solvetilVsgrid[asi, ysi, ici, :, :] = tmpvec[3:]
                
                col = np.argmax(tmpvec[2,:])
                colgrid[asi, ysi, ici] = col
                dVsgrid[asi, ysi, ici, :] = tmpvec[:3, col]
                tmpvec2[ysi, :] = dVsgrid[asi, ysi, ici, :]
            
            dEVsgrid[asi, ici, :] = Peta @ tmpvec2
                
    return solvetilVsgrid, dtilVsgrid, colgrid, dVsgrid, dEVsgrid


########## COMBINE ALL FUNCTIONS ##########

@njit
def update(dEVsgrid, \
           zphatgrid, zphat2grid, \
           zmgrid, zmhatgrid, \
           zsgrid, \
           hgrid, \
           asgrid, zpgrid, \
           icgrid, \
           Peta, etagrid, \
           Peps, epsgrid, \
           par, polpar):
    
    
    solveVmgrid, dVmgrid, dEVmgrid = makegrid_Vm(zmgrid, zmhatgrid, asgrid, icgrid, dEVsgrid, Peps, epsgrid, par, polpar)
    solvetilVZpgrid, dtilVZpgrid = makegrid_tilVZp(zphat2grid, hgrid, zmgrid, icgrid, dEVmgrid, par, polpar)
    solvetilVpgrid, dtilVpgrid = makegrid_tilVp(zphatgrid, zphat2grid, hgrid, dtilVZpgrid, solvetilVZpgrid, Peps, epsgrid, par, polpar)
    pngrid, dVpgrid = makegrid_Vp(dtilVpgrid, par)
    dEVpgrid = makegrid_EVp(zpgrid, zphatgrid, epsgrid, dVpgrid, Peps, polpar)
    solveVZsgrid, dVZsgrid = makegrid_VZs(zsgrid, etagrid, icgrid, zpgrid, hgrid, dEVpgrid, par, polpar)
    solvetilVsgrid, dtilVsgrid, colgrid, dVsgrid, new_dEVsgrid \
        = makegrid_Vs(asgrid, etagrid, icgrid, zpgrid, hgrid, zsgrid, dEVpgrid, dVZsgrid, solveVZsgrid, \
                Peta, par, polpar)
    
    return new_dEVsgrid,\
            solveVmgrid, dVmgrid, \
            solvetilVZpgrid, dtilVZpgrid, \
            solvetilVpgrid, dtilVpgrid, \
            pngrid, dVpgrid, \
            dEVpgrid, \
            solveVZsgrid, dVZsgrid, \
            solvetilVsgrid, dtilVsgrid, colgrid, dVsgrid

@njit
def VFIsolve(par, polpar, init_dEVsgrid):
    
    Peta, Peps, etagrid, epsgrid, \
    icgrid, isgrid, hgrid,\
    asgrid, zpgrid, \
    zsgrid, \
    zphatgrid, zphat2grid,\
    zmgrid, zmhatgrid = getgrids(par, polpar)
    
    dEVsgrid,\
    solveVmgrid, dVmgrid, \
    solvetilVZpgrid, dtilVZpgrid, \
    solvetilVpgrid, dtilVpgrid, \
    pngrid, dVpgrid, \
    dEVpgrid, \
    solveVZsgrid, dVZsgrid, \
    solvetilVsgrid, dtilVsgrid, colgrid, dVsgrid = update(init_dEVsgrid, \
                                                           zphatgrid, zphat2grid, \
                                                           zmgrid, zmhatgrid, \
                                                           zsgrid, \
                                                           hgrid, \
                                                           asgrid, zpgrid, \
                                                           icgrid, \
                                                           Peta, etagrid, \
                                                           Peps, epsgrid, \
                                                           par, polpar)
            
    for it in range(itmax):
        
        dEVsgrid_new,\
        solveVmgrid_new, dVmgrid, \
        solvetilVZpgrid, dtilVZpgrid, \
        solvetilVpgrid_new, dtilVpgrid, \
        pngrid_new, dVpgrid, \
        dEVpgrid, \
        solveVZsgrid, dVZsgrid, \
        solvetilVsgrid_new, dtilVsgrid, colgrid_new, dVsgrid = update(dEVsgrid, \
                                                           zphatgrid, zphat2grid, \
                                                           zmgrid, zmhatgrid, \
                                                           zsgrid, \
                                                           hgrid, \
                                                           asgrid, zpgrid, \
                                                           icgrid, \
                                                           Peta, etagrid, \
                                                           Peps, epsgrid, \
                                                           par, polpar)

        if np.sum(np.isnan(dEVsgrid_new)):
            raise ValueError("nan in grid!")

        res = np.max(np.array([np.mean(np.abs(solveVmgrid_new - solveVmgrid)), 
                               np.mean(np.abs(solvetilVpgrid_new - solvetilVpgrid)), 
                               np.mean(np.abs(pngrid_new - pngrid))*100, 
                               np.mean(np.abs(solvetilVsgrid_new - solvetilVsgrid)), 
                               np.mean(np.abs(colgrid_new - colgrid))]))
                     
        if it % 10 == 0:
            print(res)
        if res < 1e-1:
            print("Converged successfully")
            break

        dEVsgrid = step*dEVsgrid_new + (1-step)*dEVsgrid
        
        solveVmgrid,\
        solvetilVpgrid,\
        pngrid,\
        solvetilVsgrid,\
        colgrid = solveVmgrid_new,\
                solvetilVpgrid_new,\
                pngrid_new,\
                solvetilVsgrid_new,\
                colgrid_new
        
        if it == itmax - 1:
            print("Reached max iteration")
            
    
    return dEVsgrid, \
            solveVmgrid,\
            solvetilVpgrid,\
            pngrid,\
            solvetilVsgrid,\
            colgrid