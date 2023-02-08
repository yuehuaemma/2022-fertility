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

_, _, _, gridpar = pickle.load(open('../vars/params.pickle', 'rb'))

from VFI import getgrids, VFIsolve

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

Nsa = Nas*Nic
Nsb = Nas*Nic*Neta
Npa = Nzp*Nh
Npb = Nzp*Nh*Neps
Npb2 = Nzp*Nh*Neps2
Npc = Nzphat*Nh*4
Nma = Nzm*Nic*4
Nmb = Nzmhat*Nic*4

@njit
def gettransmats(solveVmgrid,\
                 solvetilVpgrid,\
                 pngrid,\
                 solvetilVsgrid,\
                 colgrid,\
                 par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    Peta, Peps, etagrid, epsgrid, \
    icgrid, isgrid, hgrid,\
    asgrid, zpgrid, \
    zsgrid, \
    zphatgrid, zphat2grid,\
    zmgrid, zmhatgrid = getgrids(par, polpar)

    transabmat = np.zeros((Nsa, Nsb))
    transpmat = np.zeros((Nsb, Npa))
    
    tranpabmat = np.zeros((Npa, Npb))
    tranpbcmat = np.zeros((Npb, Npc))
    tranpacmat = np.zeros((Npa, Npc))
    
    tranpmmat = np.zeros((Npc, Nmb))
    tranmsmat = np.zeros((Nmb, Nsa))
    
    tranallmat = np.zeros((Nsa, Nsa))

    sims = np.zeros((Nsb, 12))
    simpb = np.zeros((Npb, 3))
    simpc = np.zeros((Npc, 7))
    simm = np.zeros((Nmb, 3))

    for asi, asval in enumerate(asgrid):
        
        E = (asval <= asbar)
        
        for ici, ic in enumerate(icgrid):
            
            sai = asi*Nic + ici
            
            for ysi, ys in enumerate(etagrid):                

                sbi = asi*Nic*Neta + ici*Neta + ysi
                
                transabmat[sai, sbi] += Peta[ysi]

                col = colgrid[asi, ysi, ici]

                if col == 0:
                    mus, deltas, ap, isval, cs = solvetilVsgrid[asi, ysi, ici, :, 0]
                elif col == 1:
                    mus, deltas, ap, isval, cs = solvetilVsgrid[asi, ysi, ici, :, 1]

                hp = rhowage*ys + (1-rhowage)*theta*f(ic, isval, par, polpar)
                hpi = np.argmin((hp - hgrid)**2)
                zp, lt = rapf(ap, E, polpar)
                zpi = np.argmin((zp - zpgrid)**2)
                pai = zpi*Nh + hpi

                transpmat[sbi, pai] = 1

                sims[sbi] = col, mus, deltas, ap, isval, \
                            cs, hp, float(lt), ys, asval,\
                            ic, zp
                    
                
    for zpi, zp in enumerate(zpgrid):
        
        for hpi, hp in enumerate(hgrid):
            
            pai = zpi*Nh + hpi
            
            for epspi, epsp in enumerate(epsgrid):
                
                pbi = zpi*Nh*Neps + hpi*Neps + epspi
                tranpabmat[pai, pbi] = Peps[epspi]
                
                zphat = zp + ybarp*epsp
                zphati = np.argmin((zphat - zphatgrid[:, hpi])**2)
                
                yp = ybarp*(hp+epsp)
                
                simpb[pbi] = zp, hp, yp
                
                for n in range(4):
                    
                    pci = zphati*Nh*4 + hpi*4 + n
                    tranpbcmat[pbi, pci] = pngrid[zphati, hpi, n]
                    
    tranpacmat = tranpabmat @ tranpbcmat
                    
    for hpi, hp in enumerate(hgrid):
        for zphati, zphat in enumerate(zphatgrid[:, hpi]):
        
            for n in range(4):
                
                pci = zphati*Nh*4 + hpi*4 + n
                
                mup, deltap, am, ic, cp = solvetilVpgrid[zphati, hpi, n, :]
                ici = np.argmin((ic - icgrid)**2)
                ram = ramf(am)
                
                simpc[pci] = mup, deltap, am, ic, cp, \
                            hp, float(n)
                
                for epsmi, epsm in enumerate(epsgrid):
                    
                    zmhat = ram + ybarm*(hp + epsm)
                    zmhati = np.argmin((zmhat - zmhatgrid)**2)
                    
                    mbi = zmhati*Nic*4 + ici*4 + n
                    
                    tranpmmat[pci, mbi] += Peps[epsmi]
    
    for zmhati, zmhat in enumerate(zmhatgrid):
        for ici, ic in enumerate(icgrid):
            for n in range(4):
                
                mbi = zmhati*Nic*4 + ici*4 + n
                
                deltam, asval, cm = solveVmgrid[zmhati, ici, n, :]
                asi = np.argmin((asval - asgrid)**2)
                
                simm[mbi] = deltam, asval, cm
                
                sai = asi*Nic + ici
                
                tranmsmat[mbi, sai] = n/2
                
    tranallmat = transabmat @ transpmat @ tranpacmat @ tranpmmat @ tranmsmat
                
    return transabmat, \
            transpmat, sims, \
            tranpabmat, tranpbcmat, tranpacmat,\
            tranpmmat, simpb, simpc, \
            tranmsmat, simm, \
            tranallmat

def genergdist(tranallmat, transabmat, transpmat, tranpabmat, tranpbcmat, tranpmmat, tranmsmat):
    
    lams, vs = np.linalg.eig(tranallmat.T)
    
    for it in range(Nsa):
        tmpvec = np.real(vs[:, it])
        if all(tmpvec <= 1e-10) or all(tmpvec >= -1e-10):
            erglam = np.real(lams[it])
            ergdistall = tmpvec/np.sum(tmpvec)
            break
            
    ergdistsa = ergdistall
    ergdistsb = ergdistsa @ transabmat
    ergdistpa = ergdistsb @ transpmat
    ergdistpb = ergdistpa @ tranpabmat
    ergdistpc = ergdistpb @ tranpbcmat
    ergdistmb = ergdistpc @ tranpmmat
            
    return erglam, ergdistall, ergdistsa, ergdistsb, ergdistpa, ergdistpb, ergdistpc, ergdistmb

# @njit
def getcldist(ergdistsb, sims, transpmat, tranpabmat, tranpacmat, tranpmmat, tranmsmat, transabmat):
    
    cldistsb = np.zeros((2, Nsb))
    cldistsb[0] = ergdistsb * (1-sims[:, 0])
    cldistsb[1] = ergdistsb * sims[:, 0]

    cldistpa = cldistsb @ transpmat
    cldistpb = cldistpa @ tranpabmat
    cldistpc = cldistpa @ tranpacmat
    cldistm = cldistpc @ tranpmmat
    
    mcldistsa = cldistm @ tranmsmat
    mcldistsb = mcldistsa @ transabmat
    
    cmcldistsb = mcldistsb * sims[:, 0]
    
    return cldistsb, cldistpa, cldistpb, cldistpc, cldistm, mcldistsa, mcldistsb, cmcldistsb

def getmom1(ergdistsb, mcldistsb, sims):
    
    mom1 = np.zeros(2)
    tmp = np.zeros(2)
    
    tmp[0] = ergdistsb @ sims[:, 0]
    
    if np.sum(mcldistsb[1]) == 0:
        tmp[1] = 1e2
    else:
        tmp[1] = mcldistsb[1] @ sims[:, 0] / np.sum(mcldistsb[1])
    
    for i in range(len(tmp)):
        if tmp[i] <= 1e-6 or tmp[i] >= 1 - 1e-6:
            mom1[i] = 1e2
        else:
#             mom1[i] = log(1/(1-tmp[i])-1)
            mom1[i] = tmp[i]
    
    return mom1

def getmom2(ergdistsb, ergdistpc, sims, simpc, cldistsb, cldistpc):
    
    mom2 = np.zeros(3)
    tmp = np.zeros(3)
    
    if np.sum(cldistsb[1]) == 0:
        tmp[0] = 1e2
        tmp[2] = 1e2
    else:
        tmp[0] = cldistsb[1] @ (sims[:, 3] < -5e-2)/np.sum(cldistsb[1])
        tmp[2] = cldistpc[1] @ (simpc[:, 2] < -5e-2)/np.sum(cldistpc[1])
        
#     negnwdist = ergdistsb * np.logical_and(sims[:, 3] < -0.1, sims[:, 0] == 1)
#     if np.sum(negnwdist) == 0:
#         tmp[1] = 1e2
#     else:
#         tmp[1] = negnwdist @ (-sims[:, 3] / 10)/np.sum(negnwdist)
        
    tmp[1] = ergdistpc @ (simpc[:, 2] < -5e-2)
    
    
#     for i in range(2):
#         if np.sum(cldistpb[i]) == 0:
#             tmp[i+2] = 1e2
#         else:
#             cldistpb[i] = cldistpb[i]/np.sum(cldistpb[i])
#             tmp[i+2] = cldistpb[i] @ (simp[:, 2] < 0)/np.sum(cldistpb[i])
    
    for i in range(3):
        if tmp[i] <= 1e-6 or tmp[i] >= 1 - 1e-6:
            mom2[i] = 1e2
        else:
#             mom2[i] = log(1/(1-tmp[i])-1)
            mom2[i] = tmp[i]
    
    return mom2

@njit
def getmom3(cldistsb, sims, ergdistsb, etagrid, par, polpar):
    
    meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, asbar, nuhat = par
    ybarp, ybarm, L, phic0, phic1, phis0, phis1 = polpar
    
    mom3 = np.zeros(6)

    x = np.linspace(-bdeta2*seta, bdeta2*seta, Neta2)
    half_step = bdeta2*seta/(Neta2-1)
    Peta2 = tauchen(x, Neta2, 0, seta, half_step)
    Peta2 = Peta2[0]
    etagrid2 = np.array([exp(meta+k*seta) for k in np.linspace(-bdeta2, bdeta2, Neta2)])
    
    x = np.linspace(-bdeps2*seps, bdeps2*seps, Neps2)
    half_step = bdeps2*seps/(Neps2-1)
    Peps2 = tauchen(x, Neps2, 0, seps, half_step)
    Peps2 = Peps2[0]
    epsgrid2 = np.array([exp(meps+k*seps) for k in np.linspace(-bdeps2, bdeps2, Neps2)])

    Nsc = Nsb*Neps2
    ysvec = np.zeros(Nsc)
    lnysvec = np.zeros(Nsc)
    lnypvec = np.zeros(Nsc)
    ergdistsc = np.zeros(Nsc)
    cldistsc = np.zeros((2, Nsc))
    
    for asi in range(Nas):
        for ici in range(Nic):
            
            sai = asi*Nic + ici

            for ys2i, ys2 in enumerate(etagrid2):

                ysi = np.argmin((ys2 - etagrid)**2)

                sbi = asi*Nic*Neta + ici*Neta + ysi

                for epsi, eps in enumerate(epsgrid2):

                    sci = sbi*Neps2 + epsi

                    ysvec[sci] = ys2
                    lnysvec[sci] = log(ys2)

                    lnypvec[sci] = log(ybarp*(sims[sbi, 6] + eps))
                    ergdistsc[sci] = ergdistsb[sbi]*Peps2[epsi]
                    for i in range(2):
                        cldistsc[i, sci] = cldistsb[i, sbi]*Peps2[epsi]
    
    if np.sum(cldistsb[0]) == 0:
        mom3[0] = 1e2
        mom3[3] = 1e2
    else:
        v1 = ysvec
        v2 = lnysvec
        wts = cldistsc[0]
        tmpm, tmpsd = wtd_avg_and_std(v1, wts)
        mom3[0] = tmpm/10
        tmpm, tmpsd = wtd_avg_and_std(v2, wts)
        mom3[3] = tmpsd**2
    
    for i in range(2):
        if np.sum(cldistsb[i]) == 0:
            mom3[i+1] = 1e2
#             mom3[i+5] = 1e2
        else:
                
            mom3[i+1] = cldistsb[i] @ (ybarp*(sims[:, 6] + exp(meps + seps**2/2)))/np.sum(cldistsb[i])
            mom3[i+1] = mom3[i+1]/10
            
#             values = lnypvec
#             wts = cldistsc[i]
#             tmpm, tmpsd = wtd_avg_and_std(values, wts)
#             mom3[i+5] = tmpsd**2
    
    values = lnypvec
    wts = cldistsc[1]
    tmpm, tmpsd = wtd_avg_and_std(values, wts)
    mom3[5] = tmpsd**2
    
    wts = ergdistsc
    tmpm, tmpsd = wtd_avg_and_std(values, wts)
    mom3[4] = tmpsd**2

#     if np.sum(cldistsb[0]) == 0:
#         mom3[7] = 1e2
#     else:
#         v1 = lnysvec
#         v2 = lnypvec
#         wts = cldistsc[0]
#         tmpcov = wtd_cov(v1, v2, wts)
#         _, tmpsd1 = wtd_avg_and_std(v1, wts)
#         _, tmpsd2 = wtd_avg_and_std(v2, wts)
        
#         mom3[7] = tmpcov/(tmpsd1*tmpsd2)
    
    return mom3

def genapqdist(cldistsb, sims, transpmat, tranpabmat, tranpbcmat, tranpmmat, tranmsmat, transabmat):
    
    apqs = weighted_quantile(sims[:, 3], np.array([0.25, 0.5, 0.75]), cldistsb[1])
    apqdistsb = np.zeros((4, Nsb))
    apqdistpb = np.zeros((4, Npb))
    apqdistpc = np.zeros((4, Npc))

    apqdistsb[0] = cldistsb[1] * (sims[:, 3] < apqs[0])
    apqdistsb[1] = cldistsb[1] * np.logical_and(sims[:, 3] >= apqs[0], sims[:, 3] < apqs[1])
    apqdistsb[2] = cldistsb[1] * np.logical_and(sims[:, 3] >= apqs[1], sims[:, 3] < apqs[2])
    apqdistsb[3] = cldistsb[1] * (sims[:, 3] >= apqs[2])
    
    apqdistpb = apqdistsb @ transpmat @ tranpabmat
    apqdistpc = apqdistpb @ tranpbcmat
        
    mapqdist = apqdistpc @ tranpmmat @ tranmsmat @ transabmat
    
    return apqs, apqdistsb, apqdistpb, apqdistpc, mapqdist

# def gennegnwdist(ergdistsb, sims, transpmat, tranpacmat, tranpmmat, tranmsmat, transabmat):
    
    
#     negnwdist = np.zeros((3, Npc))

#     negnwdist[0] = ergdistsb * np.logical_and(sims[:, 3] < -1, sims[:, 0] == 1) @ transpmat @ tranpacmat
#     negnwdist[1] = ergdistsb * np.logical_and(np.logical_and(sims[:, 3] >= -1, sims[:, 3] < 1), \
#                                               sims[:, 0] == 1) @ transpmat @ tranpacmat
#     negnwdist[2] = ergdistsb * np.logical_and(sims[:, 3] >= 1, sims[:, 0] == 1) @ transpmat @ tranpacmat
        
#     mnegnwdist = negnwdist @ tranpmmat @ tranmsmat @ transabmat
    
#     return negnwdist, mnegnwdist

def gennegnwdist(ergdistsb, sims, transpmat, tranpacmat, tranpmmat, tranmsmat, transabmat):
    
    
    negnwdist = np.zeros((2, Npc))

    negnwdist[0] = ergdistsb * np.logical_and(sims[:, 3] < -5e-2, sims[:, 0] == 1) @ transpmat @ tranpacmat
    negnwdist[1] = ergdistsb * np.logical_and(sims[:, 3] >= -5e-2, sims[:, 0] == 1) @ transpmat @ tranpacmat
        
    mnegnwdist = negnwdist @ tranpmmat @ tranmsmat @ transabmat
    
    return negnwdist, mnegnwdist

def gennegnwdistpb(ergdistpb, simp, tranpmmat, tranmsmat, transabmat):
    
    negnwdistpb = np.zeros((2, Npb))
    
    negnwdistpb[0] = ergdistpb * (simp[:, 2] < 0 )
    negnwdistpb[1] = ergdistpb * (simp[:, 2] >= 0)
    
    mnegnwdistpb = negnwdistpb @ tranpmmat @ tranmsmat @ transabmat
    
    return negnwdistpb, mnegnwdistpb

# @njit
def getmom4(cldistpc, negnwdist, simpc):
    
    mom4 = np.zeros(4)
    
    for i in range(2):
        if np.sum(cldistpc[i]) == 0:
            mom4[i] = 1e2
    #         mom4[3] = 1e2
        else:
            tmpm, tmpsd = wtd_avg_and_std(simpc[:, 6], cldistpc[i])

            if tmpm >= 3 - 1e-2 or tmpm <= 1e-2:
                mom4[i] = 1e2
    #             mom4[3] = 1e2
            else:
                mom4[i] = tmpm
    #             mom4[3] = tmpsd**2
     
    for i in range(2):
        if np.sum(negnwdist[i]) == 0:
            mom4[i+2] = 1e2
#             mom4[i+4] = 1e2
        else:
            tmpm, tmpsd = wtd_avg_and_std(simpc[:, 6], negnwdist[i])
            
            if tmpm >= 3 - 1e-2 or tmpm <= 1e-2:
                mom4[i+2] = 1e2
#                 mom4[i+4] = 1e2
            else:
                mom4[i+2] = tmpm
#                 mom4[i+4] = tmpsd**2
    
    return mom4

# @njit
def getmom5(cldistpc, negnwdist, simpc, hs_coeffs):
    
    mom5 = np.zeros(4)
    
    for i in range(2):
        if np.sum(cldistpc[i]) == 0:
            mom5[i] = 1e2
        else:
            
            tmp = cldistpc[i] @ simpc[:, 3] / np.sum(cldistpc[i])
            if tmp <= 1e-3:
                mom5[i] = 1e2
            else:
                mom5[i] = (hs_coeffs[0] + simpc[:, 3]*hs_coeffs[1] + simpc[:, 3]**2*hs_coeffs[2]) @ cldistpc[i] / np.sum(cldistpc[i])
    
    for i in range(2):
        if np.sum(negnwdist[i]) == 0:
            mom5[i+2] = 1e2
        else:
            
            tmp = simpc[:, 3] @ negnwdist[i] / np.sum(negnwdist[i])
            if tmp <= 1e-3:
                mom5[i+2] = 1e2
            else:
                mom5[i+2] = (hs_coeffs[0] + simpc[:, 3]*hs_coeffs[1] + simpc[:, 3]**2*hs_coeffs[2]) \
                            @ negnwdist[i] / np.sum(negnwdist[i])
    
    return mom5

# @njit
def getmom6(cldistsb, cmcldistsb, sims):
    
    mom6 = np.zeros(2)
    tmp = np.zeros(2)
    
    tmp[0] = cldistsb[1] @ sims[:, 4]
    
    if np.sum(cmcldistsb[1]) == 0:
        tmp[1] = 1e3
    else:
        tmp[1] = cmcldistsb[1] @ sims[:, 4] /np.sum(cmcldistsb[1])
            
    mom6[:] = tmp[:]/10
    
    return mom6

def getallmoms(ergdistsb, ergdistpc, \
               cldistsb, cldistpc, mcldistsb, cmcldistsb, \
               negnwdist, \
               sims, simpc, \
               etagrid, par, polpar, hs_coeffs):
    
    momslist = []
    momslist.append(getmom1(ergdistsb, mcldistsb, sims))
    momslist.append(getmom2(ergdistsb, ergdistpc, sims, simpc, cldistsb, cldistpc))
    momslist.append(getmom3(cldistsb, sims, ergdistsb, etagrid, par, polpar))
    momslist.append(getmom4(cldistpc, negnwdist, simpc))
    momslist.append(getmom5(cldistpc, negnwdist, simpc, hs_coeffs))
    momslist.append(getmom6(cldistsb, cmcldistsb, sims))
    
    
    Nmoms = len(momslist)
    lens = np.zeros(Nmoms).astype(int)
    for i in range(Nmoms):
        lens[i] = len(momslist[i])
    
    allmoms = np.zeros(np.sum(lens))
    for i in range(Nmoms):
        allmoms[np.sum(lens[:i]):np.sum(lens[:i+1])] = momslist[i]
    
    return allmoms