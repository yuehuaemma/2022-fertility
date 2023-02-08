import numpy as np
import pickle

from math import exp


########## EXTPAR: PARAMETERS NEVER CHANGED #########

# Preference
sigma = 2
cfloor = 9.31
# cfloor = 15

rann = 0.04084393939
rpsann = 0.06182727273
rpuann = rpsann
rpbann = 0.10
rmbann = 0.07

Tc = 16
Ts = 24
Tp = 40
Tr = 65
Tm = 78

sums = 0
for a in range(Tc, Ts):
    sums += 1/(1+rann)**(a-Tc)
    
sump = 0
for a in range(Ts, Tp):
    sump += 1/(1+rann)**(a-Ts)
    
summ = 0
for a in range(Tp, Tm):
    summ += 1/(1+rann)**(a-Tp)
    
sumr = 0
for a in range(Tp, Tr):
    sumr += 1/(1+rann)**(a-Tp)
    
# Interest rates
rpd = (1+rann)**(Ts-Tc)*(sums/sump) - 1
# rps = (1+rpsann)**(Ts-Tc)*(sums/sump) - 1
rps = (1+rpsann)**10/(1+rann)**(10-Ts+Tc)*(sums/sump) - 1
rpu = (1+rpuann)**10/(1+rann)**(10-Ts+Tc)*(sums/sump) - 1
rpb = (1+rpbann)**10/(1+rann)**(10-Ts+Tc)*(sums/sump) - 1
rmd = (1+rann)**(Tp-Ts)*(sump/summ) - 1
rmb = (1+rmbann)**(Tp-Ts)*(sump/summ) - 1

# Discount rates
betaann = 1/(1+rann)
betas = (sump/sums)*betaann**(Ts-Tc)
betap = (summ/sump)*betaann**(Tp-Ts)
betam = sums/summ

sum1 = 0
for a in range(Tc, Tc+4):
    sum1 += 1/(1+rann)**(a - Tc)
sum2 = 0
for a in range(Tc, Ts):
    sum2 += 1/(1+rann)**(a - Tc)
colfrac = 1-sum1/sum2

# Child-related costs
deltavec = np.array([0, 1.26*0.5, 1, 0.76*1.5])

extpar = np.array([sigma, cfloor, \
                   rpd, rps, rpu, rpb, rmd, rmb, \
                   betas, betap, betam, \
                   colfrac,\
                   deltavec, \
                   sums, sump, summ], dtype=object)


########## POLPAR: PARAMETERS CHANGED IN POLICY ANALYSIS #########

# Earnings
ybarp = 1

# from stata earnings-profile.do
coeffs = np.array([-2.774455, .3067481, -.0035516])

tmp = 0
for a in range(Ts, Tp):
    tmp += exp(coeffs @ np.array([1, a, a**2])) / (1+rann)**(a - Ts)
Wp = tmp / sump

tmp = 0
for a in range(Tp, Tr):
    tmp += exp(coeffs @ np.array([1, a, a**2])) / (1+rann)**(a - Tp) 
Wm = tmp / summ

ybarm = Wm / Wp

# Loan limit
Lann = 23
L = Lann/sums

# Educational subsidy
psann = 10
hsann = 7.507
scann = 9.183
cgann = 27.973

phic1 = 0.
phis1 = (1 - 0.28)/0.28

tmp = 0
for a in range(6, Tc):
    tmp += (psann)/(1+rann)**(a)
phic0 = tmp/sump

tmp = 0
for a in range(Tc, 18):
    tmp += (hsann)/(1+rann)**(a - Tc)

phis0 = tmp/sums

polpar = np.array([ybarp, ybarm, L, phic0, phic1, phis0, phis1])


########## PAR0: INITIAL VALUES FOR CALIBRATION ##########

meta = 1.95841429
seta = 0.80081713
rhowage = 0.58209302
meps = 1.65031154
seps = 1.56633276

theta = 4.66573292
a = 0.6016456
b = -1.95792604
d = 0.8509686

alpha0 = 1.01431508
alpha1 = 2.21814495
rhovec = np.array([alpha0*n**alpha1 for n in range(4)])

sign = 0.00810057

lambc = 0.89224603

asbar = 10.
nuhat = 0.09215976

par0 = np.array([meta, seta, rhowage, meps, seps, theta, a, b, d, alpha0, alpha1, sign, lambc, \
                asbar, nuhat])

########## GRIDPAR: PARAMETERS RELATED TO GRIDS ##########

bdeta = 2.
Neta = 11
bdeta2 = 3.
Neta2 = 51

bdeps = 2.
Neps = 9
bdeps2 = 3.
Neps2 = 51


icmin = 0
icmax1 = 20
icmax2 = 30
Nic = 25

ismin = 0
ismax1 = 20
ismax2 = 30
Nis = 25

Nh = 50

asmax = 200
Nas = 50

zpmax = 200
Nzp = 50
Nzpzero = 20

Nzs = 40

Nzphat = 70
Nzphatzero = 25

Nzphat2 = 70
Nzphat2zero = 25

zmmax = 150
Nzm = 40

Nzmhat = 40

gridpar = np.array([bdeta, Neta, bdeta2, Neta2,\
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
                   Nzmhat], dtype = np.int64)


########## STORE ALL PARAMETER VALUES ##########
pickle.dump([extpar, polpar, par0, gridpar], open('../vars/params.pickle', 'wb'))
pickle.dump(par0, open('../vars/par0.pickle', 'wb'))