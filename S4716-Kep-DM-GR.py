# Import the required modules
from __future__ import print_function, division
import numpy as np
from math import pi
from scipy.constants import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from iminuit import Minuit
import math
import kepler
import time
start=time.time()


premo=1
prem=1
G1=G*(prem**3)/(premo)*(((365.25*24*60*60)**(2))*3.40367597004765*10**(-50))/((5.02739933*10**(-31)))
c1=c*(3.24077929*10**(-17))*(365.25*24*60*60)*prem                          #speed of light in parsec/year


#######################################fit kepler orbit

def n1(a1,Mbh1):
    n1=((a1**3)/(G1*Mbh1))**(-1/2)
    return n1

def M(tp,t,Mbh1,a1):
    M=n1(a1,Mbh1)*(t-tp)
    return M

def E1(tp,t,Mbh1,a1,e1):
    E1=kepler.solve(M(tp,t,Mbh1,a1), e1)
    return E1

def keprad(tp,t,Mbh1,a1,e1):
    keprad=a1*(1-e1*np.cos(E1(tp,t,Mbh1,a1,e1), dtype=np.float64))
    return keprad

def emel(e1):
    emel=((1+e1)/(1-e1))**(1/2)
    return emel


def theta1(tp,t,Mbh1,a1,e1):
    theta1=2*np.arctan(emel(e1)*np.tan(E1(tp,t,Mbh1,a1,e1)/2, dtype=np.float64), dtype=np.float64)
    return theta1


def dr1(tp,t,Mbh1,a1,e1):
    dr=np.sin(E1(tp,t,Mbh1,a1,e1), dtype=np.float64)*n1(a1,Mbh1)*(a1**2)*e1/keprad(tp,t,Mbh1,a1,e1)
    return dr

def dt2(tp,t,Mbh1,a1,e1):
    dt2=(n1(a1,Mbh1)*(a1**2)*((1-e1**2)**(1/2)))/(keprad(tp,t,Mbh1,a1,e1)**2)
    return dt2



#dataorbit = pd.read_excel(r'multipleorbitdata.xlsx')
dataorbit = pd.read_excel(r'OrbitdataS4716.xlsx')

def timecon(my_list):
    return [ x-2003.44 for x in my_list ]


def parsec(my_list,Ro1):
    return [ np.tan(((x/1000/3600)*0.0174532925)/2)*Ro1*2 for x in my_list ]

#def velsectoyear(my_list):
#    return [ x*365.25*24*60*60 for x in my_list ]

def power(my_list):
    return [ x**2 for x in my_list ]

def velparsectoyear(my_list):
    return [ x*365.25*24*60*60*prem*3.24077929*10**(-14) for x in my_list ]



#orbit data for S2
#orbit data for S4716
RAS2=list(dataorbit['RA(mas)S4716'])
DECS2=list(dataorbit['Decl(mas)S4716'])
RAerrorS2=list(dataorbit['ErrorRA(mas)S4716'])
DECerrorS2=list(dataorbit['ErrorDecl(mas)S4716'])
#VLRS2=list(dataorbit['Radialvelocity(kmpersec)S2'])
#errorVLSR2=list(dataorbit['Error Radial velocity (kmpersec)S2'])
tid2=list(dataorbit['Time'])
tidS2=timecon(tid2)                 #Time starts at year zero instead of 2003.44
#print('tidS2=',tidS2)
#VLRS2=velparsectoyear(VLRS2)
#errorVLSRS2=velparsectoyear(errorVLSR2)

#RAS2re= [x for x in RAS2 if str(x) != 'nan']
RAS2re=RAS2
#DECS2re= [x for x in DECS2 if str(x) != 'nan']
DECS2re=DECS2
#RAerrorS2re= [x for x in RAerrorS2 if str(x) != 'nan']
RAerrorS2re=RAerrorS2
#DECerrorS2re= [x for x in DECerrorS2 if str(x) != 'nan']
DECerrorS2re=DECerrorS2
#errorVLSRS2re= [x for x in errorVLSRS2 if str(x) != 'nan']
#VLRS2re= [x for x in VLRS2 if str(x) != 'nan']




def LSQXkep(Mbh1,Ro1,oomegaS2,OmegaS2,IS2,tp,a1,e1):                        #least square which we want to minimize
    uuS2=[]
    uuS12=[]
    for i in range(len(DECS2)):
        if math.isnan(DECS2[i]):
            continue
        else:
            radkep=keprad(tp,tidS2[i],Mbh1,a1,e1)
            the=theta1(tp,tidS2[i],Mbh1,a1,e1)
            uuS12.append(radkep)
            uuS2.append(the)
    uuS12=np.transpose(uuS12)
    uuS2=np.transpose(uuS2)
    residulDECS2=np.sum(((parsec(DECS2re,Ro1) - uuS12*(np.cos(OmegaS2, dtype=np.float64)*np.cos(oomegaS2+uuS2, dtype=np.float64)-np.sin(OmegaS2, dtype=np.float64)*np.sin(oomegaS2+uuS2, dtype=np.float64)*np.cos(IS2, dtype=np.float64)))** 2)/power(parsec(DECerrorS2re,Ro1)) , dtype=np.float64)
    residulARAS2=np.sum(((parsec(RAS2re,Ro1) - uuS12*(np.sin(OmegaS2, dtype=np.float64)*np.cos(oomegaS2+uuS2, dtype=np.float64)+np.cos(OmegaS2, dtype=np.float64)*np.sin(oomegaS2+uuS2, dtype=np.float64)*np.cos(IS2, dtype=np.float64)))** 2)/power(parsec(RAerrorS2re,Ro1)) , dtype=np.float64)
    resid=residulDECS2+residulARAS2
    return resid

print('len(S4716)=',len(DECS2))

#initial guess
Mbh=4*10**6                                                        #black hole mass in solar mass
R1=8000                                                            #distance to orbit
oomegaS2g=0.0012          #omega=0.073°= 0.001 rad
OmegaS2g=2.65            #Omega=151.54°=2.65
IS2g=2.814               #i=161.24° = 2.814 rad
tpg=2
a1g=0.00193
e1g=0.756


print('Chi square before',LSQXkep(Mbh,R1,oomegaS2g,OmegaS2g,IS2g,tpg,a1g,e1g)/((len(RAS2re)+len(DECS2re))-8))


print('fit all same time')
ms = Minuit(LSQXkep, Mbh1=Mbh,Ro1=R1,oomegaS2=oomegaS2g,OmegaS2=OmegaS2g,IS2=IS2g,tp=tpg,a1=a1g,e1=e1g)
#ms.limits['e1'] = (0.756,0.756)
ms.limits['e1'] = (0,1)
#ms.limits['oomegaS2']= (0.0012,0.0012)
ms.limits['oomegaS2']= (0,2*pi)
#ms.limits['OmegaS2']= (2.65,2.65)
ms.limits['OmegaS2']= (0,2*pi)
#ms.limits['IS2']= (2.814,2.814)
ms.limits['IS2']= (0,2*pi)
#ms.limits['Ro1']= (8000,8100)
ms.limits['Ro1']= (7500,8500)
#ms.limits['Mbh1']= (4*(10**6),4.1*(10**6))
ms.limits['Mbh1']= (4*(10**6),5*(10**6))
ms.limits['tp']= (0,20)
tttts=ms.simplex().migrad()
print(tttts)


#global parameters
print('Intial Mbh=',Mbh,'  Fitted Mbh=',ms.values[0])
print('Intial R0=',R1,'  Fitted R0=',ms.values[1])


#Orbit parameters S2
print('Intial omega S2=',oomegaS2g,'  Fitted omega S2=',ms.values[2])
print('Intial Omega S2=',OmegaS2g,'  Fitted Omega S2=',ms.values[3])
print('Intial I S2=',IS2g,'  Fitted I S2=',ms.values[4])
print('Intial tp S2=',tpg,'  Fitted tp S2=',ms.values[5])
print('Intial a S2=',a1g,'  Fitted a S2=',ms.values[6])
print('Intial e S2=',e1g,'  Fitted e S2=',ms.values[7])

Mbhreal=ms.values[0]
R0real=ms.values[1]
oomegareal=ms.values[2]
Omegareal=ms.values[3]
IS2real=ms.values[4]
tpreal=ms.values[5]
areal=ms.values[6]
ereal=ms.values[7]

print('Chi square after',LSQXkep(Mbhreal,R0real,oomegareal,Omegareal,IS2real,tpreal,areal,ereal)/((len(RAS2re)+len(DECS2re))-8))
#print('længde:',len(RAS2re)*2) #just to check the length
#time1=np.linspace(0,10,1000)

radiplot=[]
thetaplot=[]
for i in range(1000):
    dt=i*10/1000
    radiplot.append(keprad(tpreal,dt,Mbhreal,areal,ereal))
    thetaplot.append(theta1(tpreal,dt,Mbhreal,areal,ereal))

radiplot=np.transpose(radiplot)
thetaplot=np.transpose(thetaplot)

#Ved DM+GR plot udkommenteres plottet af Kepler, da koden ellers får problemer med at kører

#plt.plot(radiplot*(np.sin(Omegareal, dtype=np.float64)*np.cos(oomegareal+thetaplot, dtype=np.float64)+np.cos(Omegareal, dtype=np.float64)*np.sin(oomegareal+thetaplot, dtype=np.float64)*np.cos(IS2real, dtype=np.float64)),
#         radiplot*(np.cos(Omegareal, dtype=np.float64)*np.cos(oomegareal+thetaplot, dtype=np.float64)-np.sin(Omegareal, dtype=np.float64)*np.sin(oomegareal+thetaplot, dtype=np.float64)*np.cos(IS2real, dtype=np.float64)),color='grey',label='Kepler fit')
#
#plt.errorbar(parsec(RAS2,R0real),parsec(DECS2,R0real),xerr=parsec(RAerrorS2,R0real),yerr=parsec(DECerrorS2,R0real),marker='.',capsize=2, ls='none',color="purple",ecolor='pink',label='S4716')
#
#plt.xlabel('Right ascension [pc]')
#plt.ylabel('Declination [pc]')
#plt.plot(0,0,'ko',markeredgecolor='orange',label='Sgr. A*')         #plots a black point with orange edge to show the SMBH
#plt.legend()
#plt.grid(alpha=0.5)
#plt.axis('equal')
#plt.show()

###############################################################################################
# Here we will determine a theoretical value of the radial velocity of S4716.
# This is done due to the fact that there is no observational values of the radial velocity of
# S4716 in the article by Peissker.
###############################################################################################



###############################################################################################
### Now we want to code the non-keplerian orbit. This means adding GR and the DM potential. ###
### In this case the movement of the black hole won't be taken into account. ##################
###############################################################################################
agamma = 0.122

def ps(rhoo1, Ro1, ras1):
    ps = rhoo1 * (Ro1 / ras1) * (1 + Ro1 / ras1) ** 2
    return ps

def Rsp(Mbh1, f0):
    Rsp = ((agamma ** 2) * Mbh1 / (f0 ** 3)) ** (1 / 2)
    return Rsp

def rhoh(r, f0):
    rhoh = f0 / r
    return rhoh

def rhoh2(r, rhoo1, Ro1, ras1):
    rhoh = ps(rhoo1, Ro1, ras1) * (ras1 / r) * (1 + r / ras1) ** (-2)
    return rhoh

def rhod(r, Mbh1, f0):
    rhod = rhoh(Rsp(Mbh1, f0)) * (r / Rsp(Mbh1, f0)) ** (-7 / 3)
    return rhod

def con(ras1):
    con = -4 * pi * (ras1 ** 3) * G1
    return con

ehhcon = 6 * G1 * pi * np.power(agamma, 4 / 3, dtype=np.float64)

def ehh(f0, Mbh1):
    ehh = ehhcon * f0 * np.power(Mbh1, 2 / 3, dtype=np.float64)
    return ehh


def Rss(Mbh1):                              #Schwarzchild radius
    Rss = 2 * G1 * Mbh1 / c1 ** 2
    return Rss


def dphiext(r, Mbh1, f0):                   #The DM potential
    if r < 2 * Rss(Mbh1):
        dphiext = 0
    if r > Rsp(Mbh1, f0):
        #dphiext = G1 * 2 * pi * (f0 ** 3) * (1 - (2 * Rss(Mbh1)) / (r ** 2))
        dphiext = G1 * 2 * pi * (f0 ** 3)
    else:
        dphiext = ehh(f0, Mbh1) * (1 / r ** (4 / 3) - ((2 * Rss(Mbh1)) ** (2 / 3)) / (r ** 2))

    return dphiext

#################################################################
#tp = 2002.33 * 365 * 24 * 60 * 60  # time of passage (bliver ikke brugt til noget)

steps = 10 ** 3
timelimit=16
t1 = np.linspace(0, timelimit, steps)

def timex(t):
    timex = np.linspace(0, t, steps)
    return timex

def Lm(per1, dtheta1):
    Lm = dtheta1 * per1 ** 2
    return Lm

def preces123(r, per1, dtheta1, Mbh1):
    preces123 = (3 * G1 * Mbh1 * (Lm(per1, dtheta1)) ** 2) / ((c1 ** 2) * (r ** 4))
    return preces123

def f123(s, t, per1, dtheta1, Mbh1, f0):  # diferential equations with more parameters
    r = s[0]  # r coordinate
    theta = s[1]  # theta
    dr = s[2]  # velocity part
    dtheta = s[3]  # angular velocity
    xdot = [[], [], [], []]
    xdot[0] = dr
    xdot[1] = dtheta
    xdot[2] = ((Lm(per1, dtheta1)) ** 2) / (r ** 3) - G1 * Mbh1 / (r ** 2) - dphiext(r, Mbh1, f0) \
              - preces123(r, per1, dtheta1, Mbh1)  # r acceleration
    xdot[3] = (Lm(per1, dtheta1)) / (r ** 2)  # theta acceleration
    return xdot

def g1(per1, dtheta1, Mbh1, th1, rvel1, f0, t):
    g1 = odeint(f123, [per1, 0, rvel1, th1], timex(t),
                args=(per1, dtheta1, Mbh1, f0,))  # solves the differential equations
    return g1


# least square which we want to minimize
def LSQX1(Mbh1, f0, Ro1, OmegaS2, IS2, perS2, dthetaS2, thS2, rvelS2):
    uS2 = []
    uS12 = []
    orrS2 = []
    oS2 = []
    otS2 = []
    odtS2 = []
    for i in range(len(DECS2)):
        if math.isnan(DECS2[i]):
            continue
        else:
            SolS21 = g1(perS2, dthetaS2, Mbh1, thS2, rvelS2, f0, tidS2[i])
            hS2 = SolS21[:, 3]  # gets the solution to the specific time
            qS2 = SolS21[:, 0]
            hS12 = hS2[steps - 1]  # takes the last element in the angle solution to the specific time
            qS12 = qS2[steps - 1]  # takes the last element in the radius solution to the specific time
            uS2.append(hS12)
            uS12.append(qS12)
            uuS2 = np.transpose(uS2)
            uuS12 = np.transpose(uS12)
#    for i in range(len(VLRS2)):
#        if math.isnan(VLRS2[i]):
#            continue
#        else:
#            SolS22 = g1(perS2, dthetaS2, Mbh1, thS2, rvelS2, f0, tidS2[i])
#            wS2 = SolS22[:, 2]
#            ppS2 = SolS22[:, 3]
#            kkS2 = SolS22[:, 0]
#            k1S2 = kkS2[steps - 1]  # takes the last element in the radius solution to the specific time
#            w1S2 = wS2[steps - 1]
#            p1S2 = ppS2[steps - 1]
#            orrS2.append(k1S2)
#            oS2.append(w1S2)
#            otS2.append(p1S2)
#            odtS2.append(Lm(perS2, dthetaS2) / (k1S2) ** 2)
#            ooS2 = np.transpose(oS2)
#            ootS2 = np.transpose(otS2)
#            oodtS2 = np.transpose(odtS2)
#            oorrS2 = np.transpose(orrS2)

#    residulVLSRS2 = np.sum(((VLRS2re - np.sin(IS2, dtype=np.float64) * (
#            ooS2 * np.sin(ootS2, dtype=np.float64) + oorrS2 * oodtS2 * np.cos(ootS2, dtype=np.float64))) ** 2)
#                           / power(errorVLSRS2re), dtype=np.float64)
    residulDECS2 = np.sum(((parsec(DECS2re, Ro1) -  uuS12 * (np.cos(OmegaS2, dtype=np.float64) * np.cos(uuS2, dtype=np.float64)
                                                             - np.sin(OmegaS2, dtype=np.float64) * np.sin(
            uuS2, dtype=np.float64) * np.cos(IS2, dtype=np.float64))) ** 2) / power(parsec(DECerrorS2re, Ro1)), dtype=np.float64)
    residulARAS2 = np.sum(((parsec(RAS2re, Ro1) - uuS12 * (np.sin(OmegaS2, dtype=np.float64) * np.cos(uuS2, dtype=np.float64) +
                                                           np.cos(OmegaS2,dtype=np.float64) * np.sin(
            uuS2, dtype=np.float64) * np.cos(IS2, dtype=np.float64))) ** 2) / power(parsec(RAerrorS2re, Ro1)),
                          dtype=np.float64)
#    resid = residulVLSRS2 + residulDECS2 + residulARAS2
    resid = residulDECS2 + residulARAS2
    return resid

def foooo(rho1, ras1):
    fooo = ras1 * rho1 * (R1 / ras1) * (1 + R1 / ras1) ** 2
    return fooo

# dark matter initial conditions
rho_o = premo * 0.0101 / (prem ** 3)  # solar masses pr cubic parsec
ras = 18600 * prem  # parsec
rhos = rho_o * (R1 / ras) * (1 + R1 / ras) ** 2
f0g = foooo(rho_o, ras) ** (1 / 3)
f0g=0.1 #5.68                    #initial guess for f0 in [M_odot pc^-2]^1/3
#f0g=0.1                    #to test the code with zero f0
print() #linjeskift i output
print('rho_odot =',rho_o)
print('rs =',ras)
print('rhos =',rhos)
print('f0g =',f0g)

# initial values S2
perS2g = keprad(tpreal, 0, Mbhreal, areal, ereal)
dthetaS2g = dt2(tpreal, 0, Mbhreal, areal, ereal)
thS2g = theta1(tpreal, 0, Mbhreal, areal, ereal) + oomegareal
rvelS2g = dr1(tpreal, 0, Mbhreal, areal, ereal)
OmegaS2g = Omegareal
IS2g = IS2real

#print('DM+GR Reduced Chi square before, Chi^2 =', LSQX1(Mbh, f0g, R1, OmegaS2g, IS2g, perS2g, dthetaS2g, thS2g, rvelS2g)
#                                                    /(len(RAS2re)*2+len(VLRS2re)-9))
print('DM+GR Reduced Chi square before, Chi^2 =', LSQX1(Mbh, f0g, R1, OmegaS2g, IS2g, perS2g, dthetaS2g, thS2g, rvelS2g)
                                                    /(len(RAS2re)*2-9))

m = Minuit(LSQX1, Mbh1=Mbhreal, f0=f0g, Ro1=R0real, OmegaS2=OmegaS2g, IS2=IS2g, perS2=perS2g, dthetaS2=dthetaS2g,
           thS2=thS2g, rvelS2=rvelS2g)

m.limits['OmegaS2'] = (0, 2 * pi)
m.limits['IS2'] = (0, 2 * pi)
m.limits['Mbh1'] = (4 * (10 ** 6) * premo, 5 * (10 ** 6) * premo)
m.limits['f0'] = (0, 30)
m.limits['Ro1']= (7500,9000)
m.limits['thS2'] = (0, 2 * pi)

tttt = m.simplex().migrad()
print(tttt)

# global parameters
print('Intial Mbh=', Mbhreal, '  Fitted Mbh=', m.values[0])
print('Intial f_o=', f0g, '  Fitted f_o=', m.values[1])
print('Intial R0=', R0real, '  Fitted R0=', m.values[2])

# Orbit parameters S2
print('Intial Omega S2=', OmegaS2g, '  Fitted Omega S2=', m.values[3])
print('Intial I S2=', IS2g, '  Fitted I S2=', m.values[4])
print('Intial per S2=', perS2g, '  Fitted per S2=', m.values[5])
print('Intial dtheta S2=', dthetaS2g, '  Fitted dtheta S2=', m.values[6])
print('Intial angle S2=', thS2g, '  Fitted angle S2=', m.values[7])
print('Intial rvel S2=', rvelS2g, '  Fitted rvel S2=', m.values[8])

MbhrealDMGR = m.values[0]
foreal = m.values[1]
R0real = m.values[2]
OmegastorS2 = m.values[3]
IrealS2 = m.values[4]
perrealS2 = m.values[5]
dthetarealS2 = m.values[6]
anglerealS2 = m.values[7]
rvelrealS2 = m.values[8]


#print('DM+GR Reduced Chi square after, Chi^2 =',
#      LSQX1(MbhrealDMGR, foreal, R0real, OmegastorS2, IrealS2, perrealS2, dthetarealS2, anglerealS2, rvelrealS2)
#      /(len(RAS2re)*2+len(VLRS2re)-9))
print('DM+GR Reduced Chi square after, Chi^2 =',
      LSQX1(MbhrealDMGR, foreal, R0real, OmegastorS2, IrealS2, perrealS2, dthetarealS2, anglerealS2, rvelrealS2)
      /(len(RAS2re)*2-9))
#############################################################


def finalS2(s, t):
    r = s[0]  # r coordinate
    theta = s[1]  # theta
    dr = s[2]  # velocity part
    dtheta = s[3]  # angular velocity
    xdot = [[], [], [], []]
    xdot[0] = dr
    xdot[1] = dtheta
    xdot[2] = (Lm(perrealS2, dthetarealS2) ** 2) / (r ** 3) - G1 * MbhrealDMGR / (r ** 2) - dphiext(r, MbhrealDMGR, foreal) \
              - preces123(r, perrealS2, dthetarealS2, MbhrealDMGR)  # r acceleration
    xdot[3] = (Lm(perrealS2, dthetarealS2)) / (r ** 2)  # theta acceleration
    return xdot


print('Find solution')
z4S1 = odeint(finalS2, [perrealS2, 0, rvelrealS2, anglerealS2], t1, mxstep=5000)


plt.plot(z4S1[:, 0] * (np.sin(OmegastorS2) * np.cos(z4S1[:, 3]) + np.cos(OmegastorS2) * np.sin(z4S1[:, 3]) * np.cos(IrealS2)),
         z4S1[:, 0] * (np.cos(OmegastorS2) * np.cos(z4S1[:, 3]) - np.sin(OmegastorS2) * np.sin(z4S1[:, 3]) * np.cos(IrealS2)),
         color="black", label='Fitted orbit S4716')
plt.errorbar(parsec(RAS2, R0real), parsec(DECS2, R0real), xerr=parsec(RAerrorS2, R0real),
             yerr=parsec(DECerrorS2, R0real), fmt='.',color='violet',ecolor='fuchsia',capsize=2,ls='none',label='S4716 Data Points')
plt.xlabel('Right ascension [pc]')
plt.ylabel('Declination [pc]')
plt.plot(0,0,'ko',markeredgecolor='orange',label='Sgr. A*')         #plots a black point with orange edge to show the SMBH
plt.legend()
plt.grid(alpha=0.5)
plt.axis('equal')

end=time.time()
print('Duration =',(end-start)/60,'min')

plt.show()