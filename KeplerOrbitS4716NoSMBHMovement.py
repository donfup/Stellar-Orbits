# Import the required modules
from __future__ import print_function, division
import numpy as np
#from math import pi
from scipy.constants import *
#from cProfile import label
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import pandas as pd
#from scipy.optimize import minimize
#from scipy.stats import chisquare
from iminuit import Minuit
#import emcee
#import corner
import math
import kepler



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


#orbit data for S4716
RAS2=list(dataorbit['RA(mas)S4716'])
DECS2=list(dataorbit['Decl(mas)S4716'])
RAerrorS2=list(dataorbit['ErrorRA(mas)S4716'])
DECerrorS2=list(dataorbit['ErrorDecl(mas)S4716'])
#VLRS2=list(dataorbit['Radialvelocity(kmpersec)S2'])
#errorVLSR2=list(dataorbit['Error Radial velocity (kmpersec)S2'])
tid2=list(dataorbit['Time'])
tidS2=timecon(tid2)
#VLRS2=velparsectoyear(VLRS2)
#errorVLSRS2=velparsectoyear(errorVLSR2)

RAS2re= [x for x in RAS2 if str(x) != 'nan']
DECS2re= [x for x in DECS2 if str(x) != 'nan']
RAerrorS2re= [x for x in RAerrorS2 if str(x) != 'nan']
DECerrorS2re= [x for x in DECerrorS2 if str(x) != 'nan']


def LSQXkep(Mbh1,Ro1,oomegaS2,OmegaS2,IS2,tp,a1,e1):                  #least square which we want to minimize
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
Mbh=4.023*10**6             #black hole mass in solar mass
R1=8000                     #distance to orbit
oomegaS2g=0.0012            #omega=0.073°= 0.001 rad
OmegaS2g=2.65               #Omega=151.54°=2.65
IS2g=2.814                  #i=161.24° = 2.814 rad
tpg=2
a1g=0.00193
e1g=0.756

print('Chi square before',LSQXkep(Mbh,R1,oomegaS2g,OmegaS2g,IS2g,tpg,a1g,e1g)/((len(RAS2re)+len(DECS2re))-8))


print('fit all same time')
ms = Minuit(LSQXkep, Mbh1=Mbh,Ro1=R1,oomegaS2=oomegaS2g,OmegaS2=OmegaS2g,IS2=IS2g,tp=tpg,a1=a1g,e1=e1g)
#ms.limits['e1'] = (0.756,0.756) #Constant limits obtained from article Peissker et al.
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
#ms.limits['tp']= (0,10)
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

print('Chi square after',LSQXkep(Mbhreal,R0real,oomegareal,Omegareal,IS2real,tpreal,areal,ereal)/(len(RAS2re)+len(DECS2re))-8)
#print('længde:',len(RAS2re)*2), just to check the length
time1=np.linspace(0,10,1000)

radiplot=[]
thetaplot=[]
for i in range(1000):
    dt=i*10/1000
    radiplot.append(keprad(tpreal,dt,Mbhreal,areal,ereal))
    thetaplot.append(theta1(tpreal,dt,Mbhreal,areal,ereal))

radiplot=np.transpose(radiplot)
thetaplot=np.transpose(thetaplot)

plt.plot(radiplot*(np.sin(Omegareal, dtype=np.float64)*np.cos(oomegareal+thetaplot, dtype=np.float64)+np.cos(Omegareal, dtype=np.float64)*np.sin(oomegareal+thetaplot, dtype=np.float64)*np.cos(IS2real, dtype=np.float64)),
         radiplot*(np.cos(Omegareal, dtype=np.float64)*np.cos(oomegareal+thetaplot, dtype=np.float64)-np.sin(Omegareal, dtype=np.float64)*np.sin(oomegareal+thetaplot, dtype=np.float64)*np.cos(IS2real, dtype=np.float64)),color='grey',label='Fitted orbit S4716')
plt.errorbar(parsec(RAS2,R0real),parsec(DECS2,R0real),xerr=parsec(RAerrorS2,R0real),yerr=parsec(DECerrorS2,R0real),marker='.',capsize=2, ls='none',color="fuchsia",ecolor='violet',label='S4716 Data points')

plt.xlabel('Right ascension [pc]')
plt.ylabel('Declination [pc]')
plt.plot(0,0,'ko',markeredgecolor='orange',label='Sgr. A*')         #plots a black point with orange edge to show the SMBH
plt.legend()
plt.grid(alpha=0.5)
plt.axis('equal')
plt.show()