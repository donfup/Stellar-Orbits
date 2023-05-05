import numpy as np                                  #takes care of the math
import matplotlib.pyplot as plt                     #allows us to plot
import pandas as pd                                 #pandas lets us import from a excel-file
import kepler                                       #package for solving Kepler's eq's to obtain the eccentric anomaly
import math                                         #used for math.isnan
from iminuit import Minuit                          #package for minimizing the chi square function
import corner                                       #package for making cornerplots
from scipy.integrate import odeint                  #package for solving ordinary differential equations

print('Hello my friend, we are going to plot Kepler Orbits! B-)')

#importing and reading the excel with the data of the star orbits
#data = pd.read_excel('multipleorbitdata.xlsx')
data = pd.read_excel('OrbitdataS4716.xlsx')
#### Next up: defining the needed functions ####

#The mean motion, n, of S2
def n(a,M_bh):
    n=np.sqrt((G*M_bh)/a**3)                   #the unit for n is [s^-1]
    return n

#def n(a,M_bh):
#    n=(a**3/(G*M_bh))**(-1/2)                #another way to write the mean motion, doesn't matter which is used
#    return n

#The mean anomaly, M
def M(tp,t,M_bh,a):
    M=n(a,M_bh)*(t-tp)
    return M

#The eccentric anomaly, E
def E(tp,t,M_bh,a,e):
    E=kepler.solve(M(tp,t,M_bh,a),e)
    return E

#The term [(e+1)/(e-1)]^(1/2) is defined to make it easier further on
def term_e(e):
    term_e=((1+e)/(1-e))**(1/2)
    return term_e

#The distance to the focal point (i.e. the black hole), r(t)
def r(tp,t,M_bh,a,e):
    r=a*(1-e*np.cos(E(tp,t,M_bh,a,e), dtype=np.float64))
    return r

#The true anomaly, theta(t)
def theta(tp,t,M_bh,a,e):
    theta=2*np.arctan(term_e(e)*np.tan(E(tp,t,M_bh,a,e)/2, dtype=np.float64), dtype=np.float64)
    return theta

#The velocity, rdot(t)
def rdot(tp,t,M_bh,a,e):
    rdot=(n(a,M_bh)*(a**2)*e/r(tp,t,M_bh,a,e))*np.sin(E(tp,t,M_bh,a,e), dtype=np.float64)
    return rdot

#The angular velocity, thetadot(t)
def thetadot(tp,t,M_bh,a,e):
    thetadot=((1-e**2)**(1/2)*n(a,M_bh)*(a**2))/r(tp,t,M_bh,a,e)**2
    return thetadot

#The time (Johan har lavet den her, men forstår den ikke helt endnu)
#1992.224 er det årstal det første datapunkt er taget på.
#def timecon(my_list):
#    return [x-1992.224 for x in my_list]

#For S4716 the first data point is from 2003.44
def timecon(my_list):
    return [x-2003.44 for x in my_list]

#convert velocity from [km/s] to [pc/year]
#3.24077929*10^(-14) pc/km
def velparsectoyear(my_list):
    return [x*365.25*24*60*60*3.24077929*10**(-14) for x in my_list]

#convert from [mas] to [pc] by using the formula for linear diameter, D=d*r
#It's just trigonometry, tan(d)=D/r. But since d is a very small angle, then tan(d)≈d, => D=
# The angular diameter, d, is in radians and r is the distance to the object,
# in this case R0, the distance to the center of our Galaxy.
#first we calculate from [mas] to [rad] and then we multiply by R0
def parsec(my_list,R0):                                         #gets the same value as Johan
    return [x*1/1000*1/3600*(np.pi/180)*R0 for x in my_list]    #so this calculation is also correct

def power(my_list):
    return[x**2 for x in my_list]

#orbit data for the S4716 star
RAS2 = list(data['RA(mas)S4716'])                      #mas=micro-arc-seconds
RAS2_error = list(data['ErrorRA(mas)S4716'])
DeclS2 = list(data['Decl(mas)S4716'])
DeclS2_error = list(data['ErrorDecl(mas)S4716'])
Time = list(data['Time'])
TimeS2 = timecon(Time)

#Below [x for x in example if str(x) != 'nan'] is a sort of for loop
#that makes a list where only the numbers is put into the list
#the 'nan' (not a number) is left out of the list.
# != "is not equal to", nan = not a number
#RAS2re = [x for x in RAS2 if str(x) != 'nan']
#RAS2re_error = [x for x in RAS2_error if str(x) != 'nan']
#DeclS2re = [x for x in DeclS2 if str(x) != 'nan']
#DeclS2re_error = [x for x in DeclS2_error if str(x) != 'nan']
RAS2re=RAS2
RAS2re_error=RAS2_error
DeclS2re=DeclS2
DeclS2re_error=DeclS2_error

print('RAS2re =',RAS2re)
print('RAS2re_error =',RAS2re_error)
print('DeclS2re =',DeclS2re)
print('DeclS2re_error =',DeclS2re_error)

#Now we use The Method of Chi square.
#In the chi square method we want to minimize X^2=\sum[(O_i-C_i)^2/sigma^2]
#Where sigma^2 is the variance , O_i is the data (observed) values and C_i is the predicted values from theory
#Chi square is what we want to minimize to get the best possible fit for our Kepler Orbit
#This minimisation will be done with the Minuit function from the iminuit package (further down)

def ChiSquare(M_bh,R0,omega,Omega,I,tp,a,e):
    listS2_rad=[]                           #empty list for r
    listS2_theta=[]                         #empty list for theta

    for i in range(len(DeclS2)):            #len() return the number of elements in a list
        if math.isnan(DeclS2[i]):           #isnan "is not a number"
            continue                        #continue takes the loop directly to next iteration when DeclS2[i] isnan
        else:
            rad_S2=r(tp,TimeS2[i],M_bh,a,e)                   #predicted r(t) values
            theta_S2=theta(tp,TimeS2[i],M_bh,a,e)             #predicted theta(t) values
            listS2_rad.append(rad_S2)                         #.append adds element to an existing list
            listS2_theta.append(theta_S2)

    colS2_rad=np.transpose(listS2_rad)                         #transposes the lists so they become columns
    colS2_theta=np.transpose(listS2_theta)                     #instead of rows (list)

    #calculating the residuals for Declination and Right Ascension
    #the residual is the distance from the observed point to the calculated (theoretical point)
    #since it is chisquare, then the residual is divided with the variance (error squared)
    #the predicted locations of the stars is found by the formulas from "Orbital Elements"
    residualDeclS2=np.sum((parsec(DeclS2re,R0)-colS2_rad*(np.cos(Omega, dtype=np.float64)*np.cos(omega+colS2_theta, dtype=np.float64)-np.sin(Omega, dtype=np.float64)*np.sin(omega+colS2_theta, dtype=np.float64)*np.cos(I, dtype=np.float64)))**2/power(parsec(DeclS2re_error,R0)), dtype=np.float64)
    residualRAS2=np.sum((parsec(RAS2re,R0)-colS2_rad*(np.sin(Omega, dtype=np.float64)*np.cos(omega+colS2_theta, dtype=np.float64)+np.cos(Omega, dtype=np.float64)*np.sin(omega+colS2_theta, dtype=np.float64)*np.cos(I, dtype=np.float64)))**2/power(parsec(RAS2re_error,R0)), dtype=np.float64)

    residual=residualDeclS2+residualRAS2
    return residual

#    residualDeclS2=np.sum((parsec(DeclS2re,R0)-colS2_rad*(np.cos(Omega_S2, dtype=np.float64)*np.cos(omega_S2+colS2_theta, dtype=np.float64)-np.sin(Omega_S2, dtype=np.float64)*np.sin(omega_S2+colS2_theta, dtype=np.float64)*np.cos(I_S2, dtype=np.float64)))**2/power(parsec(DeclS2re_error,R0)), dtype=np.float64)
#    residualRAS2=np.sum((parsec(RAS2re,R0)-colS2_rad*(np.sin(Omega_S2, dtype=np.float64)*np.cos(omega_S2+colS2_theta, dtype=np.float64)+np.cos(Omega_S2, dtype=np.float64)*np.sin(omega_S2+colS2_theta, dtype=np.float64)*np.cos(I_S2, dtype=np.float64)))**2/power(parsec(RAS2re_error,R0)), dtype=np.float64)

#constants
G=6.6743*10**(-11)*(3.24*10**(-17))**3*((365.25*24*60*60)**2)          #The gravitational constant [m^3*kg^-1*s^-2] converted to [pc^3*kg^-1*yr^-2]
G=G*1.989*10**30                                                       #Converts unit for G to [pc^3*M_odot^-1*yr^-2]
print('G=',G)
#M_Sun=1.989*10**30                                                    #[kg] gives too high number for programming if this is number is in the code

#The codes needs some initial guess values to run the programme from, g=guess
#M_bhg=4.28*10**6                                                       #M_bh in solar masses (from Gillessen et. al)
M_bhg=4*10**6
#R0g=8320                                                               #distance to center of Galaxy from Earth in [pc]
R0g=8000
#omega_S2g=0.0012                                                     #omega guess
omega_S2g=0.0013
Omega_S2g=2.65                                                     #Omega guess
I_S2g=2.814                                                        #Inklination guess
tpg=2                                                                  #maybe tpg=15 as Johan did (tp guess)
a_S2g=0.00193                                                            #almost the same as with the below method
#a_S2g=0.1255*(1/3600)*(np.pi/180)                                    #semi-major axis of S2 converted to [rad] from [''] (1°/3600'')
#a_S2g=R0g*np.tan(a_S2g)                                                 #semi-major axis is calculated in [pc] (see notes)
e_S2g=0.756                                                           #eccentricity guess
#e_S2g=0.8839                                                           #eccentricity of S2 (from Gillessen et. al)

#den laver et dårligt plot når a beregnes med mange decimaler!! brug derfor a_S2g=0.5

#god ide at definere dem som værdier og give nyt navn, så programmet ikke misforstår det

#Reduced chi square is used
#The degrees of freedom, \nu=n-m, is the number of observations n minus the number of fitted parameters m
#n is len(RAS2re)*2 because we have both the RA and Decl observations.
#We have 8 fitted parameters: e_S2,omega_S2 and so on. (see them below)
print('len(RAS2re)=',len(RAS2re))
print('Reduced chi square before, Chi^2 =',ChiSquare(M_bhg,R0g,omega_S2g,Omega_S2g,I_S2g,tpg,a_S2g,e_S2g)/(len(RAS2re)*2-8))

#The Minuit package is used for minimizing the chi square function
ms=Minuit(ChiSquare, M_bh=M_bhg,R0=R0g,omega=omega_S2g,Omega=Omega_S2g,I=I_S2g,tp=tpg,a=a_S2g,e=e_S2g)
ms.limits['e']=(0,1)
ms.limits['omega']=(0,2*np.pi)
ms.limits['Omega']=(0,2*np.pi)
ms.limits['I']=(0,2*np.pi)
ms.limits['R0']=(7500,8500)
ms.limits['M_bh']=(3.75*(10**6),4.5*(10**6))
ms.limits['a']=(0,1)
ms.limits['tp']=(0,15)
tttts=ms.simplex().migrad()                 #vidst bare et random navn Johan gav den
print(tttts)

#Parameters
print('Initial M_bh =',M_bhg,           '   Fitted M_bh =',     ms.values[0])
print('Initial R0 =',R0g,               '   Fitted R0 =',       ms.values[1])
#Orbit parameters for S4716
print('Initial omega_S4716 =',omega_S2g,'   Fitted omega_S4716 =', ms.values[2])
print('Initial Omega_S4716 =',Omega_S2g,'   Fitted Omega_S4716 =', ms.values[3])
print('Initial I_S4716 =',I_S2g,        '   Fitted I_S4716 =',     ms.values[4])
print('Initial tp =',tpg,               '   Fitted tp =',       ms.values[5])
print('Initial a_S4716 =',a_S2g,        '   Fitted a_S4716 =',     ms.values[6])
print('Initial e_S4716 =',e_S2g,        '   Fitted e_S4716 =',     ms.values[7])


M_bh_real=ms.values[0]
R0_real=ms.values[1]
omega_S2_real=ms.values[2]
Omega_S2_real=ms.values[3]
I_S2_real=ms.values[4]
tp_real=ms.values[5]
a_S2_real=ms.values[6]
e_S2_real=ms.values[7]


#Calculate the reduced Xi^2 value for the fitted values
print('Reduced chi square after, X_nu^2 =',ChiSquare(M_bh_real,R0_real,omega_S2_real,Omega_S2_real,I_S2_real,tp_real,a_S2_real,e_S2_real)/(len(RAS2re)*2-8))

#Finally, it's time to make the plot!
time=np.linspace(0,30,1000)
radiplot=[]
thetaplot=[]

for i in range(1000):
    dt=i*30/1000
    radiplot.append(r(tp_real,dt,M_bh_real,a_S2_real,e_S2_real))
    thetaplot.append(theta(tp_real,dt,M_bh_real,a_S2_real,e_S2_real))

radiplot=np.transpose(radiplot)
thetaplot=np.transpose(thetaplot)


plt.plot(radiplot*(np.sin(Omega_S2_real,dtype=np.float64)*np.cos(omega_S2_real+thetaplot,dtype=np.float64)+np.cos(Omega_S2_real,dtype=np.float64)*np.sin(omega_S2_real+thetaplot,dtype=np.float64)*np.cos(I_S2_real,dtype=np.float64)),
         radiplot*(np.cos(Omega_S2_real,dtype=np.float64)*np.cos(omega_S2_real+thetaplot,dtype=np.float64)-np.sin(Omega_S2_real,dtype=np.float64)*np.sin(omega_S2_real+thetaplot,dtype=np.float64)*np.cos(I_S2_real,dtype=np.float64)),
         color='grey',label='Fitted orbit S4716')

#this one plots the Kepler Orbit for the guessed values
plt.plot(radiplot*(np.sin(Omega_S2g,dtype=np.float64)*np.cos(omega_S2g+thetaplot,dtype=np.float64)+np.cos(Omega_S2g,dtype=np.float64)*np.sin(omega_S2g+thetaplot,dtype=np.float64)*np.cos(I_S2g,dtype=np.float64)),
        radiplot*(np.cos(Omega_S2g,dtype=np.float64)*np.cos(omega_S2g+thetaplot,dtype=np.float64)-np.sin(Omega_S2g,dtype=np.float64)*np.sin(omega_S2g+thetaplot,dtype=np.float64)*np.cos(I_S2g,dtype=np.float64)),
       'r')

plt.errorbar(parsec(RAS2,R0_real),parsec(DeclS2,R0_real),xerr=parsec(RAS2_error,R0_real),yerr=parsec(DeclS2_error,R0_real),fmt='.',color='fuchsia',ecolor='violet',capsize=2,ls='none',label='S4716 Data points') #ecolor='lightblue'
plt.plot(0,0,'ko',markeredgecolor='orange',label='Sgr. A*')         #plots a black point with orange edge to show the SMBH
#plt.title('Predicted Kepler orbit for S2')
plt.xlabel('Right ascension [pc]')
plt.ylabel('Declination [pc]')
plt.grid(alpha=0.5)
plt.axis('equal')
plt.legend()
plt.show()

