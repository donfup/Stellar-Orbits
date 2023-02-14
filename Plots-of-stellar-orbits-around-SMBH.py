import numpy as np                                  #takes care of the math
import matplotlib.pyplot as plt                     #allows us to plot
import pandas as pd                                 #pandas lets us import from a excel-file

#Courtesy of Johan Sieborg, Frederik Finseth & Frederik Heyde-Petersen.

#importing and reading the excel with the data of the star orbits
data = pd.read_excel('multipleorbitdata.xlsx')
#print(data)                                        #if we wish to print the data

#Data for the S1 star
RAS1 = list(data['RA(mas)S1'])
RAS1_error = list(data['ErrorRA(mas)S1'])
DeclS1 = list(data['Decl(mas)S1'])
DeclS1_error = list(data['ErrorDecl(mas)S1'])

#Data for the S2 star
RAS2 = list(data['RA(mas)S2'])                      #mas=micro-arc-seconds
RAS2_error = list(data['ErrorRA(mas)S2'])
DeclS2 = list(data['Decl(mas)S2'])
DeclS2_error = list(data['ErrorDecl(mas)S2'])

#Data for the S4 star
RAS4 = list(data['RA(mas)S4'])
RAS4_error = list(data['ErrorRA(mas)S4'])
DeclS4 = list(data['Decl(mas)S4'])
DeclS4_error = list(data['ErrorDecl(mas)S4'])

#Data for the S8 star
RAS8 = list(data['RA(mas)S8'])
RAS8_error = list(data['ErrorRA(mas)S8'])
DeclS8 = list(data['Decl(mas)S8'])
DeclS8_error = list(data['ErrorDecl(mas)S8'])

#Data for the S9 star
RAS9 = list(data['RA(mas)S9'])
RAS9_error = list(data['ErrorRA(mas)S9'])
DeclS9 = list(data['Decl(mas)S9'])
DeclS9_error = list(data['ErrorDecl(mas)S9'])

#Data for the S14 star
RAS14 = list(data['RA(mas)S13'])
RAS14_error = list(data['ErrorRA(mas)S13'])
DeclS14 = list(data['Decl(mas)S13'])
DeclS14_error = list(data['ErrorDecl(mas)S13'])

#Data for the S14 star
#RAS14 = list(data['RA(mas)S14'])
#RAS14_error = list(data['ErrorRA(mas)S14'])
#DeclS14 = list(data['Decl(mas)S14'])
#DeclS14_error = list(data['ErrorDecl(mas)S14'])

#Data for the S31 star
#RAS31 = list(data['RA(mas)S31'])
#RAS31_error = list(data['ErrorRA(mas)S31'])
#DeclS31 = list(data['Decl(mas)S31'])
#DeclS31_error = list(data['ErrorDecl(mas)S31'])

#Data for the S38 star
RAS38 = list(data['RA(mas)S38'])
RAS38_error = list(data['ErrorRA(mas)S38'])
DeclS38 = list(data['Decl(mas)S38'])
DeclS38_error = list(data['ErrorDecl(mas)S38'])

#Data for the S55 star
RAS55 = list(data['RA(mas)S55'])
RAS55_error = list(data['ErrorRA(mas)S55'])
DeclS55 = list(data['Decl(mas)S55'])
DeclS55_error = list(data['ErrorDecl(mas)S55'])

#Data for the newly found S4716 star!
dataS4716 = pd.read_excel('OrbitdataS4716.xlsx')
RAS4716 = list(dataS4716['RA(mas)S4716'])
RAS4716_error = list(dataS4716['ErrorRA(mas)S4716'])
DeclS4716 = list(dataS4716['Decl(mas)S4716'])
DeclS4716_error = list(dataS4716['ErrorDecl(mas)S4716'])


#Convertion from mas to mpc
def mpc(my_list):
    return [np.tan(((x / 1000 / 3600) * 0.0174532925) / 2) * 8.32*(10**6) * 2 for x in my_list]


#Plotting the orbits of the S-stars around the SMBH
#plt.errorbar(RAS1,DeclS1,xerr=RAS1_error,yerr=DeclS1_error,fmt='.',color='orange',ecolor='gold',capsize=2,label='S1')
plt.errorbar(RAS2,DeclS2,xerr=RAS2_error,yerr=DeclS2_error,fmt='.',color='mediumseagreen',ecolor='limegreen',capsize=2,label='S2')
#plt.errorbar(RAS2,DeclS2,xerr=RAS2_error,yerr=DeclS2_error,fmt='.',color='grey',ecolor='silver',capsize=2,label='S2')
#plt.errorbar(RAS4,DeclS4,xerr=RAS4_error,yerr=DeclS4_error,fmt='.',color='darkkhaki',ecolor='khaki',capsize=2,label='S4')
#plt.errorbar(RAS9,DeclS9,xerr=RAS9_error,yerr=DeclS9_error,fmt='.',color='purple',ecolor='pink',capsize=2,label='S9')
#plt.errorbar(RAS14,DeclS14,xerr=RAS14_error,yerr=DeclS14_error,fmt='.',color='firebrick',ecolor='lightgrey',capsize=2,label='S14')
plt.errorbar(RAS38,DeclS38,xerr=RAS38_error,yerr=DeclS38_error,fmt='.',color='royalblue',ecolor='cornflowerblue',capsize=2,label='S38')
plt.errorbar(RAS55,DeclS55,xerr=RAS55_error,yerr=DeclS55_error,fmt='.',color='chocolate',ecolor='peru',capsize=2,label='S55')
plt.errorbar(RAS4716,DeclS4716,xerr=RAS4716_error,yerr=DeclS4716_error,fmt='.',color='fuchsia',ecolor='violet',capsize=2,label='S4716')

#plots af S2 og S4716 i [mpc]
#plt.errorbar(mpc(RAS2),mpc(DeclS2),xerr=mpc(RAS2_error),yerr=mpc(DeclS2_error),fmt='b.',ecolor='lightblue',capsize=2,label='S2')
#plt.errorbar(mpc(RAS4716),mpc(DeclS4716),xerr=mpc(RAS4716_error),yerr=mpc(DeclS4716_error),fmt='.',color='purple',ecolor='pink',capsize=2,label='S4716')

plt.plot(0,0,'ko',markeredgecolor='orange',label='Sgr. A*')         #plots a black point with orange edge to show the SMBH
plt.xlabel('Right ascension [mas]')
plt.ylabel('Declination [mas]')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)                                 #alpha adds transparency to the grid
plt.axis('equal')
plt.show()

#np.sin kører i radianer og ikke i grader, vigtigt at vide!
print('sin(90°)=',np.sin(90))
print('sin(90°)=',np.sin(np.pi/2))

#matplotlib color