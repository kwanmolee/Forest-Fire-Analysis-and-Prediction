'''
@author: Kunmao Li


1. new varaible generated:
  1). ln(area+1): float64
  Based on the characteristics of variable 'area' from the plots, remodel 'area' with logarithm transform
  2). FIRE: str
  Fire indicator: fire/no fire ,indicating whether there's fire

2. 1D Visualization 
  Figure 1-13: 13 varaibles inside the dataset 
  Figure 14: 1 new varible ln(area+1) generated 
  For every figure, 3 plots generated: 
    Univariate Histogram: x-axis: the variable itself ; y-axis: the freqeuncy/times of the data 
    Kernel Density Estimation: x-axis: the variable itself; y-axis:the probability density of the variable estimated
    Single Parameter Distribution: the combination of above two plots for better observation with classification and comparison

3. 2D Visualization(DMC & temp/RH/rain)
  Figure 15: pairplots(diagonal:the kernel density estimation; non-diagonal: scatter plots with linear regression line given)
    also, the data has been classified according to variable 'FIRE' and displayed green when no fire and red when on fire
  Figure 16: pairplots(diagonal: single variable distribution; non-diagonal: kernel density plots of two variables)
  Figure 17-20: more complicated, detailed and advanced plots given: 
    kde plots: demonstrate the density distribution between two variables with color change according to the color palatte set
    scatter plots: spot the data easily 
    axis: also display the distribution line of each variable with different color 
    color bar: relate the color shade/range with data values for identifying the level of density 

4. 3D Visualization(DMC & any two from temp,RH,rain)
  Figure 21-23:histograms: x,y-axis: any two from temp,RH,rain; z-axis: DMC
  classify the histograms by variable 'FIRE': (red histograms:fire; green histograms:no fire)

5. 4D Visualization
  Figrue 24Realize the 4-th dimension by marking the histograms with different colors based on the interval values of DMC
  x,y,z-axis: temp,RH,rain 

'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D


fire = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')
fire.month=fire.month.map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
fire.day=fire.day.map({'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7})
data=pd.DataFrame(fire,columns=['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'])

'''
1D: Univariate Histograms,Kernel Density Estimation and Single Parameter Distribution
'''
n=1
for i in ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']:
	plt.figure(n)
	plt.subplot(131)# Univariate Histograms
	plt.title('Univariate Histogram', fontsize=14, position=(0.5,1.05))
	plt.ylabel('Frequency')
	sns.distplot(fire[i],kde=False,color='r',rug=True)
	plt.subplot(132) # Kernel Density Estimation
	plt.title('Kernel Density Estimation', fontsize=14, position=(0.5,1.05))
	plt.ylabel('Probability')
	sns.distplot(fire[i],hist=False,color='g',kde_kws={'shade':True})
	plt.subplot(133) # Single Parameter Distribution
	plt.title('Single Parameter Distribution', fontsize=14, position=(0.5,1.05))
	plt.ylabel('Probability')
	sns.distplot(fire[i],color='m')
	n += 1

'''
According to the above three plots of 'area' in figure 13, the parameter is pretty distorted towards 0.0
Thus using logrithm transform ln(area+1) to remodel 'area' and generate the new varibale 'ln(area+1)'
'''
fire['ln(area+1)']=np.log(fire['area']+1)
plt.figure(14)
plt.subplot(131)# Univariate Histograms
plt.title('Univariate Histogram', fontsize=14, position=(0.5,1.05))
plt.xlabel('ln(area+1)')
plt.ylabel('Frequency')
sns.distplot(fire['ln(area+1)'],kde=False,color='r',rug=True)
plt.subplot(132) #Kernel Density Estimation
plt.title('Kernel Density Estimation', fontsize=14, position=(0.5,1.05))
plt.ylabel('Probability')
plt.xlabel('ln(area+1)')
sns.distplot(fire['ln(area+1)'],hist=False,color='g',kde_kws={'shade':True})
plt.subplot(133) # Single Parameter Distribution
plt.title('Single Parameter Distribution', fontsize=14, position=(0.5,1.05))
plt.ylabel('Probability')
plt.xlabel('ln(area+1)')
sns.distplot(fire['ln(area+1)'],color='m')


'''
Based on the  forest Fire Weather Index system, DMC is influenced by temperature(temp), reletive humidity(RH) and rain
To clarify the correlation between DMC and the rest variables initially, create the pairplots.
figure 15: 16 pairplots totally
         :4 Diagonal plots: kde plots indicating the probability density of the variable
         :14 Non-Diagonal plots: scatter diagrams with linear regression line generated                      
         p.s.: for better visualization, adding variable'FIRE',which indicates 'fire' or 'no fire'
               and mark the data values with two colors(red for fire;green for no fire)
figrue 16: 16 pairplots totally
         :4 Diagonal plots: single paramter distribution
         :14 Non-Diagonal plots: kdeplots of two variables(the core indicates the highest density)
'''

plt.figure(15)
df1=pd.DataFrame(fire,columns=['DMC','temp','RH','rain','ln(area+1)'])

# set [ln(area+1)>0]=1 ; [ln(area+1)=0]=0 
df1['FIRE'] = np.where(df1['ln(area+1)']>0, 'fire', 'no fire')
sns.pairplot(df1,vars=['DMC','temp','RH','rain'],kind='reg',hue='FIRE',palette='hls',markers=["o", "x"])
plt.show()

plt.figure(16)
g = sns.PairGrid(df1)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot,cmap="Blues_d",n_levels=20)
plt.show()

'''
2D: 
figure 17: DMC-temp & DMC-RH & DMC-rain 
figure 18-20: DMC-temp / RH / rain
p.s.: for the plots related to rain, for better observation, adjustment is needed since scatters were initially distributed around 0.00
'''
plt.figure(17)
sub=131
for i in ['temp','RH','rain']:
	plt.subplot(sub)
	plt.title('DMC-'+i, fontsize=14, position=(0.5,1.05))
	sns.kdeplot(df1[i],df1['DMC'], # demonstrate the probability distribution of two variables
           cbar = True,    # display color bar
           shade = True,   # display shades
           cmap = 'Reds',  # set the color palatte
           shade_lowest=False,  # not display periphery color/shade
           n_levels = 40   # number of curves, the higher, the smoother
           )# the color change indicates the change of density
	plt.grid(linestyle = '--')
	plt.scatter(df1[i], df1['DMC'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
	sns.rugplot(df1[i], color='g', axis='x',alpha = 0.5)
	sns.rugplot(df1['DMC'], color='r', axis='y',alpha = 0.5)
	sub += 1
plt.show()

plt.figure(18) # DMC-temp
plt.title('DMC-temp', fontsize=14, position=(0.5,1.05))
pal=sns.cubehelix_palette(8, gamma=2,as_cmap=True)
sns.kdeplot(df1['temp'],df1['DMC'],cbar = True,shade = True,cmap = pal,shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
plt.scatter(df1['temp'], df1['DMC'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(df1['temp'], color="orange", axis='x',alpha = 0.5)
sns.rugplot(df1['DMC'], color="purple", axis='y',alpha = 0.5)
plt.show()

plt.figure(19) # DMC-RH
plt.title('DMC-RH', fontsize=14, position=(0.5,1.05))
pal=sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap=True)
sns.kdeplot(df1['RH'],df1['DMC'],cbar = True,shade = True,cmap = pal,shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
plt.scatter(df1['RH'], df1['DMC'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(df1['RH'], color="blue", axis='x',alpha = 0.5)
sns.rugplot(df1['DMC'], color="green", axis='y',alpha = 0.5)

plt.figure(20) # DMC-rain
plt.title('DMC-rain', fontsize=14, position=(0.5,1.05))
sns.kdeplot(df1['rain'],df1['DMC'],cbar = True,shade = True,cmap = 'Purples',shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
plt.scatter(df1['rain'], df1['DMC'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(df1['rain'], color="orange", axis='x',alpha = 0.5)
sns.rugplot(df1['DMC'], color="purple", axis='y',alpha = 0.5)
plt.show()

fig1=plt.figure(21)
ax = fig1.add_subplot(111, projection='3d')   

x,y,z = np.array(fire['temp']),np.array(fire['rain']),np.array(fire['DMC'])

x = x.flatten('F')   
y = y.flatten('F')

'''
mark the bars with two colors for better observation
red: fire
green: no fire
'''
q=df1['FIRE']
C = []  
for a in q:
    if a == 'fire':
        C.append('red')
    else:
        C.append('green')

dx = 0.6 * np.ones_like(x) # set the width of the histograms, the constant can be adjusted based on observation of plots
dy = 0.2 * np.ones_like(y)
dz = abs(z) * z.flatten()
dz = dz.flatten() / abs(z)
z = np.zeros_like(z)

ax.set_title('temp-rain-DMC')
ax.set_xlabel('temp')
ax.set_ylabel('rain')
ax.set_zlabel('RH')
plt.axis([0,35,-6,6])#set the interval of axises to move the bunch of histograms to the centeral area for better observation

ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')



fig2=plt.figure(22)
ax = fig2.add_subplot(111, projection='3d')   
x,y,z = np.array(fire['temp']),np.array(fire['RH']),np.array(fire['DMC'])

x = x.flatten('F')   
y = y.flatten('F')

dx = 0.6 * np.ones_like(x)# set the width of the histograms, the constant can be adjusted based on observation of plots
dy = 0.2 * np.ones_like(y)
dz = abs(z) * z.flatten()
dz = dz.flatten() / abs(z)
z = np.zeros_like(z)

ax.set_title('temp-RH-DMC')
ax.set_xlabel('temp')
ax.set_ylabel('RH')
ax.set_zlabel('DMC')

ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')


fig3=plt.figure(23)
ax = fig3.add_subplot(111, projection='3d')   
x,y,z = np.array(fire['rain']),np.array(fire['RH']),np.array(fire['DMC'])

x = x.flatten('F')   
y = y.flatten('F')

dx = 0.1 * np.ones_like(x)# set the width of the histograms, the constant can be adjusted based on observation of plots
dy = 0.2 * np.ones_like(y)
dz = abs(z) * z.flatten()
dz = dz.flatten() / abs(z)
z = np.zeros_like(z)

ax.set_title('rain-RH-DMC')
ax.set_xlabel('rain')
ax.set_ylabel('RH')
ax.set_zlabel('DMC')
plt.axis([-6,6,20,100]) #set the interval of axises to move the bunch of histograms to the centeral area for better observation
ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')
plt.show()

'''
4D:
axies:x: temp y:rain z:RH

4th demension: DMC 
displaying DMC by mark the bars with different colors based on the value intervals of DMC

'''
fig = plt.figure(24)
ax = fig.add_subplot(111, projection='3d')   

x,y,z = np.array(fire['temp']),np.array(fire['rain']),np.array(fire['RH'])

x = x.flatten('F')   
y = y.flatten('F')

# Based on the interval values of DMC, mark the histograms with different colors
q=fire['DMC']
C = [] # the list serving as the color palatte 
for a in q:
    if a < 50:
        C.append('orange') # for data from DMC <50, mark it with orange 
    elif a < 100:
        C.append('blue') # for data from DMC belonging to [50,100), mark it with blue
    elif a < 150:
        C.append('purple') # for data from DMC belonging to [100,150), mark it with purple
    elif a < 200:
        C.append('red') # for data from DMC belonging to [150,200), mark it with red
    elif a > 200:
        C.append('green') # for data from DMC >200, mark it with greem

#dx，dy，dz:length width altitude
dx = 0.6 * np.ones_like(x)# set the width of the bars, the constant can be adjusted based on observation of plots
dy = 0.2 * np.ones_like(y)
dz = abs(z) * z.flatten()
dz = dz.flatten() / abs(z)
z = np.zeros_like(z)


ax.set_title('DMC-temp-rain-RH')
ax.set_xlabel('temp')
ax.set_ylabel('rain')
ax.set_zlabel('RH')
plt.axis([0,35,-6,6])#set the interval of axises to move the bunch of histograms to the centeral area for better observation

ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average')
plt.show()

