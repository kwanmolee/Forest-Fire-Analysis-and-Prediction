'''
@author: Kunmao Li

Figure 1-3: heat maps
Function: create the heat maps demonstrating the fire intensity of burned areas in the Montesinho park

Figure 1 (fire intensity: burned area of the forest)
Figure 2(fire intensity: burned area of the forest with logorithm transform):
1.  Red color : there is fire(area or ln(area+1)>0) 
	No color: no fire(area or ln(area+1)=0)
	p.s.:area is the variable indicating the burned area of the forest (in ha): 0.00 to 1090.84 
	     ln(area+1) is the logarithm transform of variable area
2.  The color bar next to the heat map indicates the change of intensity of fire in different areas within the park
	The darker/more saturated the red is, the intensity of fire in that area is higher
	The range of the color shade in the heat map is based on the actual value change of variables(area or ln(area+1))
3.	Axises for figure 1&2 are the same:
	X axis:spatial coordinate within the Montesinho park map: 1 to 9(the dataset divides the coordinatepark area into 1-9)
	Y axis:spatial coordinate within the Montesinho park map: 1 to 9(the dataset divides the coordinatepark area into 1-9)
	So the shape of heat map resembels the actual map of Montesinho park
4.	values:
	figure 1: heat map is set based on variable area
	figure 2: heat map is set based on variable ln(area+1)

Figure 3(temperature in summer months):
Reason: temperature is the major factor influencing forest fires(based on 1D-2D visulization), and temeperaturs are mainly high in summer months
so it makes sense to demonstrate the relationship between temperatures and summer months 
the output heatmap should quite similar to above two heatmaps which proves that summer months indicate high risks of forest fires
1.  Red color : temperature, and the color more saturated, the temperature higher
	No color: no temperature
2.  The color bar next to the heat map indicates the temperature range in different areas within the park
	The darker/more saturated the red is, the intensity of fire in that area is higher
	The range of the color shade in the heat map is based on the actual value change of temperatures in summer months
3.	Axises for figure 3:
	X axis:spatial coordinate within the Montesinho park map: 1 to 9(the dataset divides the coordinatepark area into 1-9)
	Y axis:spatial coordinate within the Montesinho park map: 1 to 9(the dataset divides the coordinatepark area into 1-9)
4.  values: temperatures in summer months

Figure 4: kernel density probability between month and temperature
Figure 5: 2D bars between month and temperature
Figure 6: 2D bars between month and normalized temperature
Figure 7: 2D bars demonstrating the fire in summer months and non-summer months
		red bars: summer months' fires ; blue bars: non-summer months' fires
Figure 8: 2D bars demonstrating the fire in weekdays and weekends
		red bars: weekday fires ; blue bars: weekend fires
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


fire = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')
fire.month=fire.month.map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
fire.day=fire.day.map({'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7})

fire['ln(area+1)']=np.log(fire['area']+1) # logorithm transform
fire['FIRE'] = np.where(fire['ln(area+1)']>0, 'fire', 'no fire') # convert burned area to 'fire' and other area without fire to 'no fire'

df1=pd.DataFrame(fire,columns=['X','Y','area'])
df2=pd.DataFrame(fire,columns=['X','Y','ln(area+1)'])
df1.head()
df2.head()
pt1 = df1.pivot_table(index='Y', columns='X', values='area') #set the table and group the variables
pt2 = df2.pivot_table(index='Y', columns='X', values='ln(area+1)') #set the table and group the variables
pt1.head()
pt2.head()

plt.figure(1) # heatmap1: values:area
f, ax1 = plt.subplots(figsize = (9, 9))
ax1.spines['bottom'].set_position(('data', 0)) # set the position for axis
ax1.spines['left'].set_position(('data', 0))

cmap = 'Reds' # choose the palatte for heat map
sns.heatmap(pt1, cmap = cmap, linewidths = 0.1,ax = ax1,annot=True, annot_kws={'size':9,'weight':'bold'})#annotate: add statistics in the box of heat map
plt.xlabel('X:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14,position=(0.6,1.05))
plt.ylabel('Y:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14)
plt.title('fire intensity: burned area of the forest',fontsize=14, position=(0.5,1.05))

plt.figure(2) # heatmap2: value:ln(area+1)
f, ax2 = plt.subplots(figsize = (9, 9))
ax2.spines['bottom'].set_position(('data', 0))
ax2.spines['left'].set_position(('data', 0))

cmap = 'Reds'# choose the palatte for heat map
sns.heatmap(pt2, cmap = cmap, linewidths = 0.1,ax = ax2,annot=True, annot_kws={'size':9,'weight':'bold'})#annotate: add statistics in the box of heat map
plt.xlabel('X:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14,position=(0.6,1.05))
plt.ylabel('Y:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14)
plt.title('fire intensity: burned area of the forest with logorithm transform',fontsize=14, position=(0.5,1.05))

plt.show()


'''
create new variable/column: 
summer_temp, which only displays the temperatures in summer months
non-summer_temp, which only displays the temperatures in non-summer months

'''
fire['summer_temp']=np.where(fire['temp'],0,0)
a=fire.loc[fire['month']<10]
a=a.loc[a['month']>4]
fire['summer_temp']=a['temp']
fire['summer_temp']=np.where(fire['summer_temp']>0,fire['summer_temp'],0)


df3=pd.DataFrame(fire,columns=['X','Y','summer_temp'])
df3.head()
pt3 = df3.pivot_table(index='Y', columns='X', values='summer_temp') #set the table and group the variables
pt3.head()
plt.figure(3) #heatmap3: values:temperature in summer months
f, ax3 = plt.subplots(figsize = (9, 9))
ax3.spines['bottom'].set_position(('data', 0))# set the position for axis
ax3.spines['left'].set_position(('data', 0))

cmap = 'Reds'# choose the palatte for heat map
sns.heatmap(pt3, cmap = cmap, linewidths = 0.1,ax = ax3,vmin=2.2,vmax=33.3,annot=pt3.values, annot_kws={'size':9,'weight':'bold'})
#vmin,vmax: the minimum and maximum values in heatmap(also to set the scope of color bar) 
#annotate: add statistics in the box of heat map
plt.xlabel('X:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14,position=(0.6,1.05))
plt.ylabel('Y:spatial coordinate within the Montesinho park map: 1 to 9',fontsize=14)
plt.title('temperature in summer months',fontsize=14, position=(0.5,1.05))
plt.show()




plt.figure(4) #month-temperature kde plots 
plt.title('month-temperature', fontsize=14, position=(0.5,1.05))
sns.kdeplot(fire['month'],fire['temp'], # demonstrate the probability distribution of two variables
           cbar = True,    # display color bar
           shade = True,   # display shades
           cmap = 'Reds',  # set the color palatte
           shade_lowest=False,  # not display periphery color/shade
           n_levels = 40   # number of curves, the higher, the smoother
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.scatter(fire['month'], fire['temp'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(fire['month'], color='g', axis='x',alpha = 0.5)
sns.rugplot(fire['temp'], color='r', axis='y',alpha = 0.5)
plt.show()

plt.figure(5) #month-RH kde plots 
plt.title('month-RH', fontsize=14, position=(0.5,1.05))
sns.kdeplot(fire['month'],fire['RH'], # demonstrate the probability distribution of two variables
           cbar = True,    # display color bar
           shade = True,   # display shades
           cmap = 'Oranges',  # set the color palatte
           shade_lowest=False,  # not display periphery color/shade
           n_levels = 40   # number of curves, the higher, the smoother
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.scatter(fire['month'], fire['temp'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(fire['month'], color='g', axis='x',alpha = 0.5)
sns.rugplot(fire['RH'], color='r', axis='y',alpha = 0.5)
plt.show()

def zscorenormalization(x):
	x=(x-np.min(x))/(np.max(x)-np.min(x))
	return x

plt.figure(6)#month-temperature bar plots
plt.title('month-temperature', fontsize=14, position=(0.5,1.05))
sns.barplot(x=fire['month'],y=fire['temp'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'r',  # set the color palatte
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.scatter(fire['month'], fire['temp'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(fire['month'], color='g', axis='x',alpha = 0.5)
sns.rugplot(fire['temp'], color='r', axis='y',alpha = 0.5)
plt.show()

fire['normalized_temp']=zscorenormalization(fire['temp'])

plt.figure(7)# month-normalized temperature  bar plots 
plt.title('month-normalized_temperature', fontsize=14, position=(0.5,1.05))
sns.barplot(x=fire['month'],y=fire['normalized_temp'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'blue',  # set the color palatte
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.scatter(fire['month'], fire['normalized_temp'], s=5, alpha = 0.5, color = 'k', marker='+') #scatter
sns.rugplot(fire['month'], color='g', axis='x',alpha = 0.5)
sns.rugplot(fire['normalized_temp'], color='r', axis='y',alpha = 0.5)
plt.show()

'''
create new variable/column: 
summer_fire, which only displays the fire in summer months
non-summer_temp, which only displays the fire in non-summer months

'''

fire['summer_fire']=a['ln(area+1)']
fire['summer_fire']=np.where(fire['summer_fire']>0,fire['summer_fire'],0)
fire['non-summer_fire']=np.where(fire['ln(area+1)']==fire['summer_fire'],0,fire['ln(area+1)'])

plt.figure(8)
plt.title('fire-month', fontsize=14, position=(0.5,1.05))
sns.barplot(x=fire['month'],y=fire['summer_fire'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'Red',  # set the color palatte
           )# the color change indicates the change of density
sns.barplot(x=fire['month'],y=fire['non-summer_fire'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'Blue',  # set the color palatte
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.xlabel('month',fontsize=14,position=(0.6,1.05))
plt.ylabel('fire:ln(area+1)',fontsize=14)
plt.show()

'''
create new variable/column: 
weekday_fire, which only displays the fire in weekdays
weekend_fire, which only displays the fire in weekends

'''

b=fire.loc[fire['day']<6]
fire['weekday_fire']=b['ln(area+1)']
fire['weekday_fire']=np.where(fire['weekday_fire']>0,fire['weekday_fire'],0)
fire['weekend_fire']=np.where(fire['ln(area+1)']==fire['weekday_fire'],0,fire['ln(area+1)'])

plt.figure(9)
plt.title('fire-day', fontsize=14, position=(0.5,1.05))
# weekday_fire barplots 
sns.barplot(x=fire['day'],y=fire['weekday_fire'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'Red',  # set the color palatte
           )# the color change indicates the change of density
# weekend_fire barplots 
sns.barplot(x=fire['day'],y=fire['weekend_fire'], # demonstrate the probability distribution of two variables
           data = fire,   # display shades
           color = 'Blue',  # set the color palatte
           )# the color change indicates the change of density
plt.grid(linestyle = '--')
plt.xlabel('day',fontsize=14,position=(0.6,1.05))
plt.ylabel('fire:ln(area+1)',fontsize=14)
plt.show()


c=fire.loc[fire['FIRE']=='fire']
plt.figure(10) #DC-temp in only fire cases
plt.title('DC-temp', fontsize=14, position=(0.5,1.05))
pal='Purples'
sns.kdeplot(c['temp'],c['DC'],cbar = True,shade = True,cmap = pal,shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
plt.scatter(c['temp'], c['DC'], s=5, alpha = 0.5, color = 'r', marker='+') #scatter(green:no fire; red:fire)
sns.rugplot(c['temp'], color="orange", axis='x',alpha = 0.5)
sns.rugplot(c['DC'], color="purple", axis='y',alpha = 0.5)
plt.show()

plt.figure(11) #DMC-temp/RH in only fire cases
plt.title('Fire Intensity')
plt.subplot(121)
plt.title('DMC-temp', fontsize=20, position=(0.5,1.05))
pal='Blues'
sns.kdeplot(c['temp'],c['DMC'],cbar = True,shade = True,cmap = pal,shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
#plt.scatter(c['temp'], c['DMC'], s=5, alpha = 0.5, color = 'r', marker='*') #scatter(green:no fire; red:fire)
sns.rugplot(c['temp'], color="red", axis='x',alpha = 0.5)
sns.rugplot(c['DMC'], color="red", axis='y',alpha = 0.5)
plt.subplot(122) #DC-temp in only fire cases
plt.title('DMC-RH', fontsize=20, position=(0.5,1.05))
pal='Blues'
sns.kdeplot(c['RH'],c['DMC'],cbar = True,shade = True,cmap = pal,shade_lowest=False,n_levels = 40)
plt.grid(linestyle = '--')
#plt.scatter(c['RH'], c['DMC'], s=5, alpha = 0.5, color = 'r', marker='*') #scatter(green:no fire; red:fire)
sns.rugplot(c['RH'], color="red", axis='x',alpha = 0.5)
sns.rugplot(c['DMC'], color="red", axis='y',alpha = 0.5)
plt.show()

plt.figure(12)
a1=fire[fire['rain']==0.0]
a1=sum(a1['rain'])
a2=fire[(1.0<=fire['rain'])&(fire['rain']<=2.0)]
a2=sum(a2['rain'])
a3=fire[(5.5<=fire['rain'])&(fire['rain']<=6.5)]
a3=sum(a3['rain'])
raindata=[a1,a2,a3]
labels=['0.0','(1.0,2.0)','(5.5,6.5)']
clr=['green','blue','yellow']
plt.axis('equal')
plt.pie(raindata, labels=labels, autopct='%1.1f%%',colors=clr)
plt.title('rain value distribution')
plt.show()






