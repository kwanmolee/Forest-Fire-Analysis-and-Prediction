import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression,Lasso,LassoCV,LassoLarsCV,RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict

fire = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')
fire.month=fire.month.map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
fire.day=fire.day.map({'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7})
fire['ln(area+1)']=np.log(fire['area']+1)
df1=pd.DataFrame(fire,columns=['DMC','temp','RH','rain','ln(area+1)'])

X=df1[['temp','RH','rain']]
X1=df1[['temp','RH']]
X2=df1[['temp','rain']]
X3=df1[['RH','rain']]
Y=df1['DMC']

'''
generate a function for inputting regression models and the independent variables needed to be fitted
:parm X: series of independent variables needed to be fitted
:parm model:regression models
:parm n: the index of plot output

For each plot of the regression method,there are two subplots
a. estimate the fitting effect/result by comparing the distribution of scatters from true values and predicted values
and also a score of fitting effect will be given. The higher, the better.
b. estimate the predicted value and true value (based on the cross validation)
when the scatters are distributed to the diagonal line more closely, the fitting effect/result is better

'''
def regression_method(X,model,n):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    plt.figure(n)
    plt.subplot(211)
    plt.plot(np.arange(len(y_pred)), y_test,'ro-',label='true value')
    plt.plot(np.arange(len(y_pred)),y_pred,'bx-',label='predicted value')
    title=['Linear Regression','Decision Tree','LassoCV','LassoLarsCV',
    'SVM Regression','KNeighborsRegressor','Random Forest Tree',
    'RidgeCV','AdaBoostRegressor','GradientBoostingRegressor']
    plt.title(str(title[n-1])+': ''score: %f'%score)
    plt.legend()
    plt.subplot(212)
    predicted = cross_val_predict(model, X, Y, cv=10)
    plt.scatter(Y, predicted)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.show()
    print (str(title[n-1])+'MSE: ',metrics.mean_squared_error(y_test, y_pred))
    print (str(title[n-1])+'RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #print (model.intercept_)
    #print (model.coef_)

'''
:initiate the regression methods used later

'''
lin_reg=LinearRegression()
tree_reg = DecisionTreeRegressor()
las_CV=LassoCV()
laslar_CV=LassoLarsCV()
svr=svm.SVR()
knn = neighbors.KNeighborsRegressor()
rf =ensemble.RandomForestRegressor(n_estimators=500)
rgCV=RidgeCV()
ada = ensemble.AdaBoostRegressor(n_estimators=500)
gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)

'''
: test all independent variables(temp,rain,RH) and the dependent variable(DMC)
: omit one independent variables from (temp,rain,RH) each time and see the corresponding fitting effect of DMC and rest indep variables
'''
for i in [X,X1,X2,X3]:
    m=1
    regression_method(i,lin_reg,1*m)
    regression_method(i,tree_reg,2*m)
    regression_method(i,las_CV,3*m)
    regression_method(i,laslar_CV,4*m)
    regression_method(i,svr,5*m)
    regression_method(i,knn,6*m)
    regression_method(i,rf,7*m)
    regression_method(i,rgCV,8*m)
    regression_method(i,ada,9*m)
    regression_method(i,gbrt,10*m) 
    m+=1


