#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:22:54 2020

@author: jialei
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

#%%


measure_data = pd.read_excel(r'9.15 Cell Growth/9-15 cell density.xlsx').to_numpy()
measure_data = measure_data[:,2].reshape(9,3)
y_data = np.log10(np.mean(measure_data,axis=1).tolist())
y_data = (y_data-np.min(y_data))/(np.max(y_data)-np.min(y_data))

x_data = np.genfromtxt("x_opt.txt")

data_min = np.min(x_data,axis=0)
data_max = np.max(x_data,axis=0)

x_data = (x_data-data_min)/(data_max-data_min)

n = x_data.shape[0]


for i in range(x_data.shape[1]):
    plt.scatter(x_data[:,i],y_data) 
    plt.xlabel('Feature '+str(i+1))    
    plt.ylabel('Output')    
    plt.show()

#%%


from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict



list_fea=np.asarray([1,2,3,8])-1
x_data = x_data[:,list_fea]


regr = LinearRegression()
# regr = Ridge(alpha = 0.001)

regr.fit(x_data, y_data)

y_hat = regr.predict(x_data)
print(regr.coef_)
print('Training R2 is '+str(1-np.var(y_hat-y_data)/np.var(y_data)))

cv_pred = cross_val_predict(regr, x_data, y=y_data, cv=KFold(y_data.size), n_jobs=1)


print('Testing R2 is '+str(1-np.var(cv_pred-y_data)/np.var(y_data)))

print('Testing R2 (remove the last data) is '+str(1-np.var(cv_pred[:8]-y_data[:8])/np.var(y_data[:8])))

print('RMSE is '+str(np.sqrt(np.mean(cv_pred-y_data)**2)))



plt.plot(y_data,cv_pred,'o')
plt.plot([0.2,1],[0.2,1])
plt.xlabel('truth')
plt.ylabel('predicted')
plt.show()







