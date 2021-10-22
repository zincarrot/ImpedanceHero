 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:22:54 2020

@author: jialei, zeke
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

#%%

np.random.seed(0)

measure_data = pd.read_excel(r'9.15 Cell Growth/9-15 cell density.xlsx').to_numpy()
measure_data = measure_data[:,2].reshape(9,3)
T_list=[2,14,26,82,94,106,118,130,176]


freq_list= [300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, \
            20000, 30000, 40000, 50000, 80000, 100000]
freq_list = np.asarray(freq_list)


def imp_relax(f,para):
    n = para[0]
    Q = 10**para[1]
    fc = 10**para[2]
    a = para[3]
    ch = 10**para[4]
    ce = 10**para[5]
    cl = 10**para[6]
    jj = 0+1j
    val = 2*np.pi*jj*f*((jj*f/fc)**a*(ch-ce*jj)+cl-ce*jj)
    val = (1+(jj*f/fc)**a)/(val+1e-12)
    val = val+1/((jj*Q)**n+1e-12)
    return val

bounds_min =[1e-6,-15,2,1e-6,-15,-15,-15]
bounds_max =[1,   -3, 10,1,   -3, -3, -3]
bounds = np.asarray([bounds_min,bounds_max]).T

N=100
x_ini_mat = np.array(np.random.uniform(size=N*7)).reshape([N,7])
x_ini_mat = x_ini_mat *(np.array(bounds_max)-np.array(bounds_min))+bounds_min
    


#%%  find features
feat = []

for it in range(len(T_list)):
    tt=T_list[it]
    print(str(it) + ' >>>---- ' +str(tt))
#for it in range(1):
#    tt = 130
    file_name = '9.15 Cell Growth/20190915_Sensor1_Test_'+str(tt)+'.xlsx'
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data,[imp_data.shape[0],15,5])[:,:,2:4]
    imp_data[imp_data==0]=np.nan
    imp_mean = np.nanmean(imp_data,axis=0)
    imp_std = np.nanstd(imp_data,axis=0)
    imp_comp = imp_mean[:,0]+imp_mean[:,1]*(0+1j)
    
    
    x_ini = (np.asarray(bounds_min)+np.asarray(bounds_max))/2
    
    def obj_MSE(para):
#        MSE = np.mean((np.abs(imp_comp - imp_relax(freq_list, para)))**2)
#        return np.log(np.sqrt(MSE))
        MSE = np.abs(imp_comp - imp_relax(freq_list, para))/np.abs(imp_comp)
        return np.log(np.mean(MSE**2))
    
    
    opt = minimize(obj_MSE,x_ini,method='L-BFGS-B',bounds =bounds,options={'maxiter':200})
    x_opt=opt.x
    obj_function = opt.fun
    
    
    for i in range(N):
        opt = minimize(obj_MSE,x_ini_mat[i,:],method='L-BFGS-B',bounds =bounds,options={'maxiter':200})
        if opt.fun<obj_function:
#            print(i)
            x_opt=opt.x
            obj_function = opt.fun
    
#    print(imp_comp)
#    print(imp_relax(freq_list, x_opt))
#    
    plt.scatter(imp_mean[:,0],imp_mean[:,1])
    plt.show()

    feat.append(x_opt)
    

feat = np.asarray(feat)
feat_norm = (feat -bounds_min) /(np.array(bounds_max)-np.array(bounds_min))


#%% using features for prediction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import KFold, cross_val_predict

np.random.seed(0)

feat_norm = (feat -bounds_min) /(np.array(bounds_max)-np.array(bounds_min))
measure_data_norm = measure_data/1e6

measure_var = np.var(measure_data_norm,axis=1)
yy = np.mean(measure_data_norm,axis=1)[:8]
xx = feat_norm[:8,]

yy = np.hstack((measure_data_norm[:8,0], measure_data_norm[:8,1], measure_data_norm[:8,2]))
xx = np.vstack((xx,xx,xx))

kernel =C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#C(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)* RBF(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)

gp = GaussianProcessRegressor(kernel=kernel, alpha=np.mean(measure_var),
                              n_restarts_optimizer=10)

gp.fit(xx, yy)
y_pred, sigma = gp.predict(xx, return_std=True)

cv_pred = cross_val_predict(gp, xx, y=yy, cv=KFold(yy.size), n_jobs=1)

print(np.mean(np.abs(cv_pred-yy)/np.abs(yy)))

plt.plot(yy,cv_pred,'o')
plt.plot([0.2,1],[0.2,1])
plt.xlabel('truth')
plt.ylabel('predicted')
plt.show()
