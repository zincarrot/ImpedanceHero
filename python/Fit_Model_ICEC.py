#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:22:54 2020

@author: jialei, zincarrot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sympy import *

# %%

np.random.seed(0)

# T_list = [2, 14, 26, 82, 94, 106, 117, 130, 176]
T_list=[1]
sensor_list=[0,1,2,3]

#freq_list = [100,200,300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 80000, 100000]
freq_list = [300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 80000, 100000]
    
freq_list = np.asarray(freq_list)[::-1]
freq_list2 = np.linspace(100, 1000000, 100)

def imp_relax(f, para):
    n = para[0]
    Q = 10 ** para[1]
    R=10**para[2]
    A = 10 ** para[3]
    B= 10**para[4]
    C = 10**para[5]
    # cw2 = 10**para[6]
    
    rf=2*np.pi*f
    zw=A*(1j*rf*B+C)**0.5/(1j*rf*B)

    zcpe=1/(Q*(1j*rf)**n)
    ztot=zcpe+R+zw
    return ztot

bounds_min = [0, -8, 0, -5, -5, 0]
bounds_max = [1, -2, 4, 6, 5, 6]
bounds = np.asarray([bounds_min, bounds_max]).T

hi_freq_para = np.array([0.5, 0.5, 1, 1, 1, 1, 1])
lo_freq_para = np.array([1, 1, 0.7, 0.7, 0.5, 1, 0.3])

N = 1000
x_ini_mat = np.array(np.random.uniform(size=N * len(bounds_max))).reshape([N, len(bounds_max)])
x_ini_mat = x_ini_mat * (np.array(bounds_max) - np.array(bounds_min)) + bounds_min

# %%  find features
feat = []
c_error = []
z_error = []
r2 = []

# plt.scatter(ecc(freq_list,x_init).real,
#             ecc(freq_list,x_init).imag)
# plt.scatter(ep(freq_list,x_init).real,
#             ep(freq_list,x_init).imag)
# plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
#     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
# plt.show()
x_init=np.array([0.6, -5,  1, 4,   0, 3])
#%% iterate
for it in range(len(sensor_list)):
    tt = T_list[0]
    print(str(it) + ' >>>---- ' + str(tt))
    # for it in range(1):
    #    tt = 130
    file_name = '20210607_dual_2percent//oldR/20210607_Sensor{}_Test_{}.xlsx'.format(it, tt)
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data, [imp_data.shape[0], len(freq_list), 5])[:, :, 2:4]
    imp_data[imp_data == 0] = np.nan
    imp_data[imp_data>1e30] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.nanstd(imp_data, axis=0)
    imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)

    x_ini = (np.asarray(bounds_min) + np.asarray(bounds_max)) / 2

    def c(f, z):
        return 1 / (1j * 2 * np.pi * f * z)

    def m(f, z):
        return 1j * 2 * np.pi * f * z

    def y(f,z):
        return 1/z

    def z(f,z):
        return z
    
    def logz(f,z):
        return np.log(z)

    def r2_MSE(para,refr=c):
        return 1 - np.sum(error(para,refr)**2)

    def obj_MSE(para,refr=c):
        MSE = error(para,refr) **2
        return np.mean(MSE)
    
    def obj_MaxEr(para,refr=c):
        MSE = error(para,refr) 
        return np.max(MSE)

    def obj_MSE_rollAngle(para,refr=c):
        MSE = error(para,refr) 
        rollMSE=rollAngleError(para,refr)
        return np.mean(MSE)+np.mean(rollMSE)

    def obj_MSE_rollMag(para,refr=c):
        MSE = error(para,refr) 
        rollMSE=rollMagError(para,refr)
        return np.mean(MSE)+np.mean(rollMSE)
    
    def obj_MSE_all(para,refr=c):
        MSE = error(para,refr) 
        magMSE=rollMagError(para,refr)
        angleMSE=rollAngleError(para,refr)
        return np.mean(MSE)+np.mean(magMSE)+np.mean(angleMSE)
        
    def obj_MaxEr_all(para,refr=c):
        MSE = error(para,refr) 
        magMSE=rollMagError(para,refr)
        angleMSE=rollAngleError(para,refr)
        return np.max(MSE)+np.max(magMSE)+np.max(angleMSE)
    
    def obj_logzMaxEr(para):
        return obj_MaxEr(para,logz)
    def obj_logzMaxEr_all(para):
        return obj_MaxEr_all(para,logz)
    
    def obj_logzMSE_all(para):
        return obj_MSE_all(para,logz)
    def obj_cMSE_all(para):
        return obj_MSE_all(para,c)
    def obj_zMSE_all(para):
        return obj_MSE_all(para,z)
    def obj_mMSE_all(para):
        return obj_MSE_all(para,m)
    def obj_yMSE_all(para):
        return obj_MSE_all(para,y)
    def obj_logzMSE(para):
        return obj_MSE(para,logz)
    def obj_cMSE(para):
        return obj_MSE(para,c)
    def obj_yMSE(para):
        return obj_MSE(para,y)
    def obj_mMSE(para):
        return obj_MSE(para,m)
    def obj_zMSE(para):
        return obj_MSE(para,z)
    def obj_logzMSE_rollMag(para):
        return obj_MSE_rollMag(para,logz)
    def obj_cMSE_rollMag(para):
        return obj_MSE_rollMag(para,c)
    def obj_yMSE_rollMag(para):
        return obj_MSE_rollMag(para,y)
    def obj_mMSE_rollMag(para):
        return obj_MSE_rollMag(para,m)
    def obj_zMSE_rollMag(para):
        return obj_MSE_rollMag(para,z)
    def obj_logzMSE_rollAngle(para):
        return obj_MSE_rollAngle(para,logz)
    def obj_cMSE_rollAngle(para):
        return obj_MSE_rollAngle(para,c)
    def obj_yMSE_rollAngle(para):
        return obj_MSE_rollAngle(para,y)
    def obj_mMSE_rollAngle(para):
        return obj_MSE_rollAngle(para,m)
    def obj_zMSE_rollAngle(para):
        return obj_MSE_rollAngle(para,z)
    def r2_logzMSE(para):
        return r2_MSE(para,logz)
    def r2_cMSE(para):
        return r2_MSE(para,c)
    def r2_yMSE(para):
        return r2_MSE(para,y)
    def r2_mMSE(para):
        return r2_MSE(para,m)
    def r2_zMSE(para):
        return r2_MSE(para,z)

    def error(para,refr=c):
        if refr==logz:
            error=np.abs(refr(freq_list, imp_comp) - refr(freq_list, imp_relax(freq_list, para)))
        else:
            error = np.abs(refr(freq_list, imp_comp) - refr(freq_list, imp_relax(freq_list, para))) / np.abs(
            refr(freq_list, imp_comp))
        return np.abs(error)

    def roll1(para):
        return para[:-1]-np.roll(para,-1)[:-1]

    def rollAngleError(para,refr=c):
        if refr==logz:
            error=np.imag(roll1(refr(freq_list, imp_comp)) - roll1(refr(freq_list, imp_relax(freq_list, para))))
        else:
            error = (np.angle(roll1(refr(freq_list, imp_comp))) - np.angle(roll1(refr(freq_list, imp_relax(freq_list, para)))))/np.angle(roll1(refr(freq_list, imp_comp)))
        return np.abs(error)

    def rollMagError(para,refr=c):
        if refr==logz:
            error=np.real(roll1(refr(freq_list, imp_comp)) - roll1(refr(freq_list, imp_relax(freq_list, para))))
        else:
            error = (np.abs(roll1(refr(freq_list, imp_comp))) - np.abs(
            roll1(refr(freq_list, imp_relax(freq_list, para))))) / np.abs(roll1(refr(freq_list, imp_comp)))
        return np.abs(error)

    # obj = [obj_zMSE, obj_yMSE, obj_mMSE, obj_cMSE]
    # obj=[obj_logzMSE_all,obj_logzMSE,obj_logzMSE_rollAngle,obj_logzMSE_rollMag,obj_cMSE,obj_cMSE_rollAngle, obj_cMSE_rollMag,obj_yMSE,obj_yMSE_rollAngle, obj_yMSE_rollMag,obj_mMSE,obj_mMSE_rollAngle, obj_mMSE_rollMag,obj_zMSE,obj_zMSE_rollAngle, obj_zMSE_rollMag]
    # obj=[obj_logzMSE_all,obj_logzMSE,obj_logzMSE_rollAngle,obj_logzMSE_rollMag]
    # obj=[obj_logzMSE_all,obj_zMSE_all,obj_cMSE_all,obj_yMSE_all,obj_mMSE_all]
    obj=[obj_logzMSE_all]
    opt = minimize(obj[0], x_ini_mat[0], method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
    x_opt = opt.x
    obj_function = opt.fun
    print(x_opt)
    print(obj_function)
    print(r2_MSE(x_opt))

    plt.figure()
    plt.scatter( imp_comp.real,
                    imp_comp.imag)
    plt.scatter(imp_relax(freq_list, x_opt).real,
        imp_relax(freq_list, x_opt).imag)
    plt.show()

    x_init = np.array(np.random.uniform(size=len(bounds_max)))
    x_init = x_init * ((np.array(bounds_max) - np.array(bounds_min)) * 0.3) + (
                x_opt - (x_opt - np.array(bounds_min)) * 0.3)
    # x_init=np.array([7.69375701e-01, -5.19202670e+00,  3.23692532e+00,  9.99999753e-01,  -4.19501982e-09, -2.00273208e+00, -5.55093071e+00,  2.57459058e+00])
    
    for i in range(N):
        x_old=x_opt
        obj_function_old = obj_function
        r2_old = r2_logzMSE(x_opt)
        for j in obj:
            opt = minimize(j, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500,'disp':False})
            if r2_logzMSE(x_init) < r2_logzMSE(opt.x):
                x_init=opt.x
            if r2_logzMSE(x_opt) < r2_logzMSE(opt.x):
                print(str(i) + str(j))
                x_opt = opt.x
                obj_function = opt.fun
                print(x_opt)
                print(r2_logzMSE(x_opt))
                print(obj_function)
                plt.scatter( imp_comp.real,
                    imp_comp.imag)
                plt.scatter(imp_relax(freq_list, x_opt).real,
                    imp_relax(freq_list, x_opt).imag)
                plt.show()
            
        scale=(obj_logzMSE_all(x_opt)**0.5)*x_opt*np.array([1,1,1,1,1,1])*10
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #             (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
        #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
        # plt.show()
        if r2_logzMSE(x_opt) > 0.99:
            print('break!')
            # bounds_min = [0, -10, 2.5, 0.1, -1e-3, -8, x_opt[6], -4]
            # bounds_max = [1, 0,4.5, 1, 1e-3, 2, -2, 4]
            # bounds = np.asarray([bounds_min, bounds_max]).T
            break
    
    for i in range(N):
        x_old=x_opt

        obj_function_old = obj_function
        r2_old = r2_logzMSE(x_opt)
        for j in obj:
            opt = minimize(j, x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500,'disp':False})
            if r2_logzMSE(x_init) < r2_logzMSE(opt.x):
                x_init=opt.x
            if r2_logzMSE(x_opt) < r2_logzMSE(opt.x):
                print(str(i) + str(j))
                x_opt = opt.x
                obj_function = opt.fun
                print(x_opt)
                print(r2_logzMSE(x_opt))
                print(obj_function)
                plt.scatter( imp_comp.real,
                    imp_comp.imag)
                plt.scatter(imp_relax(freq_list, x_opt).real,
                    imp_relax(freq_list, x_opt).imag)
                plt.show()
            
        scale=(obj_logzMSE_all(x_opt))*x_opt*np.array([1,1,1,1,1,1])*5
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #             (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
        #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
        # plt.show()
        if r2_logzMSE(x_opt) > 0.999:
            print('break!')
            break
        
    print(x_opt)
    plt.figure()
    plt.scatter( imp_comp.real,
                    imp_comp.imag)
    plt.plot(imp_relax(freq_list2, x_opt).real,
        imp_relax(freq_list2, x_opt).imag)
    plt.savefig('ICEC_saline_s{}_t{}.jpg'.format(it, tt))

    feat.append(x_opt)

    c_error.append(obj_MSE(x_opt))
    z_error.append(obj_MSE(x_opt))
    r2.append(r2_MSE(x_opt))

feat = np.asarray(feat)
feat_norm = (feat - bounds_min) / (np.array(bounds_max) - np.array(bounds_min))

print(r2)
print(np.sqrt(r2))

# measure_data_norm = measure_data / 1e6

# xlist=range(9)
# plt.figure()
# plt.plot(xlist,feat_norm[:,1])
# plt.plot(xlist,feat_norm[:,6])
# plt.plot(xlist,measure_data_norm[:,0])
# plt.show()
#
# plt.figure()
# plt.plot(feat_norm[:,1],feat_norm[:,6])
# plt.show()





# %% using features for prediction
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.model_selection import KFold, cross_val_predict
#
# np.random.seed(0)
#
# feat_norm = (feat - bounds_min) / (np.array(bounds_max) - np.array(bounds_min))
#
#
# measure_var = np.var(measure_data_norm, axis=1)
# yy = np.mean(measure_data_norm, axis=1)[:8]
# xx = feat_norm[:8, ]
#
# yy = np.hstack((measure_data_norm[:8, 0], measure_data_norm[:8, 1], measure_data_norm[:8, 2]))
# xx = np.vstack((xx, xx, xx))
#
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# # C(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)* RBF(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)
#
# gp = GaussianProcessRegressor(kernel=kernel, alpha=np.mean(measure_var),
#                               n_restarts_optimizer=10)
#
# gp.fit(xx, yy)
# y_pred, sigma = gp.predict(xx, return_std=True)
#
# cv_pred = cross_val_predict(gp, xx, y=yy, cv=KFold(yy.size), n_jobs=1)
#
# print(np.mean(np.abs(cv_pred - yy) / np.abs(yy)))
#
# plt.plot(yy, cv_pred, 'o')
# plt.plot([0.2, 1], [0.2, 1])
# plt.xlabel('truth')
# plt.ylabel('predicted')
# plt.show()
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.model_selection import KFold, cross_val_predict

# np.random.seed(0)

# feat_norm = (feat -bounds_min) /(np.array(bounds_max)-np.array(bounds_min))
# measure_data_norm = measure_data/1e6

# measure_var = np.var(measure_data_norm,axis=1)
# yy = np.mean(measure_data_norm,axis=1)[:]
# xx = feat_norm[:,]

# yy = np.hstack((measure_data_norm[:,0], measure_data_norm[:,1], measure_data_norm[:,2]))
# xx = np.vstack((xx,xx,xx))

# kernel =C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# #C(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)* RBF(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)

# gp = GaussianProcessRegressor(kernel=kernel, alpha=np.mean(measure_var),
#                               n_restarts_optimizer=10)

# gp.fit(xx, yy)
# y_pred, sigma = gp.predict(xx, return_std=True)

# cv_pred = cross_val_predict(gp, xx, y=yy, cv=KFold(yy.size), n_jobs=1)

# print(np.mean(np.abs(cv_pred-yy)/np.abs(yy)))

# plt.plot(yy,cv_pred,'o')
# plt.plot([0.2,1],[0.2,1])
# plt.xlabel('truth')
# plt.ylabel('predicted')
# plt.show()