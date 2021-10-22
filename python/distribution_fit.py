# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:46:37 2020

@author: zincarrot
"""

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# %%

np.random.seed(0)

measure_data = pd.read_excel(r'9.15 Cell Growth/9-15 cell density.xlsx').to_numpy()
measure_data = measure_data[:, 2].reshape(9, 3)
T_list = [2, 14, 26, 82, 94, 106, 117, 130, 176]

freq_list = [300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, \
             20000, 30000, 40000, 50000, 80000, 100000]
freq_list = np.asarray(freq_list)
freq_list_fc = np.linspace(1000, 10000, 100)

#relaxation strengths

#fitting parameters

# construct transformation matrix rs -> ecc
trans_freq=1/(1+(1j*np.outer(freq_list,1/freq_list_fc)))

# construct transformation matrix fit -> rs
distr=lambda mean,scale,mul: norm.pdf(freq_list_fc,loc=mean,scale=scale)*mul

#relaxation process
ecc = lambda rs,ch :np.dot(trans_freq, rs)+ch

zcc = lambda rs,ch: 1/(1j*2*np.pi*freq_list*ecc(rs,ch))
zcpe= lambda Q,n: 1 / (Q * (1j * 2 * np.pi * freq_list))
ztot = lambda rs,Q,n,r,ch: zcc(rs,ch)+zcpe(Q,n)+r#
logz = lambda rs,Q,n,r,ch: np.log(ztot(rs,Q,n,r,ch))
z2c=lambda z: 1/(1j*2*np.pi*freq_list*z)
model= lambda p: logz(distr(p[0],p[1],p[2]),p[3],p[4],p[5],p[6]) 

c_model= lambda p: z2c(ztot(distr(p[0],p[1],p[2]),p[3],p[4],p[5],p[6]))
ecc_model=lambda p: np.dot(trans_freq,distr(p[0],p[1],p[2]) )+p[6]

# %% optimize
for it in range(len(T_list)):
    tt = T_list[it]
    print(str(it) + ' >>>---- ' + str(tt))
    # for it in range(1):
    #    tt = 130
    sensor = 0
    file_name = '9.15 Cell Growth/20190915_Sensor{}_Test_{}.xlsx'.format(sensor, tt)
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data, [imp_data.shape[0], 15, 5])[:, :, 2:4]
    imp_data[imp_data == 0] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.nanstd(imp_data, axis=0)
    imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)
    def c(f, z):
        return 1 / (1j * 2 * np.pi * f * z)

    def m(f, z):
        return 1j * 2 * np.pi * f * z

    def y(f,z):
        return 1/z

    def z(f,z):
        return z

    def r2_MSE(para):
        return 1 - np.sum(error(para)**2)

    def obj_MSE(para):
        MSE = error(para) 
        return np.mean(MSE)

    def error(para):
        error=np.abs(np.log(imp_comp) - model(para))
        return np.abs(error)

    def roll1(para):
        return para[:-1]-np.roll(para,-1)[:-1]

    def rollAngleError(para):
        error=np.imag(roll1(np.log(imp_comp))) - roll1(model(para))
        return np.abs(error)

    def rollMagError(para):
        error=np.real(roll1(np.log(imp_comp))) - roll1(model(para))
        return np.abs(error)
    
    x_init=[3000,500,1,1e-5,0.8,25,1e-5]
    # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (c_model(x_init)))).real,
    #     (1 / (1j * 2 * np.pi * freq_list * (c_model(x_init)))).imag)
    # plt.show()
    # plt.plot(freq_list_fc,distr(6000,500,1e-3))
    # plt.show()
    
    
    
    
    opt = minimize(obj_MSE, x_init, method='L-BFGS-B', options={'maxiter': 500})
    x_opt = opt.x
    obj_function = opt.fun
    print(x_opt)
    print(obj_function)
    print(r2_MSE(x_opt))
    plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                  (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
    plt.scatter(c_model(x_opt).real, c_model(x_opt).imag)
    plt.show()
    x_init=x_opt
    
    for i in range(10000):
        x_old=x_opt
        obj_function_old = obj_function
        r2_old = r2_MSE(x_opt)
        opt = minimize(obj_MSE, x_init, method='L-BFGS-B', options={'maxiter': 500,'disp':False})
        if r2_MSE(x_init) < r2_MSE(opt.x):
            x_init=opt.x
        if r2_MSE(x_opt) < r2_MSE(opt.x):
            print(str(i))
            x_opt = opt.x
            obj_function = opt.fun
            print(x_opt)
            print(r2_MSE(x_opt))
            print(obj_function)
            plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                  (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
            plt.scatter(c_model(x_opt).real, c_model(x_opt).imag)
            plt.show()
        scale=(obj_MSE(x_opt))*x_opt*np.array([1,1,1,1,1,1,1])
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #           (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter(c_model(x_opt).real, c_model(x_opt).imag)
        # plt.show()
        if r2_MSE(x_opt) > 0.9999:
            print('break!')
            break
    # rs=np.linalg.inv(trans_freq)@imp_comp
    # plt.plot(freq_list_fc_ind,rs.real)
    # plt.plot(freq_list_fc_ind,rs.imag)
    # plt.show()
    
    
    