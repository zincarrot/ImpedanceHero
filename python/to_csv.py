# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:21:29 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


T_list = [2, 14, 26, 82, 94, 106, 117, 130, 176]
freq_list = [300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, \
             20000, 30000, 40000, 50000, 80000, 100000]
freq_list = np.asarray(freq_list)
    
def loadfit(filename):
    return np.loadtxt(filename)

def imp_relax(f, para):
    n = para[0]
    Q = 10 ** para[1]

    fc = 10 ** para[2]

    a = para[3]
    ch = para[4]
    kl = 10 ** para[5]
    rs = 10 ** para[6]
    r = 10 ** para[7]

    jj = 0 + 1j
    ecc =  ch +kl / (jj * 2 * np.pi * f) + (rs) / (1 + (jj * f / fc) ** a)
    # ecc = (rs) / (1 + (jj * f / fc) ** a) + ch
    zcc = 1 / (jj * 2 * np.pi * f * ecc)
    
    # ztot=zcc+1/(Q*(jj*2*np.pi*f)**n)
    zcpe = 1 / (Q * (jj * 2 * np.pi * f)**n) #+1 / (kl * (jj * 2 * np.pi * f))
    # zcpe = 1 / Q 
    ztot = zcc + zcpe*r/(zcpe+r)
    # val = 2*np.pi*jj*f*((jj*f/fc)**a*(ch-ce*jj)+cl-ce*jj)
    # val = (1+(jj*f/fc)**a)/(val)
    # val = val+1/(Q*(jj*2*np.pi*f)**n)
    return ztot



#%% export fit results

feat=loadfit('x_opt3.txt')

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
    imp_data[imp_data>1e30] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.nanstd(imp_data, axis=0)
    imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)
    
    datareal=(1 / (1j * 2 * np.pi * freq_list * imp_comp)).real
    dataimag=(1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag
    fitreal=(1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, feat[it])))).real
    fitimag=(1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, feat[it])))).imag
    result=np.array([freq_list,datareal,dataimag,fitreal,fitimag]).T
    np.savetxt("Plots/data/fit_s{}_t{}.csv".format(sensor,tt),result,delimiter=',')
    