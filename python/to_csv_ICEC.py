# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:21:29 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


T_list=[1]
sensor_list=[0,1,2,3]

freq_list = [100,200,300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, \
             20000, 30000, 40000, 50000, 80000, 100000]
freq_list = np.asarray(freq_list)[::-1]
freq_list2_ind=np.linspace(1.9, 5.1, 100)
freq_list2 = 10**freq_list2_ind
    
def loadfit(filename):
    return np.loadtxt(filename)

def imp_relax(f, para):
    n = para[0]
    Q = 10 ** para[1]
    re=10**para[2]

    rm = 10 ** para[3]
    rw = 10**para[4]

    cw = 10**para[5]
    # cw2 = 10**para[6]
    
    rf=2*np.pi*f
    zcw=1/(1j*rf*cw)
    # zcw2=1/(1j*rf*cw2)
    zw=rw*zcw/(rw+zcw)
    # zw2=rm2+zcw2
    zcpe=1/(Q*(1j*rf)**n)
    ztot=zcpe+rm+zw
    return ztot


#%% export fit results

feat=loadfit('x_opt_ICEC.txt')

for it in range(len(sensor_list)):
    tt = T_list[0]
    print(str(it) + ' >>>---- ' + str(tt))
    # for it in range(1):
    #    tt = 130
    file_name = '20200818_saline/20200818_Sensor{}_Test_{}.xlsx'.format(it, tt)
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data, [imp_data.shape[0], len(freq_list), 5])[:, :, 2:4]
    imp_data[imp_data == 0] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.nanstd(imp_data, axis=0)
    imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)
    
    datareal=(1 / (1j * 2 * np.pi * freq_list * imp_comp)).real
    dataimag=(1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag
    fitreal=(1 / (1j * 2 * np.pi * freq_list2 * (imp_relax(freq_list2, feat[it])))).real
    fitimag=(1 / (1j * 2 * np.pi * freq_list2 * (imp_relax(freq_list2, feat[it])))).imag
    resultdata=np.array([freq_list,datareal,dataimag]).T
    resultfit=np.array([freq_list2,fitreal,fitimag]).T
    np.savetxt("fitICEC_s{}_t{}_data.csv".format(it,tt),resultdata,delimiter=',')
    np.savetxt("fitICEC_s{}_t{}_fit.csv".format(it,tt),resultfit,delimiter=',')
    