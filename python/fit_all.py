# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:09:58 2021

@author: zincarrot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp
import csv
import sys

#%% set bounds
bounds_min = [0.6, -7, 0, 0.6, -1e-7, -6, -8, -3]
bounds_max = [1, -3, 4, 1, 1e-7, 2, -2, 4]
bounds = np.asarray([bounds_min, bounds_max]).T

T_list = [2, 14, 26, 82, 94, 106, 117, 130, 176]
freq_list= [300, 500, 1000, 2000, 4000, 5000, 7000, 10000, 15000, \
        20000, 30000, 40000, 50000, 80000, 100000]
freq_list = np.asarray(freq_list)

N = 10000

#%% read data
    
measure_data = pd.read_excel(r'9.15 Cell Growth/9-15 cell density.xlsx').to_numpy()
measure_data = measure_data[:,2].reshape(9,3).astype(np.float)

feat_ref=np.loadtxt('x_opt3.txt')
    
feat=dict()


#%% model

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

#%% fit



def fitall(sensor):
    
    for i in range(len(T_list)):
        refit(sensor,i)
        forwardfit(sensor,i)
        backwardfit(sensor,i)
        
def fitall_pool(sensor):
    import fit_all
    
    pool=mp.Pool(mp.cpu_count())
    pool.starmap(fit_all.refit,[(i,sensor) for i in range(len(fit_all.T_list))])
    
    for i in range(len(T_list)):
        pool.apply(fit_all.forwardfit,args=(sensor,i))
    
    for i in range(len(T_list)):
        pool.apply(fit_all.backwardfit,args=(sensor,i))
        
    pool.close()

def refit(sensor,i):
    feat=loadfeat()
    result=fitspectrum(sensor,T_list[i],feat_ref[i])
    feat[T_list[i]]=result
    savefeat(feat)

def forwardfit(sensor,i):
    feat=loadfeat()
    if(i!=len(T_list)-1):
        fit_list=range(T_list[i],T_list[i+1])
    else:
        fit_list=range(T_list[i],177)
    ref=feat[T_list[i]]
    for n in fit_list:
        try:
            if feat[n]['r2'] > 0.9995:
                ref=feat[n]
                print('skip!')
                sys.stdout.flush()
                continue
            else:
                print(ref)
                result=fitspectrum(sensor,n,ref['features'])
                feat[n]=result
                ref=result
        except:
            print(ref)
            result=fitspectrum(sensor,n,ref['features'])
            feat[n]=result
            ref=result
        finally:
            savefeat(feat)
        
def backwardfit(sensor,i):
    feat=loadfeat()
    if(i!=0):
        fit_list=range(T_list[i],T_list[i-1],-1)
    else:
        fit_list=range(T_list[i],0,-1)
    ref=feat[T_list[i]]
    for n in fit_list:
        try:
            if feat[n]['r2'] > 0.9998:
                ref=feat[n]
                print('skip!')
                continue
            else:
                print(ref)
                result=fitspectrum(sensor,n,ref['features'])
        except:
            print(ref)
            result=fitspectrum(sensor,n,ref['features'])
        
        try:
            if feat[n]['r2'] < result['r2']:
                feat[n]=result
                ref=result
            else:
                ref=feat[n]
        except:
            feat[n] = result
            ref=result
        finally:
            savefeat(feat)
        
def fitspectrum(sensor,n,ref):
    print("fitting spectrum #{}".format(n))
    sys.stdout.flush()
    file_name = '9.15 Cell Growth/20190915_Sensor'+str(sensor)+'_Test_'+str(n)+'.xlsx'
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data,[imp_data.shape[0],15,5])[:,:,2:4]
    # check nan?
    imp_data[imp_data==0]=np.nan
    imp_data[imp_data>1e30]=np.nan
    imp_mean = np.nanmean(imp_data,axis=0)
    imp_std = np.nanstd(imp_data,axis=0)
    imp_comp = imp_mean[:,0]+imp_mean[:,1]*(0+1j)
    
    x_init = ref
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
        MSE = error(para,refr) 
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
    opt = minimize(obj[0], x_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
    x_opt = opt.x
    obj_function = opt.fun
    print(x_opt)
    print(obj_function)
    print(r2_MSE(x_opt))

    plt.figure()
    plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
    plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).real,
                (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).imag)
    plt.show()

    # x_init = np.array(np.random.uniform(size=8))
    # x_init = x_init * ((np.array(bounds_max) - np.array(bounds_min)) * 0.3) + (
    #             x_opt - (x_opt - np.array(bounds_min)) * 0.3)
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
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                #     (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).real,
                #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).imag)
                # plt.show()
            
        scale=(obj_logzMSE_all(x_opt))*x_opt*np.array([10,10,5,10,10,5,5,5])*0.1
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #             (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
        #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
        # plt.show()
        if r2_logzMSE(x_opt) > 0.95:
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
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                #     (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).real,
                #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).imag)
                # plt.show()
            
        scale=(obj_logzMSE_all(x_opt))*x_opt*np.array([10,10,5,10,10,5,5,5])*0.05
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #             (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
        #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
        # plt.show()
        if r2_logzMSE(x_opt) > 0.998:
            print('break!')
            # bounds_min = [0, -10, 2.5, 0.1, -1e-3, -8, x_opt[6], -4]
            # bounds_max = [1, 0,4.5, 1, 1e-3, 2, -2, 4]
            # bounds = np.asarray([bounds_min, bounds_max]).T
            break
    
    # obj=[obj_logzMSE_all,obj_zMSE_all,obj_cMSE_all,obj_yMSE_all,obj_mMSE_all]
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
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
                #     (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
                # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).real,
                #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).imag)
                # plt.show()
            
        scale=(obj_logzMSE_all(x_opt))*x_opt*np.array([10,10,5,10,10,5,5,5])*0.1
        x_init=x_opt+np.random.random(size=len(x_opt))*scale-0.5*scale
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real,
        #             (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
        # plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).real,
        #     (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_init)))).imag)
        # plt.show()
        if r2_logzMSE(x_opt) > 0.9995:
            print('break!')
            break
        
    print(x_opt)
    plt.figure()
    plt.scatter((1 / (1j * 2 * np.pi * freq_list * imp_comp)).real, (1 / (1j * 2 * np.pi * freq_list * imp_comp)).imag)
    plt.scatter((1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).real,
                (1 / (1j * 2 * np.pi * freq_list * (imp_relax(freq_list, x_opt)))).imag)
    # plt.savefig('sensor{}_fit{}_2.jpg'.format(sensor, tt))
    
    result={'features':x_opt,'r2':r2_MSE(x_opt)}
    return result

#%% save

def restorefeat(fi):
    fe=fi.tolist()
    fe[1]=10**fe[1]
    fe[2]=10**fe[2]
    fe[5]=10**fe[5]
    fe[6]=10**fe[6]
    fe[7]=10**fe[7]
    return fe

def savefeat(feat):
    with open('feat.csv','w',newline='') as csvfile:
        featwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        featwriter.writerow(['ord','n','Q','fc','a','ch','kl','rs','r','r2'])
        for item in feat:
            rfeat=restorefeat(feat[item]['features'])
            featwriter.writerow([item]+rfeat+[feat[item]['r2']])

def loadfeat():
    with open('feat.csv',newline='') as csvfile:
        featreader=csv.DictReader(csvfile,delimiter=',', quotechar='|')
        for row in featreader:
            feat[row['ord']]={'features':np.array([row['n'],row['Q'],row['fc'],row['a'],row['ch'],row['kl'],row['rs'],row['r']]),'r2':row['r2']}
    return feat
#%% unit test
   
