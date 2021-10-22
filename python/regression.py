# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:17:23 2020

@author: jialei, zeke
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE


def loadfit(filename):
    return np.loadtxt(filename)

def loadxls(filename):
    return pd.read_excel(filename).to_numpy()
    
def realfeat(paras):
    realfeat=[]
    for para in paras:
        n = para[0]
        Q = 10 ** para[1]
        fc = 10 ** para[2]
        a = para[3]
        ch = para[4]
        kl = 10 ** para[5]
        rs = 10 ** para[6]
        # r = 10 ** para[7]
        realfeat.append([n,Q,fc,a,ch,kl,rs])
    return np.array(realfeat)

def feat_guess1(paras):
    feat=[]
    for para in paras:
        n = para[0]
        Q = 10 ** para[1]
        fc = 10 ** para[2]
        a = para[3]
        ch = para[4]
        kl = 10 ** para[5]
        rs = 10 ** para[6]
        lgrs=para[6]
        lgfc=para[2]
        lgQ=para[1]
        # lgr=para[7]
        lgkl=para[5]
        # r = 10 ** para[7]
        feat.append([lgrs,lgfc,lgQ])
    return np.array(feat)

def feat_guess2(paras):
    feat=[]
    for para in paras:
        n = para[0]
        Q = 10 ** para[1]
        fc = 10 ** para[2]
        a = para[3]
        ch = para[4]
        kl = 10 ** para[5]
        rs = 10 ** para[6]
        lgrs=para[6]
        lgfc=para[2]
        lgQ=para[1]
        # lgr=para[7]
        lgkl=para[5]
        # r = 10 ** para[7]
        feat.append([lgrs,lgfc])
    return np.array(feat)

def feat_guess3(paras):
    feat=[]
    for para in paras:
        n = para[0]
        Q = 10 ** para[1]
        fc = 10 ** para[2]
        a = para[3]
        ch = para[4]
        kl = 10 ** para[5]
        rs = 10 ** para[6]
        # r = 10 ** para[7]
        feat.append([rs*(fc**4)])
    return np.array(feat)

def feat_guess4(paras):
    feat=[]
    for para in paras:
        Q = para[0]
        a = para[1]
        ch =para[2]
        fc = para[3]
        kl = para[4]
        n = para[5]
        r = para[6]
        rs = para[7]
        feat.append([rs*(fc**4)])
    return np.array(feat)

def feat_guess5(paras):
    feat=[]
    for para in paras:
        Q = para[0]
        a = para[1]
        ch =para[2]
        fc = para[3]
        kl = para[4]
        n = para[5]
        r = para[6]
        rs = para[7]
        feat.append([np.log10(rs),np.log10(fc)])
    return np.array(feat)

def feat_guess6(paras):
    feat=[]
    for para in paras:
        Q = para[0]
        a = para[1]
        ch =para[2]
        fc = para[3]
        kl = para[4]
        n = para[5]
        r = para[6]
        rs = para[7]
        feat.append([np.log10(rs),np.log10(fc),np.log10(kl)])
    return np.array(feat)

# %%  preprocess
idx=[0,1,2,3,4,5,6,7]

measure_data = pd.read_excel(r'9.15 Cell Growth/9-15 cell density.xlsx').to_numpy()
measure_data = measure_data[:,2].reshape(9,3).astype(np.float)

bounds_min = [0.6, -7, 0, 0.6, -1e-7, -6, -8, -3]
bounds_max = [1, -3, 4, 1, 1e-7, 2, -2, 4]

feat=loadxls('matlab_results.xlsx')
# print(feat)
feat=feat_guess6(feat)


np.random.seed(0)

# feat_norm = (feat -bounds_min) /(np.array(bounds_max)-np.array(bounds_min))
measure_data_norm = np.log10(measure_data)
# measure_data_norm = measure_data
# feat_norm=(feat-np.min(feat,axis=0))/(np.max(feat,axis=0)-np.min(feat,axis=0))
feat_norm=feat
# poly=PolynomialFeatures(degree=3)
# feat_norm=poly.fit_transform(feat_norm)


# print(feat_norm)
measure_var = np.var(measure_data_norm,axis=1)
yy = np.mean(measure_data_norm,axis=1)[idx]
xx = feat_norm[idx,]

# yy = np.hstack((measure_data_norm[idx,0], measure_data_norm[idx,1], measure_data_norm[idx,2]))
# xx = np.vstack((xx,xx,xx))
# %%  regression
kernel =C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#C(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)* RBF(np.repeat(10,7), np.repeat([1e-3, 1e3],7).reshape(2,7).T)

# regr=LinearRegression()
regr=LinearRegression(fit_intercept=True)
# gp = GaussianProcessRegressor(kernel=kernel, alpha=np.mean(measure_var),n_restarts_optimizer=10)
# gp=MLPRegressor(random_state=1, max_iter=500)

# selector = RFE(gp, n_features_to_select=1, step=1)
# selector.fit(xx,yy)
regr.fit(xx, yy)
y_pred= regr.predict(xx)
y_hat = regr.predict(xx)
# y_pred, sigma = gp.predict(xx, return_std=True)

# y_pred = selector.predict(xx)
cv_pred = cross_val_predict(regr, xx, y=yy, cv=KFold(yy.size), n_jobs=1)

# print(np.mean(np.abs(10**cv_pred-10**yy)/np.abs(10**yy)))

yy=10**yy
y_hat=10**y_hat
cv_pred=10**cv_pred
print(regr.coef_)
print(regr.intercept_)
print('Training R2 is '+str(1-np.var(y_hat-yy)/np.var(yy)))
print('Testing R2 is '+str(1-np.var(cv_pred-yy)/np.var(yy)))
print('RMSE is '+str(np.sqrt(np.mean(cv_pred-yy)**2)))
# %%  plot
plt.figure()
plt.plot(yy)
plt.show()

# plt.figure()
# plt.plot(feat[:,0])
# plt.show()

# %%  plot
plt.plot(yy,cv_pred,'o')
#plt.plot([5.4,6],[5.4,6])
plt.xlabel('truth')
plt.ylabel('predicted')
plt.show()

# %% save
# result=np.array([yy,cv_pred,y_pred]).T
# # np.savetxt('Phy-Scal.csv',result,delimiter=',')
# # np.savetxt('Phy-Insp.csv',result,delimiter=',')
# np.savetxt('KCV-EPex.csv',result,delimiter=',')


