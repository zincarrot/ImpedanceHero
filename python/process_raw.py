# -*- coding: utf-8 -*-
"""
@author: zincarrot
"""

import numpy as np
import pandas as pd
import os
import regex as re
from tqdm import tqdm


def process_raw(fname,flist):
    '''
    Process raw impedance data.
    fname: file name of raw data
    flist: frequency list (read from parameter)
    '''
    imp_data = pd.read_excel(fname, index_col=None).to_numpy()
    imp_data = np.reshape(imp_data, [imp_data.shape[0], len(flist), 5])[:, :, 2:4]
    imp_data[imp_data == 0] = np.nan
    imp_data[imp_data>1e30] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.mean(np.nanstd(imp_data, axis=0),axis=1)
    # imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)
    imp_real=imp_mean[:, 0]
    imp_imag=imp_mean[:, 1]
    return imp_real,imp_imag,imp_std

def get_flist(fname):
    with open(fname) as f:
        for line in f:
            if line.split(': ')[0]=='Signal Frequency (Hz)':
                flist=line.split(': ')[1].split('.000000')
                flist.pop()
                flist=[float(i) for i in flist]
                # print(flist)
                return flist

def saveprocessed(savedir,flist,impedance,std=None):
    processed_data=np.vstack(flist,impedance).T
    print(processed_data)
    np.savetxt(savedir+'\\data.csv',processed_data,delimiter=',')
    if std!=None:
        processed_std=np.vstack(flist,std).T
        np.savetxt(savedir+'\\std.csv',processed_std,delimiter=',')

def process_raw_all(folder):
    '''
    Process all raw impedance data in a folder.
    '''

    for dir in tqdm(os.listdir(folder)):
        savedir=folder+'\\..\\..\\processed_data\\'+folder.split('\\')[-1]
        os.makedirs(savedir,exist_ok=True)
        if re.match('\d{8}_Parameter_Test_1.txt',dir):
            flist=get_flist(os.path.join(folder,dir))
            
        if re.match('\d{8}_Sensor\d+_Test_\d+.xlsx',dir):
            real,imag,std=process_raw(os.path.join(folder,dir),flist)
            processed_data=np.vstack([flist,real,imag])
            processed_std=np.vstack([flist,std])
            np.savetxt(savedir+'\\'+re.findall(r'Sensor\d+_Test_\d+',dir)[0]+'_data.csv',processed_data,delimiter=',')
            np.savetxt(savedir+'\\'+re.findall(r'Sensor\d+_Test_\d+',dir)[0]+'_std.csv',processed_std,delimiter=',')

if __name__ == '__main__':
    dirs=os.listdir('.\\raw_data')
    # fname=r'E:\\Dropbox (GaTech)\\Lab files\\code\\Impedance_fit\\raw_data\\20200818_saline\\20200818_Sensor0_Test_1 - Copy.xlsx'
    process_raw_all('E:\\Dropbox (GaTech)\\Lab files\\code\\Impedance_fit\\raw_data\\20200818_saline')
    
