# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:21:29 2020

@author: zincarrot
"""

import numpy as np
import pandas as pd
import os


def process_raw(fname,flist):
    '''
    Process raw impedance data.
    fname: file name of raw data
    flist: frequency list (read from parameter)
    '''
    sensor = 1
    file_name = '9.15 Cell Growth/20190915_Sensor{}_Test_{}.xlsx'.format(sensor, tt)
    imp_data = pd.read_excel(file_name).to_numpy()
    imp_data = np.reshape(imp_data, [imp_data.shape[0], 15, 5])[:, :, 2:4]
    imp_data[imp_data == 0] = np.nan
    imp_data[imp_data>1e30] = np.nan
    imp_mean = np.nanmean(imp_data, axis=0)
    imp_std = np.nanstd(imp_data, axis=0)
    imp_comp = imp_mean[:, 0] + imp_mean[:, 1] * (0 + 1j)


def process_raw_all(folder):
    '''
    Process all raw impedance data in a folder.
    '''
    print(folder)


if __name__ == '__main__':
    raw_data_folder='..\\raw_data'
    for root, dirs, files in os.walk(raw_data_folder):
        process_raw_all(os.path.join(root,dirs[0]))