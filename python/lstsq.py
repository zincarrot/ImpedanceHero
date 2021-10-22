# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:38:10 2021

@author: wizar
"""

from scipy.linalg import lstsq
import numpy as np


A=[]
row1=[1,0,1.028]
row2=[4,-1,4.852]
row3=[1,-1,3.736]
row4=[0,1,-3.363]
A.append(row1)
A.append(row2)
A.append(row3)
A.append(row4)
A=np.array(A)

b=[0,3,0,0]
b=np.array(b)

x=lstsq(A,b,cond=None)
print(x)
print(A@x[0])