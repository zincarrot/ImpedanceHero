import numpy as np


def log(func):
    def logf(*args, **kwargs):
        return np.log(func(*args, **kwargs))
    return logf

def real(func):
    def realf(**kwargs):
        return np.real(func(**kwargs))
    return realf


def eqmodel_test1(f, params): # ['E', 'R', ('R', 'C')]
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )

def eqmodel_test2(f, Q0, n0, R0, R1, C1): # ['CPE0', 'R0', ('R1', 'C1')]
    return ( 1 / ( Q0 * ((2j * np.pi * f) ** n0)) + R0 + 1 / ( 1 / ( R1 ) + 1 / ( 1 / (2j * np.pi * f * C1) ) ) )

def ppal_1(f, Q0, n0, R0, R1, R2, Q1, n1): # ['CPE0', 'R0', ('R1', ['R2', 'CPE1'])]
    return ( 1 / ( Q0 * ((2j * np.pi * f) ** n0)) + R0 + 1 / ( 1 / ( R1 ) + 1 / ( ( R2 + 1 / ( Q1 * ((2j * np.pi * f) ** n1)) ) ) ) )

def ppal_3(p, f): # ['CPE0', ('R1', ['R2', 'C0'])]
    return ( 1 / ( p['Q0'] * ((2j * np.pi * f) ** p['n0'])) + 1 / ( 1 / ( p['R1'] ) + 1 / ( ( p['R2'] + 1 / (2j * np.pi * f * p['C0']) ) ) ) )

def ppal_4(p, f): # ['CPE1', ('R1', ['C2', 'R2'])]
    return ( 1 / ( p['Q1'] * ((2j * np.pi * f) ** p['n1'])) + 1 / ( 1 / ( p['R1'] ) + 1 / ( ( 1 / (2j * np.pi * f * p['C2']) + p['R2'] ) ) ) )