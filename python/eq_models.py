import numpy as np


def eqmodel_test1(f, params): # ['E', 'R', ('R', 'C')]
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )

def eqmodel_test2(f, Q0, n0, R0, R1, C1): # ['CPE0', 'R0', ('R1', 'C1')]
    return ( 1 / ( Q0 * ((2j * np.pi * f) ** n0)) + R0 + 1 / ( 1 / ( R1 ) + 1 / ( 1 / (2j * np.pi * f * C1) ) ) )