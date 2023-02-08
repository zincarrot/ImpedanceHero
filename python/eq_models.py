import numpy as np


def eqmodel_test1(f, params):
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )
def eqmodel_test2(f, params):
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )
def eqmodel_test3(f, params):
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )