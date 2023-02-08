import numpy as np


def eqmodel_test1(f, params): # ['E', 'R', ('R', 'C')]
    return ( 1 / ( params[0] * ((2j * np.pi * f) ** params[1])) + params[2] + 1 / ( 1 / ( params[3] ) + 1 / ( 1 / (2j * np.pi * f * params[4]) ) ) )