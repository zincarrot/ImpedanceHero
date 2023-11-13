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

def twish_1(p, f): 
    pp=p['pp'] # cell ratio
    Cm=p['Cm'] # membrane capacitance
    Rm=p['Rm'] # membrane resistance
    Cc=p['Cc'] # cell capacitance
    Q=p['Q'] # Q
    n=p['n'] # n
    Rl=p['Rl'] # media resistance
    Cs=p['Cs'] # stray capacitance

    Zm = Rm / (1+ 2j*np.pi*f*Rm*Cm)
    Zloc = 1/(pp/Zm+(1-pp)/(Zm+2j*np.pi*f*Cc))
    Zcpe = 1/((2j*np.pi*f)**n*Q)
    return 1/(1/(Zloc+Zcpe+Rl)+2j*np.pi*f*Cs)

def twish_2(p, f):  # RCRC model
    R1=p['R1'] # membrane resistance
    C1=p['C1'] # membrane capacitance
    R2=p['R2'] # 
    C2=p['C2'] # 
    Q=p['Q'] # Q
    n=p['n'] # n
    Rl=p['Rl'] # media resistance (longitudinal resistance)
    Cs=p['Cs'] # stray capacitance

    Z1=R1/(1+2j*np.pi*f*R1*C1)
    Z2=R2/(1+2j*np.pi*f*R2*C2)
    Zcpe=1/((2j*np.pi*f)**n*Q)
    Zloc=Z1+Z2
    return 1/(1/(Zloc+Zcpe+Rl)+2j*np.pi*f*Cs)

def twish_3(p, f):  # RCRC model
    R1=p['R1'] # membrane resistance
    C1=p['C1'] # membrane capacitance
    Q=p['Q'] # Q
    n=p['n'] # n
    Rl=p['Rl'] # media resistance (longitudinal resistance)
    Cs=p['Cs'] # stray capacitance

    Z1=R1/(1+2j*np.pi*f*R1*C1)
    Zcpe=1/((2j*np.pi*f)**n*Q)
    return 1/(1/(Z1+Zcpe+Rl)+2j*np.pi*f*Cs)

def twish_4(p, f):  # Gerischer model
    R1=p['R1'] # membrane resistance
    C1=p['C1'] # membrane capacitance
    d=p['d'] #
    rl=p['rl'] # media resistance (longitudinal resistance)
    Q=p['Q'] # Q
    n=p['n'] # n
    Rl=p['Rl'] # media resistance (longitudinal resistance)
    Cs=p['Cs'] # stray capacitance

    cw=C1/d
    rw=R1*d
    Z1=R1/(1+2j*np.pi*f*R1*C1)
    Zcpe=1/((2j*np.pi*f)**n*Q)
    ZG=np.sqrt(2j*np.pi*f*cw*rw+rl/rw)/(2j*np.pi*f*cw)
    Zcore=1/(1/ZG+1/Z1)
    return 1/(1/(Zcore+Zcpe+Rl)+2j*np.pi*f*Cs)

def twish_5(p, f):  # corrected model
    rl=p['rl'] # membrane resistance
    cw=p['cw'] # membrane capacitance
    rw=p['rw'] #
    d=p['d'] # media resistance (longitudinal resistance)
    D=p['D'] # D
    Q=p['Q'] # Q
    n=p['n'] # n
    L=p['L'] # media resistance (longitudinal resistance)
    Cs=p['Cs'] # stray capacitance

    Cw=cw*d
    Rw=rw/d
    Z1=Rw/(1+2j*np.pi*f*Rw*Cw)
    Zcpe=1/((2j*np.pi*f)**n*Q)
    ZG=np.sqrt(rw/((2j*np.pi*f*cw*rw+1)*rl**3))
    Zcore=1/(1/ZG+1/Z1)
    return 1/(1/(Zcore+Zcpe+rl*D+2j*np.pi*f*L)+2j*np.pi*f*Cs)