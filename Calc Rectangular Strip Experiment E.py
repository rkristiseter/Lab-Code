# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:00 2021

@author: Robin Kristiseter
"""

#Code for Calculating G and Error for Brass and Steel for Lab Experiment E, Physics DLM, University of Bristol

import numpy as np
from sympy import symbols, diff


L, M, R0, R, b, P, d, dL, dM, dR0, dR, db, dP, dd = symbols('L M R0 R b P d dL dM dR0 dR db dP dd', real=True)
f = ((L*M*(R**2-R0))/(P**2*b*d**3))

f2 = (diff(f, L)*dL + diff(f, M)*dM + diff(f, R0)*dR0 + diff(f, R)*dR + diff(f, d)*dd + diff(f, b)*db + diff(f, P)*dP)

#print(f2)

const = 24*np.pi**2

R0 = #Add R0 for Brass or Steel
M = 0.1
R = 116*10**-3
P =  #Add period of Brass or Steel
L = 30.3*10**-3
b = 2.3*10**-3
d = 0.128*10**-3

dR0 = 0.04*10**-3
dM = 1*10**-3
dR = 0.02 *10**-3
dP = 0.44
dL = dR
db = dR
dd = dR

G = const*((L*M*(R**2-R0))/(P**2*b*d**3))

err0 = 2*L*M*R*dR/(P**2*b*d**3) - L*M*dR0/(P**2*b*d**3) - 3*L*M*dd*(R**2 - R0)/(P**2*b*d**4) - L*M*db*(R**2 - R0)/(P**2*b**2*d**3) - 2*L*M*dP*(R**2 - R0)/(P**3*b*d**3) + L*dM*(R**2 - R0)/(P**2*b*d**3) + M*dL*(R**2 - R0)/(P**2*b*d**3)
err = err0*const

print(G)
print(err)


