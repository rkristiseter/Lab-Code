# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:00 2021

@author: Robin Kristiseter
"""
#Code for calclulating Shear Modulus and Error of Nylon for Lab Experiment E, Physics DLM, University of Bristol

import numpy as np
from sympy import symbols, diff

L, r, R, R0, P, M, dL, dr, dR, dR0, dP, dM = symbols('L r R R0 P M dL dr dR dR0 dP dM', real=True)
f = L*M*(R**2-R0)/(P**2*r**4)
f2 = diff(f, L)*dL + diff(f, r)*dr + diff(f, R)*dR + diff(f, R0)*dR0 + diff(f, P)*dP + diff(f, M)*dM
#print(f2)

const = 16*np.pi

L = 30.3*10**-3
r = 0.238*10**-3
R = 116.98*10**-3
R0 = -14752*10**-6
P = # Add period of nylon
M = 0.1

dL = 0.02*10**-3
dr = dL
dR = dL
dR0 = 2*dL
dP = 0.44
dM = 1*10**-3

err0 = 2*L*M*R*dR/(P**2*r**4) - L*M*dR0/(P**2*r**4) - 4*L*M*dr*(R**2 - R0)/(P**2*r**5) - 2*L*M*dP*(R**2 - R0)/(P**3*r**4) + L*dM*(R**2 - R0)/(P**2*r**4) + M*dL*(R**2 - R0)/(P**2*r**4)
err = err0*const

G = const*L*M*(R**2-R0)/(P**2*r**4)

print(G)
print(err)