# -*- coding: utf-8 -*-
"""
Test av 9-noders firkantelement (Q9):
  - stivlegemebevegelser på B-matrisen og stivhetsmatrisen
  - konstant tøyning på B-matrisen
  - konsistent last summerer til total volumlast
"""

import numpy as np
import quads_with_TODO as quad

# ----- Topology -------------------------------------------------
# Skjevt element med litt buede kanter, slik at Jacobi-matrisen ikke er konstant.
# Nodenummerering: hjørner 1-4 ved (xsi,eta) = (1,1), (-1,1), (-1,-1), (1,-1),
# midtnoder 5-8 mellom 1-2, 2-3, 3-4, 4-1, og senternode 9.
ex = np.array([2.2, 0.0, 0.1, 2.0,  1.1, 0.0, 1.0, 2.15,  1.05])
ey = np.array([1.5, 1.2, 0.0, 0.2,  1.4, 0.6, 0.05, 0.85, 0.7])

th = 0.1

E  = 2.1e11
nu = 0.3

D = np.array([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

eq = [1.0, 3.0]

rigX = np.zeros((18,1))
rigY = np.zeros((18,1))
rigR = np.zeros((18,1))

for i in range(9):
    rigX[i*2  ,0] = 1.0
    rigY[i*2+1,0] = 1.0
    rigR[i*2  ,0] = ey[i]
    rigR[i*2+1,0] = -ex[i]

Ke, fe = quad.quad9_Kmatrix(ex,ey,D,th,eq)

print('Stiffness matrix:\n', Ke)
print('Consistent forces:\n', fe)

fx = Ke @ rigX
fy = Ke @ rigY
fr = Ke @ rigR

print('Force from rigX translation:\n',fx)
print('Force from rigY translation:\n',fy)
print('Force from rigR rotation:\n',fr)
print('Max |Ke @ rig| / max |Ke|: {:.2e}'.format(
    np.max(np.abs(np.hstack([fx, fy, fr]))) / np.max(np.abs(Ke))))

# Sum av konsistente krefter skal være lik total volumlast b * t * areal
area = 0.0
gp, gw = quad.gauss_points(3)
for i in range(3):
    for j in range(3):
        _, detJ = quad.quad9_Bmatrix(gp[i], gp[j], ex, ey)
        area += detJ * gw[i] * gw[j]
print('Sum fe x: {:.6f}  forventet {:.6f}'.format(fe[0::2].sum(), eq[0]*th*area))
print('Sum fe y: {:.6f}  forventet {:.6f}'.format(fe[1::2].sum(), eq[1]*th*area))


constEx = np.zeros((18,1))
constEy = np.zeros((18,1))
constGamma1 = np.zeros((18,1))
constGamma2 = np.zeros((18,1))

for i in range(9):
    constEx[i*2  ,0] = ex[i]
    constEy[i*2+1,0] = ey[i]
    constGamma1[i*2  ,0] = ey[i]
    constGamma2[i*2+1,0] = ex[i]

for xsi, eta in [(-0.5, 0.3), (0.0, 0.0), (1.0, -1.0)]:
    print('xsi, eta', xsi, eta)

    Be, detJ = quad.quad9_Bmatrix(xsi, eta, ex, ey)

    print('Be @ rigX:\n', Be @ rigX)
    print('Be @ rigY:\n', Be @ rigY)
    print('Be @ rigR:\n', Be @ rigR)

    Ex = Be @ constEx
    Ey = Be @ constEy
    G1  = Be @ constGamma1
    G2  = Be @ constGamma2
    print('Ex:\n',Ex)
    print('Ey:\n',Ey)
    print('G:\n',G1)
    print('G:\n',G2)
