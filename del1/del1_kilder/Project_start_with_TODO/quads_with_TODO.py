# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt
import sys

Array2D = npt.NDArray[np.float64]
Array1D = npt.NDArray[np.float64]

def gauss_points(iRule: int) -> tuple[list[float], list[float]]:
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi: float, eta: float) -> Array1D:
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    N = np.array((
        0.25 * (1.0 + xsi) * (1.0 + eta),
        0.25 * (1.0 - xsi) * (1.0 + eta),
        0.25 * (1.0 - xsi) * (1.0 - eta),
        0.25 * (1.0 + xsi) * (1.0 - eta),
    ))
    return N

def quad4_shapefuncs_grad_xsi(xsi: float, eta: float) -> Array1D:
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----
    Ndxi = np.array((
        0.25*(1 + eta),
        0.25*(-1 - eta),
        0.25*(-1 + eta),
        0.25*(1 - eta),
    ))
    return Ndxi


def quad4_shapefuncs_grad_eta(xsi: float, eta: float) -> Array1D:
    """
    Calculates derivatives of shape functions wrt. eta
    """
    # ----- Derivatives of shape functions with respect to eta -----
    Ndeta = np.array((
        0.25*(1.0 + xsi),
        0.25*(1.0 - xsi),
        0.25*(-1.0 + xsi),
        0.25*(-1.0 - xsi),
    ))
    return Ndeta

def quad4_Bmatrix(xsi: float, eta: float, ex, ey) -> tuple[Array2D, float]:
    """
    Strain-displacement matrix B (3x8) and det(J) at (xsi, eta)
    """
    Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
    Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)

    # Matrix H and G defined according to page 52 of Waløens notes
    H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
    G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

    J = G @ H
    invJ = np.linalg.inv(J)  # Inverse of Jacobian
    detJ = np.linalg.det(J)  # Determinant of Jacobian

    dN = invJ @ G  # Derivatives of shape functions with respect to x and y
    dNdx = dN[0]
    dNdy = dN[1]

    B = np.zeros((3,8))
    for i in range(4):
        B[0][2*i] = dNdx[i]
        B[1][2*i+1] = dNdy[i]
        B[2][2*i+1] = dNdx[i]
        B[2][2*i] = dNdy[i]

    return B, detJ



def quad4_Kmatrix(
        ex: list[float],
        ey: list[float], 
        D: Array2D, 
        thickness: float,
       eq: Optional[Sequence[float]] = None,
) -> tuple[Array2D, Array2D]:
    """
    Compute the stiffness matrix for a four node membrane element.

    Parameters:

        ex  = [x1 ... x4]           Element coordinates. Row matrix
        ey  = [y1 ... y4]
        D   =           Constitutive matrix
        thickness:      Element thickness
        eq = [bx; by]       bx:     body force in x direction
                            by:     body force in y direction

    Returns:

        Ke : element stiffness matrix (8 x 8)
        fe : equivalent nodal forces (4 x 1)

    """
    t = thickness

    if eq is None:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Strain displacement matrix calculated at position xsi, eta
            B, detJ = quad4_Bmatrix(xsi, eta, ex, ey)

            # Displacement interpolation at position xsi, eta
            N2 = np.zeros((2,8))
            for i in range(4):
                N2[0][2*i] = N1[i]
                N2[1][2*i+1] = N1[i]


            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]


    return Ke, fe  # Returns stiffness matrix and nodal force vector


def quad4_cornerstresses( 
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    elDispVec: Array2D,
) -> list[list[float]]:
    """
    Compute the corner stresses for all 3 corner nodes

    :param list ex: element x coordinates [x1, x2, x3, x4]
    :param list ey: element y coordinates [y1, y2, y3, y4]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :return list of list of corner stresses
    """

    # Corner natural coordinates
    xi_eta_corner = np.array([[1.0, 1.0],
                           [-1.0, 1.0],
                           [-1.0, -1.0],
                           [1.0, -1.0]])

    cornerStresses = []

    for i in range(4):
        B, detJ = quad4_Bmatrix(xi_eta_corner[i][0], xi_eta_corner[i][1], ex, ey)
        stress = D @ (B @ elDispVec)
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses


# ---------------------------------------------------------------------------
# 9-noders Lagrange-element (Q9)
#
# Nodenummerering i naturlige koordinater (samme hjørnerekkefølge som Q4):
#   hjørner   1:( 1, 1)  2:(-1, 1)  3:(-1,-1)  4:( 1,-1)
#   midtnoder 5:( 0, 1)  6:(-1, 0)  7:( 0,-1)  8:( 1, 0)   (5 mellom 1-2, 6 mellom 2-3, ...)
#   senter    9:( 0, 0)
# ---------------------------------------------------------------------------
quad9_xsi_eta_nodes = np.array([
    [ 1.0,  1.0], [-1.0,  1.0], [-1.0, -1.0], [ 1.0, -1.0],
    [ 0.0,  1.0], [-1.0,  0.0], [ 0.0, -1.0], [ 1.0,  0.0],
    [ 0.0,  0.0]])


def _lagrange2(s: float, s_node: float) -> float:
    """1D kvadratisk Lagrange-polynom som er 1 i s_node og 0 i de to andre av {-1, 0, 1}"""
    if s_node > 0.5:
        return 0.5 * s * (s + 1.0)
    if s_node < -0.5:
        return 0.5 * s * (s - 1.0)
    return 1.0 - s * s


def _lagrange2_grad(s: float, s_node: float) -> float:
    """Derivert av _lagrange2 med hensyn på s"""
    if s_node > 0.5:
        return s + 0.5
    if s_node < -0.5:
        return s - 0.5
    return -2.0 * s


def quad9_shapefuncs(xsi: float, eta: float) -> Array1D:
    """
    Formfunksjoner for 9-noders element evaluert i (xsi, eta)
    """
    N = np.zeros(9)
    for i, (xi_i, eta_i) in enumerate(quad9_xsi_eta_nodes):
        N[i] = _lagrange2(xsi, xi_i) * _lagrange2(eta, eta_i)
    return N


def quad9_shapefuncs_grad_xsi(xsi: float, eta: float) -> Array1D:
    """
    Deriverte av formfunksjonene med hensyn på xsi
    """
    Ndxsi = np.zeros(9)
    for i, (xi_i, eta_i) in enumerate(quad9_xsi_eta_nodes):
        Ndxsi[i] = _lagrange2_grad(xsi, xi_i) * _lagrange2(eta, eta_i)
    return Ndxsi


def quad9_shapefuncs_grad_eta(xsi: float, eta: float) -> Array1D:
    """
    Deriverte av formfunksjonene med hensyn på eta
    """
    Ndeta = np.zeros(9)
    for i, (xi_i, eta_i) in enumerate(quad9_xsi_eta_nodes):
        Ndeta[i] = _lagrange2(xsi, xi_i) * _lagrange2_grad(eta, eta_i)
    return Ndeta


def quad9_Bmatrix(xsi: float, eta: float, ex, ey) -> tuple[Array2D, float]:
    """
    Tøyning-forskyvningsmatrise B (3x18) og det(J) i (xsi, eta)
    """
    Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
    Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)

    H = np.transpose([ex, ey])    # 9x2: globale koordinater
    G = np.array([Ndxsi, Ndeta])  # 2x9: gradienter i naturlige koordinater

    J = G @ H
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)

    dN = invJ @ G  # Deriverte mhp. x og y
    dNdx = dN[0]
    dNdy = dN[1]

    B = np.zeros((3, 18))
    for i in range(9):
        B[0, 2*i]   = dNdx[i]
        B[1, 2*i+1] = dNdy[i]
        B[2, 2*i]   = dNdy[i]
        B[2, 2*i+1] = dNdx[i]

    return B, detJ


def quad9_Kmatrix(
        ex: list[float],
        ey: list[float],
        D: Array2D,
        thickness: float,
        eq: Optional[Sequence[float]] = None,
) -> tuple[Array2D, Array2D]:
    """
    Calculates the stiffness matrix for a 9 node isoparametric element in plane stress

    :param list ex: element x coordinates [x1, ..., x9]
    :param list ey: element y coordinates [y1, ..., y9]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [18 x 18]
    :return mat fe: consistent load vector [18 x 1] (if eq!=None)
    """
    t = thickness

    if eq is None:
        f = np.zeros((2, 1))
    else:
        f = np.array([eq], dtype=float).T

    Ke = np.zeros((18, 18))
    fe = np.zeros((18, 1))

    numGaussPoints = 3  # 3x3 Gauss: full integrasjon for Q9
    gp, gw = gauss_points(numGaussPoints)

    for iGauss in range(numGaussPoints):
        for jGauss in range(numGaussPoints):
            xsi = gp[iGauss]
            eta = gp[jGauss]
            w = gw[iGauss] * gw[jGauss]

            B, detJ = quad9_Bmatrix(xsi, eta, ex, ey)

            N = quad9_shapefuncs(xsi, eta)
            N2 = np.zeros((2, 18))
            for i in range(9):
                N2[0, 2*i]   = N[i]
                N2[1, 2*i+1] = N[i]

            Ke += B.T @ D @ B * detJ * t * w
            fe += N2.T @ f * detJ * t * w

    return Ke, fe


def quad9_cornerstresses(
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    elDispVec: Array2D,
) -> list[list[float]]:
    """
    Compute the corner stresses for all 4 corner nodes

    :param list ex: element x coordinates [x1, x2, x3, x4, x5, x6, x7, x8, x9]
    :param list ey: element y coordinates [y1, y2, y3, y4, y5, y6, y7, y8, y9]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :return list of list of corner stresses
    """

    # Corner natural coordinates
    xi_eta_corner = np.array([[1.0, 1.0],
                           [-1.0, 1.0],
                           [-1.0, -1.0],
                           [1.0, -1.0]])

    cornerStresses = []

    for i in range(4):
        B, detJ = quad9_Bmatrix(xi_eta_corner[i][0], xi_eta_corner[i][1], ex, ey)
        stress = (D @ (B @ elDispVec)).flatten()
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses
