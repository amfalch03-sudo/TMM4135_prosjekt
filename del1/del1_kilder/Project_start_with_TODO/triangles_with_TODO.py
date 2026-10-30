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

Array2D = npt.NDArray[np.float64]
Array1D = npt.NDArray[np.float64]


def tri3_area(ex: list[float], ey: list[float]) -> float:
    """
    Compute the area of a triangle element

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :return area of the triangle
    """

    tmp = np.array([[1, ex[0], ey[0]],
                     [1, ex[1], ey[1]],
                     [1, ex[2], ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area
    A = A2 / 2.0
    return A


def tri3_Bmatrix(ex: list[float], ey: list[float]) -> Array2D:
    """
    Compute the strain displacement matrix for a 3 node triangular membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :return  [3 x 6] strain displacement matrix
    """

    A = tri6_area(ex, ey)
    A2 = 2.0 * A

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    zi_px = np.zeros(3)  # Partial derivative with respect to x
    zi_py = np.zeros(3)  # Partial derivative with respect to y

    for i in range(3):
        j = cyclic_ijk[i + 1]
        k = cyclic_ijk[i + 2]
        zi_px[i] = (ey[j] - ey[k]) / A2
        zi_py[i] = (ex[k] - ex[j]) / A2

    B = np.array([
        [zi_px[0], 0, zi_px[1], 0, zi_px[2], 0],
        [0, zi_py[0], 0, zi_py[1], 0, zi_py[2]],
        [zi_py[0], zi_px[0], zi_py[1], zi_px[1], zi_py[2], zi_px[2]]])

    return B


def tri3_Kmatrix(
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    eq: Optional[Sequence[float]] = None,
) -> tuple[Array2D, Array2D]:
    """
    Compute the stiffness matrix for a two dimensional beam element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    A = tri6_area(ex, ey)
    A2 = 2.0 * A

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    zi_px = np.zeros(3)  # Partial derivative with respect to x
    zi_py = np.zeros(3)  # Partial derivative with respect to y

    for i in range(3):
        j = cyclic_ijk[i + 1]
        k = cyclic_ijk[i + 2]
        zi_px[i] = (ey[j] - ey[k]) / A2
        zi_py[i] = (ex[k] - ex[j]) / A2

    B = tri3_Bmatrix(ex, ey)

    Ke = (B.T @ D @ B) * A * th

    if eq is None:
        fe = np.zeros((6, 1))
    else:
        fx = A * th * eq[0] / 3.0
        fy = A * th * eq[1] / 3.0
        fe = np.array([[fx], [fy], [fx], [fy], [fx], [fy]])

    return Ke, fe


def tri3_cornerstresses(
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    elDispVec: Array2D,
) -> list[list[float]]:
    """
    Compute the corner stresses for all 3 corner nodes

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :return list of list of corner stresses
    """

    B = tri3_Bmatrix(ex, ey)

    strain = B @ elDispVec
    stress = D @ strain

    cornerStresses: list[list[float]] = []
    for inod in range(3):
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses


def zeta_partials_x_and_y(ex: list[float], ey: list[float]) -> tuple[Array2D, Array2D]:
    """
    Compute partials of area coordinates with respect to x and y.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    """

    tmp = np.array([[1, ex[0], ey[0]],
                    [1, ex[1], ey[1]],
                    [1, ex[2], ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    zeta_px = np.zeros(3)  # Partial derivative with respect to x
    zeta_py = np.zeros(3)  # Partial derivative with respect to y

    # TODO: fill out missing parts (or reformulate completely)
    zeta_px[0] = (tmp[1,2] - tmp[2,2])/A2
    zeta_px[1] = (tmp[2,2] - tmp[0,2])/A2
    zeta_px[2] = (tmp[0,2] - tmp[1,2])/A2

    zeta_py[0] = (tmp[2,1] - tmp[1,1])/A2
    zeta_py[1] = (tmp[0,1] - tmp[2,1])/A2
    zeta_py[2] = (tmp[1,1] - tmp[0,1])/A2

    return zeta_px, zeta_py

# Functions for 6 node triangle


def tri6_area(ex: list[float], ey: list[float]) -> float:
    tmp = np.array([[1, ex[0], ey[0]],
                    [1, ex[1], ey[1]],
                    [1, ex[2], ey[2]]])

    A = np.linalg.det(tmp) / 2

    return A


def tri6_shape_functions(zeta: Sequence[float]) -> Array2D:
    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    N6 = np.zeros(6)

    # TODO: fill out missing parts (or reformulate completely)
    for i in range(3):
        j = cyclic_ijk[i+1]

        N6[i] = zeta[i]*(zeta[i] - 1/2)*2
        N6[i+3] = zeta[i]*zeta[j]*4

    return N6


def tri6_shape_function_partials_x_and_y(
    zeta: Sequence[float],
    ex: list[float],
    ey: list[float],
) -> tuple[Array2D, Array2D]:
    zeta_px, zeta_py = zeta_partials_x_and_y(ex, ey)

    N6_px = np.zeros(6)
    N6_py = np.zeros(6)

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    # TODO: fill out missing parts (or reformulate completely)
    for i in range(3):
        j = cyclic_ijk[i+1]
        N6_px[i] = (4*zeta[i] - 1)*zeta_px[i]
        N6_px[i+3] = 4*(zeta[j]*zeta_px[i] + zeta[i]*zeta_px[j])
        N6_py[i] = (4*zeta[i] - 1)*zeta_py[i]
        N6_py[i+3] = 4*(zeta[j]*zeta_py[i] + zeta[i]*zeta_py[j])

    return N6_px, N6_py


def tri6_Bmatrix(zeta: Sequence[float], ex: list[float], ey: list[float]) -> Array2D:
    nx, ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.zeros((3, 12))

    # TODO: fill out missing parts (or reformulate completely)
    
    for j in range(6):
        Bmatrix[0,2*j] = nx[j]
        Bmatrix[2,2*j] = ny[j]
        Bmatrix[1,2*j+1] = ny[j]
        Bmatrix[2,2*j+1] = nx[j]

    return Bmatrix

# print(tri6_Bmatrix([0.2, 0.3, 0.5],))

def tri6_Kmatrix(
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    eq: list[float] | None = None,
) -> tuple[Array2D, Array2D]:
    """
    Compute the stiffness matrix for a two dimensional triangular element.

    :param list ex: element x coordinates [x1, x2, x3, x4, x5, x6]
    :param list ey: element y coordinates [y1, y2, y3, y4, y5, y6]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [12 x 12]
    :return mat fe: consistent load vector [12 x 1] (if eq!=None)
    """

    zetaInt = np.array([[0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])

    wInt = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    A = tri6_area(ex, ey)

    Ke = np.zeros((12, 12))
    fe = np.zeros((12, 1))

    # TODO: fill out missing parts (or reformulate completely)
    
    # TODO: remove/replace this with actual stiffness matrix computation
    for i in range(3):
        B = tri6_Bmatrix(zetaInt[i], ex, ey)
        Ke += A * th * wInt[i] * (B.T @ D @ B)

    if eq is not None:
        # TODO: compute the consistent load vector based on the distributed loads eq
        for g in range(3): 
            N6 = tri6_shape_functions(zetaInt[g])
            for a in range(6):
                fe[2*a,] += A*th*wInt[g]*N6[a].T*eq[0]
                fe[2*a+1,] += A*th*wInt[g]*N6[a].T*eq[1]

    return Ke, fe


def tri6_cornerstresses(
    ex: list[float],
    ey: list[float],
    D: Array2D,
    th: float,
    elDispVec: Array2D,
) -> list[list[float]]:
    """
    Compute the corner stresses for all 3 corner nodes

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :return list of list of corner stresses
    """

    # Corner values
    zetaCorner = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

    cornerStresses: list[list[float]] = []
    for i in range(3):
        # TODO: Compute the correct corner stresses here
        B = tri6_Bmatrix(zetaCorner[i], ex, ey)
        stress = (D @ B @ elDispVec).flatten() #(3,1)
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses

  
