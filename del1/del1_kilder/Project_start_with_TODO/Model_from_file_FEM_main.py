# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:38:14 2018

@author: bjohau
"""
import numpy as np
import fem_utilities as fem_util
import fem_models
import copy

# Distributed load in x and y, load pr unit area
eq = np.array([0.,1.0e3])
eq = np.array([0.,0.])
eq = None  # No distributed load

model = fem_models.ModelFromInputFile("Test1_input.yml")

model.vtu_write_connected_mesh("GeometryOnly_ConnectedMesh.vtu")

# Assemble stiffness matrix
K, R = model.assemble_stiffness_matrix_and_load_vector()

# Solve the system of equations
r, R0 = fem_util.solveq(K, R, model.bcdofs)

# Compute element corner stresses
elementCornerStresses = model.compute_element_corner_stresses(r)

# Sum of reaction forces
R0Sum = np.zeros(2)
ndofs = len(R0)
for i in range(0,ndofs,2):
    R0Sum[0] += R0[i  ,0]
    R0Sum[1] += R0[i+1,0]
print("Total reaction force: x:{:12.3e} y:{:12.3e})".format(R0Sum[0],R0Sum[1]))

# Draw the displacements and stresses
model.vtu_write_stl_style_mesh("Results.vtu",dispVector=r,elementCornerStresses=elementCornerStresses)

