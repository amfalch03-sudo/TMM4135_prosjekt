import numpy as np
import triangles_with_TODO as tri
import quads_with_TODO as quad

import fem_utilities as fem_util
import copy
import meshio
import yaml

class FemModel:
    """
    Description
    -----------
    Parent class containing the implementation for the particular element
    models.
    """

    def __init__(self):
        self.coords = []     # nodal coordinates for all nodes
        self.elnods = []     # element node connectivity
        self.eldofs = []     # element degree of freedom mapping. Length 2*<numNodes> for each element
        self.bcdofs = []     # 0-based list of fixed boundary conditions
        self.nodeloads = []  # node loads in x and y for all global nodes
        self.eq = None

        self.thickness = 0.1  # Default element properties, i.e. thickness
        E  = 2.1e11 # Steel Young's modulus in Pa
        nu = 0.3    # Steel Poisson's ratio
        self.Dmat = np.array([   # Default material matrix
                [ 1.0,  nu,  0.],
                [  nu, 1.0,  0.],
                [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)


    def assemble_stiffness_matrix_and_load_vector(self):
        ndofs = len(self.coords) * 2
        K = np.zeros((ndofs,ndofs))
        R = np.zeros((ndofs,1))

        #Set the load at the right hand edge
        for i in range(len(self.nodeloads)):
            R[(i*2+0),0] += self.nodeloads[i][0]
            R[(i*2+1),0] += self.nodeloads[i][1]

        # Element thickness stored in ep[1]
        thickness = self.ep[1]

        numElements = len(self.elnods)
        for iel in range(numElements):
            ex_el = []
            ey_el = []

            nElNodes = len(self.elnods[iel])
            for inod in range(nElNodes):
                ex_el.append(self.coords[self.elnods[iel][inod]][0])
                ey_el.append(self.coords[self.elnods[iel][inod]][1])

            if nElNodes == 3:
                K_el, f_el = tri.tri3_Kmatrix(ex_el,ey_el,self.Dmat,thickness,self.eq)
            elif nElNodes == 6:
                K_el, f_el = tri.tri6_Kmatrix(ex_el,ey_el,self.Dmat,thickness,self.eq)
            elif nElNodes == 4:
                K_el, f_el = quad.quad4_Kmatrix(ex_el,ey_el,self.Dmat,thickness,self.eq)
            elif nElNodes == 9:
                K_el, f_el = quad.quad9_Kmatrix(ex_el,ey_el,self.Dmat,thickness,self.eq)

            fem_util.assem(self.eldofs[iel], K, K_el, R, f_el)

        return K, R

    def compute_element_corner_stresses(self,dispVector):

        elementCornerStresses = []
        numElements = len(self.elnods)
        thickness = self.thickness

        for iel in range(numElements):
            ex_el = []
            ey_el = []

            nElNodes = len(self.elnods[iel])
            for inod in range(nElNodes):
                ex_el.append(self.coords[self.elnods[iel][inod]][0])
                ey_el.append(self.coords[self.elnods[iel][inod]][1])

            nElDofs = len(self.eldofs[iel])
            elDisp = np.zeros(nElDofs)
            for idof in range(nElDofs):
                elDisp[idof] = dispVector[self.eldofs[iel][idof],0]

            if nElNodes == 3:
                cornerStresses = tri.tri3_cornerstresses(ex_el,ey_el,self.Dmat,thickness,elDisp)
            elif nElNodes == 6:
                cornerStresses = tri.tri6_cornerstresses(ex_el,ey_el,self.Dmat,thickness,elDisp)
            elif nElNodes == 4:
                cornerStresses = quad.quad4_cornerstresses(ex_el,ey_el,self.Dmat,thickness,elDisp)
            elif nElNodes == 9:
                cornerStresses = quad.quad9_cornerstresses(ex_el,ey_el,self.Dmat,thickness,elDisp)

            elementCornerStresses.append(copy.deepcopy(cornerStresses))

        return elementCornerStresses


    def vtu_write_stl_style_mesh(self, fileName, dispVector=None, elementCornerStresses=None):
        # Write geomtry output to Paraview as a .vtu file
        # the mesh is written in a "stl style" where each element is written as a separate triangle or quad, i.e. the nodes are not shared between elements. This is done to allow for element stresses being discontinous across element boundaries. 
        # Both 3 and 6 node triangles are written as 3 node triangles
        # Both 4 and 9 node quads are written as 4 node quads
        points = []
        cells = []
        dispArr = []
        stressArr = []

        for iel in range(len(self.elnods)):
            numElementNodes = len(self.elnods[iel])
            if numElementNodes == 3 or numElementNodes == 6:
                numCorners = 3
            elif numElementNodes == 4 or numElementNodes == 9:
                numCorners = 4

            # Add the points
            for ip in range(numCorners):
                inode = self.elnods[iel][ip]
                points.append([self.coords[inode][0], self.coords[inode][1], 0])

                if dispVector is not None:
                    dispArr.append([dispVector[inode*2,0],dispVector[inode*2+1,0],0])

                if elementCornerStresses is not None:
                    stressArr.append(elementCornerStresses[iel][ip])

            ip = len(points) - numCorners
            if numCorners == 3:
                cells.append(("triangle", [[ip,ip+1,ip+2]]))
            elif numCorners == 4:
                cells.append(("quad", [[ip,ip+1,ip+2,ip+3]]))

        point_data = {}
        if dispVector is not None:
            point_data["displacement"] = dispArr
        if elementCornerStresses is not None:
            point_data["stress"] = stressArr

        mesh = meshio.Mesh(points, cells, point_data=point_data)
        mesh.write(fileName)

    def vtu_write_connected_mesh(self,fileName):
        """
        Write the connected mesh to a .vtu file for visualization in Paraview.
        The routine writes the mesh to a .vtu file with the nodes shared between elements, i.e. the mesh is connected.
        This is done to allow for visualization of the mesh in Paraview.
        The element stresses are not written to this file, as they are discontinuous across element boundaries.
        """
        points = []
        for ip in range(len(self.coords)):
            points.append([self.coords[ip][0], self.coords[ip][1], 0])

        cells = []
        for ie in range(len(self.elnods)):
            numElementNodes = len(self.elnods[ie])
            if numElementNodes == 3 or numElementNodes == 6:
                cells.append(("triangle", [self.elnods[ie][:3]]))
            elif numElementNodes == 4 or numElementNodes == 9:
                cells.append(("quad", [self.elnods[ie][:4]]))

        mesh = meshio.Mesh(points, cells)
        mesh.write("geo_connected_nodes.vtu")


class CantileverModel(FemModel):
    def __init__(self,length, height, numElementNodes, numNodesX, numNodesY, endloadXY, eq=None):
        """
        Creates a cantilever FEM model with given dimensions and mesh density

        Args:
            length (float): length
            height (float): height
            numElementNodes (int): element type by number of nodes (3,4,6,9)
            numNodesX (int): number of nodes in the x direction
            numNodesY (int): number of nodes in the y direction
            endloadXY (load in x and y): end load in x and y directions
            eq (list/array, optional): distributed load (i.e. gravity-ish). Defaults to None.
                                       Should be a list/array [fx, fy] when not None
        """

        FemModel.__init__(self)

        # number of patches that will fit a 9 node element
        numPatchX = (numNodesX-1) // 2
        numPatchX = 1 if numPatchX < 1 else numPatchX
        numPatchY = (numNodesY-1) // 2
        numPatchY = 1 if numPatchY < 1 else numPatchY

        numNodesX = numPatchX*2 + 1
        numNodesY = numPatchY*2 + 1

        if numElementNodes == 6 or numElementNodes == 9:
            numElementsX = (numNodesX-1) // 2
            numElementsY = (numNodesY-1) // 2
        else:
            numElementsX = numNodesX -1
            numElementsY = numNodesY -1

        # Cantilever with dimensions H x L x thickness
        H         = height
        L         = length
        thickness =  0.1

        # Distributed load in x and y, load pr unit area
        if eq is not None:
            self.eq = copy.deepcopy(eq)

        # Material properties and thickness

        self.ep = [1,thickness]
        E  = 2.1e11
        nu = 0.3
        self.Dmat = np.array([
                [ 1.0,  nu,  0.],
                [  nu, 1.0,  0.],
                [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

        numElements = numElementsX * numElementsY
        if numElementNodes in [3,6]:
            numElements *= 2

        L_elx = L / (numNodesX-1)
        L_ely = H / (numNodesY-1)

        # Set the node coordinates 

        for i in range(numNodesX):
            for j in range(numNodesY):
                self.coords.append([L_elx * i, L_ely * j])
                self.nodeloads.append([0.0,0.0])  # Initialize node loads to zero

        # Set the element connectivites and element dofs
        for ip in range(numPatchX):
            ii = ip*2
            for jp in range(numPatchY):
                jj = jp*2
                # 0 based node numbers, 9 nodes of a 3x3 patch
                nod9 = np.array([
                    (ii  )*numNodesY + (jj  ),
                    (ii+1)*numNodesY + (jj  ),
                    (ii+2)*numNodesY + (jj  ),
                    (ii  )*numNodesY + (jj+1),
                    (ii+1)*numNodesY + (jj+1),
                    (ii+2)*numNodesY + (jj+1),
                    (ii  )*numNodesY + (jj+2),
                    (ii+1)*numNodesY + (jj+2),
                    (ii+2)*numNodesY + (jj+2)],'i')

                if numElementNodes == 3:
                    for i in range(2):
                        for j in range(2):
                            self.elnods.append([nod9[3*i+j],nod9[3*i+j+1],nod9[3*(i+1)+j+1]])
                            self.elnods.append([nod9[3*(i+1)+j+1],nod9[3*(i+1)+j],nod9[3*i+j]])
                elif numElementNodes == 6:
                    self.elnods.append([nod9[0],nod9[2],nod9[8],nod9[1],nod9[5],nod9[4]])
                    self.elnods.append([nod9[8],nod9[6],nod9[0],nod9[7],nod9[3],nod9[4]])
                elif numElementNodes == 4:
                    for i in range(2):
                        for j in range(2):
                            self.elnods.append([nod9[3*i+j],nod9[3*i+j+1],nod9[3*(i+1)+j+1],nod9[3*(i+1)+j]])
                elif numElementNodes == 9:
                    self.elnods.append([nod9[0],nod9[2],nod9[8],nod9[6],
                                     nod9[1],nod9[5],nod9[7],nod9[3],
                                     nod9[4]])


        for iel in range(len(self.elnods)):
            dofs = []
            for inod in range(len(self.elnods[iel])):
                dofs.append(self.elnods[iel][inod] * 2) # The x dofs
                dofs.append(self.elnods[iel][inod] * 2 + 1)  # The y dofs

            self.eldofs.append(dofs)

        # Set fixed boundary condition on left side, i.e. nodes 0-nNody
        #bc = np.array(np.zeros(numNodesY*2),'i')
        #idof = 1
        for i in range(numNodesY):
            self.bcdofs.append(i*2  )  # 0-based dof i.e. fixing x
            self.bcdofs.append(i*2+1)  # 0-based dof i.e. fixing y


        # Set node loads at right hand side
        fx = endloadXY[0] / numNodesY
        fy = endloadXY[1] / numNodesY
        for i in range(numNodesY):
            self.nodeloads[-(i+1)] = [fx, fy]  # 0-based dof i.e. fixing x


class ModelFromInputFile(FemModel):
    def __init__(self, file_path):
        """
        Reads a FEM model from a input file using yaml format. The input file should contain the following fields:
        - coords: list of nodal coordinates, given as a list of lists [[id: xyz: x1,y1],[x2,y2],...]
        - elnods: list of element connectivities given as a list of lists [[id: nodes: n1,n2,n3,...],[n4,n5,n6,...],...]
        - nodeloads: list of nodal loads   given as a list of lists [[id: load: fx,fy],[fx2,fy2],...]
        - bcdofs: list of fixed boundary condition dofs
        - eq: distributed load. Same load in x and y for all elements. Should be a list/array [fx, fy] when not None


        Args:
            file_path (str): path to the input file containing model parameters
        """

        FemModel.__init__(self)

        # Read the data from the input file

        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        # Decipher the nodal coordinates, nodal loads, and boundary conditions from the input data

        nodes = data["Nodes"]

        # set up the coordinates and map of node ids to 0-based indices
        node_id_to_index: dict[int, int] = {}

        for i, node in enumerate(nodes):
            node_id = node['id']
            node_x = node['xy'][0]
            node_y = node['xy'][1]
            node_id_to_index[node_id] = i
            self.coords.append([node_x, node_y])
            fx = 0.0
            fy = 0.0
            if 'load' in node:
                fx = node['load'][0]
                fy = node['load'][1]
            self.nodeloads.append([fx, fy])  # Initialize node loads to zero
            if 'BC' in node:
                if node['BC'][0] == 1:
                    self.bcdofs.append(i * 2)  # 0-based dof i.e. fixing x
                if node['BC'][1] == 1:
                    self.bcdofs.append(i * 2 + 1)  # 0-based dof i.e. fixing y

        # Read additional node load not associated with a node in the "Nodes" section, if present
        node_loads = data["NodeLoads"]

        for i, node in enumerate(node_loads):
            node_id = node['node_id']
            idx_node =node_id_to_index[node_id]
            fx = 0.0
            fy = 0.0
            if 'load' in node:
                fx = node['load'][0]
                fy = node['load'][1]
            self.nodeloads[idx_node] += [fx, fy]

        # Read the nodal Boundary Conditions (BC) not associated with a node in the "Nodes" section, if present

        node_bcs = data["NodeBCs"]

        for i, node in enumerate(node_bcs):
            node_id = node['id']
            idx_node =node_id_to_index[node_id]
            if 'BC' in node:
                if node['BC'][0] == 1:
                    self.bcdofs.append(idx_node * 2)  # 0-based dof i.e. fixing x
                if node['BC'][1] == 1:
                    self.bcdofs.append(idx_node * 2 + 1)  # 0-based dof i.e. fixing 


        # Read the elements
        elements = data["Elements"]

        element_id_to_index: dict[int, int] = {}

        for i, element in enumerate(elements):
            id = element['id']
            element_id_to_index[id] = i
            node_ids = element['nodes']
            node_indices = [node_id_to_index[node_id] for node_id in node_ids]
            self.elnods.append(node_indices)

            numElementNodes = len(node_indices)
            type = element.get('type', None)
            if type is None:
                raise ValueError(f"Element type is not specified for element {i}. Must be 'T3', 'T6', 'Q4', or 'Q9'.")
            else:
                if type == 'T3':
                    if numElementNodes != 3:
                        raise ValueError("Invalid number of nodes for T3 triangle element: Must be 3")
                elif type == 'T6':
                    if numElementNodes != 6:
                        raise ValueError("Invalid number of nodes for T6 triangle element: Must be 6")
                elif type == 'Q4':
                    if numElementNodes != 4:
                        raise ValueError("Invalid number of nodes for Q4 quadrilateral element: Must be 4")
                elif type == 'Q9':
                    if numElementNodes != 9:
                        raise ValueError("Invalid number of nodes for Q9 quadrilateral element: Must be 9")
                else:
                    raise ValueError(f"Unknown element type: {type}. Must be 'triangle' or 'quad'.")


        
        thickness = data.get("thickness", 0.1)  # Read thickness from the input file, default to 0.1 if not provided
        self.ep = [1, thickness]

        E = 2.1e+11
        nu = 0.3

        if "Material" in data:
            material = data["Material"]
            E  = float(material.get("E", E))
            nu = float(material.get("nu", nu))
        else:
            E  = float(data.get("E", E))
            nu = float(data.get("nu", nu))

        self.Dmat = np.array([
                [ 1.0,  nu,  0.],
                [  nu, 1.0,  0.],
                [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

        numElements = len(self.elnods)

        # Make the element dof mapping for each element
        for iel in range(len(self.elnods)):
            dofs = []
            for inod in range(len(self.elnods[iel])):
                dofs.append(self.elnods[iel][inod] * 2) # The x dofs
                dofs.append(self.elnods[iel][inod] * 2 + 1)  # The y dofs

            self.eldofs.append(dofs)



