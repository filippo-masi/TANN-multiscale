'''
Created on May 24, 2022
@author: Ioannis Stefanou & Filippo Masi
'''

import os # allows for easier handling of paths
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed 
# = '0' all messages are logged (default behavior)
# = '1' INFO messages are not printed
# = '3' INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf

from dolfin import *
import numpy as np
from ngeoAI.AI_material import AIUserMaterial3D # import material library based on Neural Network models
from ngeoFE.feproblem import UserFEproblem, General_FEproblem_properties
from ngeoFE.fedefinitions import FEformulation
import pickle # allows export of the results 
material_folder = "../../../numerical_geolab_materials/UMATERIALS/TANN_material" # folder of the saved Neural Network
reference_data_path = "./results/"


### Decomment to deactivate QuadratureRepresentationDeprecationWarning prints
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# Define FE model dimensions (x,y,z)
lx, ly, lz = 10., 10., 20.
# Define number of FE (x,y,z)
nx, ny, nz = 12, 12, 24

class AI3DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=6
        # Number of Gauss points
        self.ns=1

    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector
        """
        gde=[
            Dx(v[0],0),                 #gamma_11
            Dx(v[1],1),                 #gamma_22
            Dx(v[2],2),                 #gamma_33
            Dx(v[1],2)+Dx(v[2],1),      #gamma_23
            Dx(v[0],2)+Dx(v[2],0),      #gamma_13
            Dx(v[0],1)+Dx(v[1],0),      #gamma_12
            ]
        return as_vector(gde)
    
    def create_element(self,cell):
        """
        Set desired element
        """
        # Defines a Lagrangian FE of degree 1 for the displacements
        element_disp=VectorElement("Lagrange",cell,degree=1,dim=3)
        return element_disp  


class top(SubDomain):
    def inside(self,x,on_boundary):
        return abs(x[2] - lz)<=DOLFIN_EPS_LARGE
  
class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[2] - 0.)<=DOLFIN_EPS_LARGE
    
        
class FEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation):
        self.description="Example of 3D FE application with TANN"
        # Attributes: FE model dimensions (dir: x,y,z)
        self.lx, self.ly, self.lz = lx, ly, lz
        # Attributes: number of FE (dir: x,y,z)
        self.nx, self.ny, self.nz = nx, ny, nz
        
        super().__init__(FEformulation)  
       
    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        self.genprops = General_FEproblem_properties()
        # Number of internal state variables of the thermodynamic formulation
        self.p_nIsvars = 22
        # Total number of state variables
        self.genprops.p_nsvars = 14 + self.p_nIsvars
    
    def create_mesh(self):
        """
        Set mesh and subdomains 
        """
        # Generate mesh       
        mesh=BoxMesh(Point(-.5*self.lx,-.5*self.ly,0*self.lz),
                     Point(.5*self.lx,.5*self.ly,1.*self.lz),
                     self.nx,self.ny,self.nz)
        cd=MeshFunction("size_t", mesh, mesh.topology().dim())
        fd=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh,cd,fd
    

    
    def create_subdomains(self,mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) 
        return subdomains

    def mark_boundaries(self,boundaries):
        """
        Mark boundaries
        """
        boundaries.set_all(0)
        fc={"top":1,
            "bottom":2,
            "left":3,
            "right":4,
            "back":5,
            "front":6,} 
        
        self.fc=fc
        
        top0=top()
        top0.mark(boundaries,fc["top"])
        bottom0=bottom()
        bottom0.mark(boundaries,fc["bottom"])
        return    
    
    
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """
        BC_DC=self.BCtype["DC"]
        fc=self.fc
        bcs = [[fc["bottom"], [BC_DC, [0], 0.]],
               [fc["bottom"], [BC_DC, [1], 0.]],
               [fc["bottom"], [BC_DC, [2], 0.]]
               ]
        theta = np.radians(30.) # torsional angle
        bcs.extend(self.set_torsion_bcs(theta,[0,0,self.lz])) # torsional bcs
        return bcs
    
    def set_torsion_bcs(self,theta,center):
        """
        Set displacement boundary conditions under torsion of angle = theta and center of rotation = center
        
        :param theta: rotation angle (radians) - input
        :type double
        :param center: center of rotation - input
        :type numpy array
        
        :return: boundary conditions for torsional displacement
        :rtype: list
        """
        hv = center[2]
        coords = self.mesh.coordinates()
        coords = coords[abs(coords[:,2]-hv)<DOLFIN_EPS_LARGE]
        
        bcs=[]
        for pt in coords:
            pt[0:1]=pt[0:1]-center[0:1]
            u0 = ((pt[0])*np.cos(theta)-(pt[1])*np.sin(theta))-pt[0]
            u1 = ((pt[0])*np.sin(theta)+(pt[1])*np.cos(theta))-pt[1]
            
            bc=[[[100,[pt[0],pt[1],pt[2]]]   ,   [self.BCtype["DC"],[0],u0] ],
                [[100,[pt[0],pt[1],pt[2]]]   ,   [self.BCtype["DC"],[1],u1]],
                ]
            bcs.extend(bc)
        return bcs
    
    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats=[]
        annmat = AIUserMaterial3D(material_folder,self.p_nIsvars)
        mats.append(annmat)
        return mats
    

    
FEformulation = AI3DFEformulation()
FEproblem = FEproblem(FEformulation)
FEproblem.slv.tmax=1.
FEproblem.slv.convergence_tol=10**-6
saveto=reference_data_path+"AI_model_results.xdmf"
converged=FEproblem.solve(saveto,silent=False,summary=False)
print("Analysis completed")


# Export results as numpy arrays
uN = FEproblem.feobj.usol.vector().get_local()
stressGP = np.reshape(FEproblem.feobj.sigma2.vector().get_local(),(-1,FEproblem.feobj.p_nstr))
svarsGP = np.reshape(FEproblem.feobj.svars2.vector().get_local(),(-1,FEproblem.feobj.p_nsvars))
outputs = [uN,stressGP,svarsGP]

w_in_file = reference_data_path+"AI_results"
with open(w_in_file, 'wb') as f_obj:
    pickle.dump(outputs, f_obj)
    
w_in_file = reference_data_path+"AI_unit_test"
with open(w_in_file, 'wb') as f_obj:
    pickle.dump(np.sum(svarsGP[:,-2],axis=0), f_obj)
    
print(lx*ly*lz*np.sum(svarsGP[:,-2],axis=0)/(6*nx*ny*nz))


