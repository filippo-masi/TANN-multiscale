'''
@author: Filippo Masi and Ioannis Stefanou
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = (3.5,3.5)

from lattice_class.lattice_material import build_lattice,lattice_element,lattice, solver

xmax=10;ymax=10;zmax=20 # box's dimensions
V = xmax*ymax*zmax # volume
nx = 5
ny = nx
nz = (nx-1)*2+1

s=0. # magnitude of the perturbation (uniform spatial distribution) of the nodal coordinates

constructor = build_lattice(nx,ny,nz,s)
coorM,conneM,groupM,perbc = constructor.build(xmax,ymax,zmax)



###########################################################
#                      Set BCs, initialize                #
###########################################################

Lcell= zmax/(nz-1)
param=[[1.,1./3.,0.1,Lcell**2],
       [1.,1./3.,0.1,Lcell**2/2.],
       [1.,1./3.,0.1,Lcell**2/4.]]

coorM[:,0]=coorM[:,0]-xmax/2
coorM[:,1]=coorM[:,1]-ymax/2
mylattice=lattice(coorM,conneM,param,groupM,nsvars=8,sparse=False,perbc=None) 

ini_lattice_coords = mylattice.coords
nparticles = mylattice.ndofs//mylattice.ndim
nbars = mylattice.elements.shape[0]
nsvars = 4*nbars+2*6+1+1+1

ini_lattice_coords = mylattice.coords
T = np.radians(30.)
D = 0
DCBC=[]
for i in range(ini_lattice_coords.shape[0]):
    if ini_lattice_coords[i,2]==0.:
        DCBC.append([3*i+0,0.,"DC"])
        DCBC.append([3*i+1,0.,"DC"])
        DCBC.append([3*i+2,0.,"DC"])
    if ini_lattice_coords[i,2]==zmax:
        r = np.sqrt((xmax/2-ini_lattice_coords[i,0])**2+(ymax/2-ini_lattice_coords[i,1])**2)
        xx = ini_lattice_coords[i,0]; yy = ini_lattice_coords[i,1]
        ux = ((xx)*np.cos(T)-(yy)*np.sin(T))-xx
        uy = ((xx)*np.sin(T)+(yy)*np.cos(T))-yy
        DCBC.append([3*i+0,ux,"DC"])
        DCBC.append([3*i+1,uy,"DC"])
        
svarsGP = np.zeros(nsvars)
add_svarsGP = (mylattice.coords)
svarsGP[-1] = V

###########################################################
#                      Solution                           #
###########################################################

print("Solving")
slv=solver(mylattice)
res,svarsGP,add_svarsGP = slv.solve(DCBC,svarsGP,add_svarsGP)

# Plot deformed and reference configurations
constructor.plot_sol(mylattice.connectivity,mylattice.coords,mylattice.Uglobal[:mylattice.ndofs//mylattice.ndim],plot3D=True,show_old=True)   
