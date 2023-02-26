'''
@author: Filippo Masi and Ioannis Stefanou
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D
import time
plt.rcParams["figure.figsize"] = (3.5,3.5)

from lattice_class.lattice_material import build_lattice,lattice_element,lattice, solver





xmax=1;ymax=1;zmax=1 # lattice dimensions
nx, ny, nz = (3,3,2) # number of unit cell
s=0. # magnitude of the perturbation (uniform spatial distribution) of the nodal coordinates

constructor = build_lattice(nx,ny,nz,s)
coorM,conneM,groupM,perbc = constructor.build(xmax,ymax,zmax)



###########################################################
#                      Set BCs, initialize                #
###########################################################


param=[[1.,1./3.,0.,1./((nx-1)*(ny-1))],
       [1.,1./3.,0.,1./2./((nx-1)*(ny-1))],
       [1.,1./3.,0.,1./4./((nx-1)*(ny-1))]]

mylattice=lattice(coorM,conneM,param,groupM,nsvars=8,sparse=False,perbc=perbc) 
ini_lattice_coords = mylattice.coords

nparticles = mylattice.ndofs//mylattice.ndim
nbars = mylattice.elements.shape[0]
nsvars = 4*nbars+2*6+1+1+1
nnodes =  ini_lattice_coords.shape[0]

n1=constructor.nop(nx//2,ny//2,nz//2);
if nz != 1:
    DCBC=[[n1*3+0,0.,"DC"],
          [n1*3+1,0.,"DC"],
          [n1*3+2,0.,"DC"]] 
    
DCBC.append([1,np.zeros((3,3)),"PR"])

###########################################################
#                      Data-generation                    #
###########################################################
increments = 40*3 # number of increments
n_reset= increments # number of increments before reset
       

dep = np.zeros((increments,6)) # strain increments

de = 5.e-2 # magnitude of the strain increments

# Define strain increment path
sub = increments//3
dep[:sub,0] = de* np.ones((sub))
dep[sub:2*sub,0] = -de*np.ones((sub))
dep[2*sub:3*sub,0] = de*np.ones((sub))

# Initialize variables
ep_tdt = np.zeros((increments,6))
svars_tdt=np.zeros((increments, nsvars))
add_svars_tdt=np.zeros((increments,nparticles,3))

outliers=[]

for i in range(increments):
    # Resetting and initializing
    if not i % n_reset:
        mylattice=lattice(coorM,conneM,param,groupM,nsvars=8,sparse=False,perbc=perbc)   
        svarsM = np.zeros(nsvars)
        add_svarsM = np.zeros((nparticles,3))
        svarsM[-1] = 1
    ep_tdt[i] = svarsM[6:12]

    # Apply periodic BCs
    dep_tens = constructor.Voigt_to_Tensor(dep[i],strain=True)
    DCBC[-1][1] = dep_tens
    
    # Solve / do increment
    ep_tdt[i]+= dep[i]
    svarsM[6:12] = ep_tdt[i]
    slv=solver(mylattice); res,svarsM,add_svarsM=slv.solve(DCBC,svarsM,add_svarsM)    
    # Copy svars    
    svars_tdt[i] = svarsM
    add_svars_tdt[i] = add_svarsM
    
    # Check for convergence
    if res==False:
        ep_tdt[i]-=dep[i] 
        outliers=np.append(outliers,i)

print("Increment completed")


# Plot deformed and reference configurations
constructor.plot_sol(mylattice.connectivity,mylattice.coords,mylattice.Uglobal[:mylattice.ndofs//mylattice.ndim],plot3D=True,show_old=True)   

###########################################################
#                      Post-processing                    #
###########################################################

# Extract information @ time t
svars_t=np.array([svars_tdt[i-1] for i in range(0,len(svars_tdt))]); svars_t[0]=0.*svars_t[0]
svars_t[np.arange(0,len(svars_t),n_reset)]=np.zeros(len(svars_tdt[0]))
svars_t[np.arange(0,len(svars_t),n_reset),-1] = 1
add_svars_t=np.array([add_svars_tdt[i-1] for i in range(0,len(add_svars_tdt))]); add_svars_t[0]=0.*add_svars_t[0]
add_svars_t[np.arange(0,len(add_svars_t),n_reset)]=0*ini_lattice_coords

# Extract data @ time t
stress_t=svars_t[:,:6]
strain_t=svars_t[:,6:12]
micro_stress_t=svars_t[:,12:12+nbars]
micro_strain_t=svars_t[:,12+nbars:12+2*nbars]
micro_plstrain_t=svars_t[:,12+2*nbars:12+3*nbars]
micro_l_t=svars_t[:,12+3*nbars:12+4*nbars]
F_t=svars_t[:,-3]
D_t=svars_t[:,-2]
v_t=svars_t[:,-1]
u_t = add_svars_t

# Extract data @ time tdt
stress_tdt=svars_tdt[:,:6]
strain_tdt=svars_tdt[:,6:12]
micro_stress_tdt=svars_tdt[:,12:12+nbars]
micro_strain_tdt=svars_tdt[:,12+nbars:12+2*nbars]
micro_plstrain_tdt=svars_tdt[:,12+2*nbars:12+3*nbars]
micro_l_tdt=svars_tdt[:,12+3*nbars:12+4*nbars]
F_tdt=svars_tdt[:,-3]
D_tdt=svars_tdt[:,-2]
v_tdt=svars_tdt[:,-1]
u_tdt = add_svars_tdt


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (4,3)

plt.plot(strain_t[:,0],stress_t[:,0],color='red', alpha = 1, marker='o', linestyle='-',linewidth=1., markersize=3)
plt.ylabel('$\sigma_{11}$')
plt.xlabel('$\\varepsilon_{11}$')
plt.show()