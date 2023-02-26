from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve as scsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ss_el_pl( t, y, strainIncr, params, global_tol, tol_strain, tol_stress):
    YM, k, H = params
    
    if yieldf(y[1], k)>global_tol*tol_stress and y[3]>=-global_tol:
        g = strainIncr * np.array([ [(YM*H)/(YM+H)], [0.], [YM/(YM+H)], [YM/((YM+H)*np.sign(y[1]))]])
    else:
        g = strainIncr * np.array([ [YM],            [YM], [0.],        [0.]])
    return g.flatten()

def yieldf(chi, k):
    return np.absolute(chi) - k

def jacobian(statetdt,params,global_tol,tol_strain,cd):
    if statetdt[3]>=global_tol*tol_strain: 
        return np.array([(1.-cd)*params[0]*params[2]/(params[0]+params[2])+cd*params[0]])
    else: 
        return np.array(params[0])
def Tensor_to_Voigt(tensor):
     '''vector=[a11,a22,a33,a23,a32,a13,a31,a12,a21]'''
     vector=np.asarray([tensor[0,0],tensor[1,1],tensor[2,2],
                        tensor[1,2],tensor[0,2],tensor[0,1]],
                        dtype=np.float64)
     return vector

def hyper_increment(DEpsilon, state, params):
    pl=False
    y0, t0 = state, 0.
    
    tol_strain = 1.e+3; tol_stress = 1.e-3; global_tol = 1.e-6
    cd = 0.5
    statetdt = solve_ivp(fun=lambda t, y: ss_el_pl(t,y,DEpsilon,params,global_tol,tol_strain,tol_stress),
                         t_span=[0.,1.],
                         y0=y0,
                         method='RK45',
                         rtol=1.e-8,
                         atol=global_tol*np.array([tol_stress,tol_stress,tol_strain,tol_strain])
                        )
    
    statetdt = statetdt.y[:,-1:].flatten()
    ddsdde = jacobian(statetdt,params,global_tol,tol_strain,cd)
    return statetdt, ddsdde


class build_lattice:
    """
    Constructor
    """
    def __init__(self,nx,ny,nz,s):
        self.colors = ['red', 'darkblue', 'darkorange']
        self.nx,self.ny,self.nz = nx,ny,nz
        self.s = s
        return
    
    def nop(self,i,j,k):
        return j+i*self.ny+k*self.nx*self.ny
    
    def build(self,xmax,ymax,zmax):
        dhx=0.;dhy=0.;dhz=0.
        if self.nx>1: dhx=xmax/(self.nx-1)
        if self.ny>1: dhy=ymax/(self.ny-1)
        if self.nz>1: dhz=zmax/(self.nz-1)
        
        coorM=np.zeros((self.nx*self.ny*self.nz,3))
        nels=(self.nx*self.ny*self.nz)**2
        
        conneM=np.zeros((nels,2),dtype=np.uint32)
        
        el=0
        perbc=[]
        for k in range(self.nz):
            for i in range(self.nx):
                for j in range(self.ny):
                    # periodic connectivity
                    if i==self.nx-1 and j!=self.ny-1: perbc.append([self.nop(i,j,k),self.nop(i-self.nx+1,j,k)])
                    if j==self.ny-1: perbc.append([self.nop(i,j,k),self.nop(i,j-self.ny+1,k)])
                    if k==self.nz-1 and i!=self.nx-1 and j!=self.ny-1: perbc.append([self.nop(i,j,k),self.nop(i,j,k-self.nz+1)])
                    
                    # nodes'coordinates
                    if i!=0 and j!=0 and i!=self.nx-1 and j!=self.ny-1: 
                        perturbation=np.array([np.random.uniform(-self.s,self.s)*dhx,np.random.uniform(-self.s,self.s)*dhy,0.])
                    else:
                        perturbation=np.zeros(3)
                    coorM[self.nop(i,j,k)]=np.array([i*dhx,j*dhy,k*dhz])+perturbation
                    if  i==self.nx-1: coorM[self.nop(i,j,k)]=coorM[self.nop(0,j,k)]+np.array([dhx*(self.nx-1),0,0])
                    if  j==self.ny-1: coorM[self.nop(i,j,k)]=coorM[self.nop(i,0,k)]+np.array([0,dhy*(self.ny-1),0])
                    if  k==self.nz-1: coorM[self.nop(i,j,k)]=coorM[self.nop(i,j,0)]+np.array([0,0,dhz*(self.nz-1)])
                    
                    # nodes' connectivity - elements
                    if i<self.nx-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i+1,j,k)]); el+=1
                    if j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i,j+1,k)]); el+=1
                    if i<self.nx-1 and j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i+1,j+1,k)]); el+=1 
                    if i>0 and j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i-1,j+1,k)]); el+=1   
                    if k>0:
                        conneM[el]=np.array([self.nop(i,j,k),self.nop(i,j,k-1)]); el+=1
                        if j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i,j+1,k-1)]); el+=1   
                        if j>0: conneM[el]=np.array([self.nop(i,j,k),self.nop(i,j-1,k-1)]); el+=1   
                        if i<self.nx-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i+1,j,k-1)]); el+=1
                        if i>0: conneM[el]=np.array([self.nop(i,j,k),self.nop(i-1,j,k-1)]); el+=1
                        if i>0 and j>0: conneM[el]=np.array([self.nop(i,j,k),self.nop(i-1,j-1,k-1)]); el+=1
                        if i>0 and j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i-1,j+1,k-1)]); el+=1
                        if i<self.nx-1 and j<self.ny-1: conneM[el]=np.array([self.nop(i,j,k),self.nop(i+1,j+1,k-1)]); el+=1
                        if i<self.nx-1 and j>0: conneM[el]=np.array([self.nop(i,j,k),self.nop(i+1,j-1,k-1)]); el+=1          
                        
        conneM=conneM[:el]
        groupM=np.zeros((el),dtype=np.uint32)
        toleps=1e-4
        e1=np.array([1.,0,0]);e2=np.array([0,1.,0]);e3=np.array([0,0,1.])
        for i in range(el):    
            a,b=coorM[conneM[i]]
            ab=b+a; ab=ab/np.linalg.norm(ab)    
            if np.abs(a[2]-b[2])<=toleps and (np.abs(a[2]-0)<=toleps or np.abs(a[2]-zmax)<=toleps):
                groupM[i]=1
            if np.abs(a[1]-b[1])<=toleps and (np.abs(a[1]-0)<=toleps or np.abs(a[1]-ymax)<=toleps):
                groupM[i]=1
            if np.abs(a[0]-b[0])<=toleps and (np.abs(a[0]-0)<=toleps or np.abs(a[0]-xmax)<=toleps):
                groupM[i]=1
            if ((np.abs(a[0]-b[0])<=toleps and (np.abs(a[0]-0)<=toleps or np.abs(a[0]-xmax)<=toleps)) and 
               (np.abs(a[1]-b[1])<=toleps and (np.abs(a[1]-0)<=toleps or np.abs(a[1]-ymax)<=toleps)) ):
                groupM[i]=2
            if ((np.abs(a[0]-b[0])<=toleps and (np.abs(a[0]-0)<=toleps or np.abs(a[0]-xmax)<=toleps)) and 
               (np.abs(a[2]-b[2])<=toleps and (np.abs(a[2]-0)<=toleps or np.abs(a[2]-zmax)<=toleps))) :
                groupM[i]=2
            if ((np.abs(a[1]-b[1])<=toleps and (np.abs(a[1]-0)<=toleps or np.abs(a[1]-ymax)<=toleps)) and
               (np.abs(a[2]-b[2])<=toleps and (np.abs(a[2]-0)<=toleps or np.abs(a[2]-zmax)<=toleps))) :
                groupM[i]=2
        
        print("number of elements",el)
        print("number of periodic constraints",len(perbc))
        perbc0=None
        
        self.plot_elems(conneM,coorM,groupM,True)
        return coorM,conneM,groupM,perbc
    
    def plot_elems(self,cooneM,coor,group,plot3D=False):
        if plot3D==True: 
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            fig, ax = plt.subplots()
        for i in range(cooneM.shape[0]):
            cons=cooneM[i]
            p1 = coor[cons[0]]; p2 = coor[cons[1]]
            if plot3D==True:
                color=self.colors[group[i]]
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color=color,linewidth=1.5, marker='o',markersize=0)
                ax.set_zlabel("z")
            else:
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],'r-',linewidth=1, marker='o',markersize=5, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()
        return
    
    def plot_nodes(self,coor):
        plt.plot(coor[:,0],coor[:,1],color='red',linewidth=0, marker='o',markersize=3, alpha=0.7)
        plt.show()
        return

    def plot_sol(self,cooneM,coor,solU,show_old=True,plot3D=False):
        if plot3D==True: 
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            fig, ax = plt.subplots()
        coorp=coor+solU
        
        for cons in cooneM:
            if show_old==True:
                p1 = coor[cons[0]]
                p2 = coor[cons[1]]
                if plot3D==True:
                    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'r-',linewidth=1, marker='o',markersize=5)
                    ax.set_zlabel("z")
                else:
                    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],'r-',linewidth=1, marker='o',markersize=5, alpha=0.7)
            
            p1 = coorp[cons[0]]; p2 = coorp[cons[1]]
            if plot3D==True:
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'b-',linewidth=1, marker='o',markersize=5)
                ax.set_zlabel("z")
            else:
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],'b-',linewidth=1, marker='o',markersize=5, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()
        return
    
    def Voigt_to_Tensor(self,vector, strain=False):
        mult=1.
        if strain==True: mult=.5
        tensor=np.asarray([vector[0],       mult*vector[5], mult*vector[4],
                            mult*vector[5],  vector[1],      mult*vector[3],
                            mult*vector[4],  mult*vector[3], vector[2]],
                            dtype=np.float64)
        tensor=tensor.reshape((-1,3))
        return tensor


class lattice_element:
    """
    Class for one lattice element (bar)
    """
    def __init__(self,connection,coordinates,iid,param):
        self.coordA=coordinates[0];self.coordB=coordinates[1]
        self.nodeA=connection[0];self.nodeB=connection[1]
        dx=self.coordA-self.coordB
        self.L=np.sum(dx**2)**.5
        self.nv=dx/self.L
        a=np.expand_dims(self.nv, axis=0)
        self.nvinvj=np.matmul(a.transpose(),a)
        self.id=iid
        self.param=param[:3]
        self.S = param[3]

    def elements_jac_rhs(self,dU,svars):
        deps=self.get_eps(dU)
        force_t=svars[0]
        force_tdt,svars_tdt,ddsdde=self.material_increment(deps,svars)
        return (force_tdt-force_t)*self.S*self.nv, ddsdde*self.S*self.nvinvj/self.L, svars_tdt
    
    def get_eps(self,dU):
        dUaxial=np.dot(dU,self.nv)
        deps=dUaxial/self.L
        return deps    
    
    def material_increment(self,deps,svars):
        svarstdt = svars.copy()
        alphat = svarstdt[2].copy()
        
        sol,ddsdde=hyper_increment(deps,svarstdt[:4],self.param)
        sigmatdt,Xtdt,alphatdt,l=sol
        svarstdt[0] = sigmatdt
        svarstdt[1] = Xtdt
        svarstdt[2] = alphatdt
        epsilontdt = deps+svarstdt[4]
        svarstdt[4] = epsilontdt
        svarstdt[5] = self.S*self.L*(0.5*self.param[0]*(epsilontdt-alphatdt)**2+0.5*self.param[2]*alphatdt**2)
        svarstdt[6] = self.S*self.param[1]*self.L*np.abs(alphatdt-alphat)
        svarstdt[7] = self.L
        return sigmatdt, svarstdt, ddsdde
   
    
class lattice:
    """
    Class for lattice assembly
    """
    def __init__(self,coordinates,connectivity,param,group,nsvars,sparse=False,perbc=None):
        self.connectivity=connectivity
        self.coords=coordinates
        self.ndim=3
        self.nnodes=coordinates.shape[0]
        self.ndofs=self.ndim*self.nnodes
        self.param=param
        self.group=group
        self.elements=self.generate_elements()
        self.nelements=len(self.elements)
        self.sparse=sparse
        self.nsvars=nsvars
        self.svars=np.zeros((len(self.elements),nsvars))
        self.perbc=perbc
        self.nperbc=0
        self.nbars = self.elements.shape[0]
        
        
        if self.perbc!=None:
            self.nperbc=self.ndim*len(self.perbc)
            self.jacperbc_Klu=self.get_jacperbc_matrices()
        self.Uglobal=np.zeros(( (self.ndofs+self.nperbc)//self.ndim , self.ndim ) )
    
    def get_jacperbc_matrices(self):
        Klu=np.zeros((self.nperbc,self.ndofs))
        n=self.ndim
        for i in range(self.nperbc//self.ndim):
            j=self.perbc[i][0]
            Klu[n*i:n*(i+1),n*j:n*(j+1)] = np.eye(self.ndim)
            j=self.perbc[i][1]
            Klu[n*i:n*(i+1),n*j:n*(j+1)] =-np.eye(self.ndim)
        return Klu
                
    def generate_elements(self):
        elements=np.array([lattice_element(self.connectivity[i],self.coords[self.connectivity[i]],i,self.param[self.group[i]]) 
                           for i in range(self.connectivity.shape[0])])
        return elements
    
    def assemble_global_jac_rhs(self,dUglobal,svars,BCs,ZeroDC=True):
        global_svars=np.zeros((self.nelements,self.nsvars))
        global_rhs=self.initialize_matrix(1,self.ndofs+self.nperbc)
        global_jac=self.initialize_matrix(self.ndofs+self.nperbc,self.ndofs+self.nperbc)
        ndim=self.ndim
        for el in self.elements:
            dUelement=dUglobal[el.nodeA]-dUglobal[el.nodeB]
            svarst_elem=svars[el.id]
            dforcetdt, jactdt, global_svars[el.id] = el.elements_jac_rhs(dUelement,svarst_elem)
            #rhs
            global_rhs[0,ndim*el.nodeA:(ndim*(el.nodeA+1))] +=  dforcetdt
            global_rhs[0,ndim*el.nodeB:(ndim*(el.nodeB+1))] += -dforcetdt
            #jac
            global_jac[ndim*el.nodeA:ndim*(el.nodeA+1),ndim*el.nodeA:ndim*(el.nodeA+1)] +=  jactdt
            global_jac[ndim*el.nodeA:ndim*(el.nodeA+1),ndim*el.nodeB:ndim*(el.nodeB+1)] +=  -jactdt
            global_jac[ndim*el.nodeB:ndim*(el.nodeB+1),ndim*el.nodeA:ndim*(el.nodeA+1)] +=  -jactdt
            global_jac[ndim*el.nodeB:ndim*(el.nodeB+1),ndim*el.nodeB:ndim*(el.nodeB+1)] +=  jactdt
         
        #set perjac&rhs
        if self.perbc!=None:
            global_jac[:self.ndofs,-self.nperbc:]=self.jacperbc_Klu.T
            global_jac[-self.nperbc:,:self.ndofs]=self.jacperbc_Klu
            global_rhs[0,self.ndofs:]+=self.jacperbc_Klu @ dUglobal.flatten()[:self.ndofs]
            global_rhs[0,:self.ndofs]+=self.jacperbc_Klu.T @ dUglobal.flatten()[-self.nperbc:]
        
        #set BCs
        global_jac, global_rhs = self.set_BCs(global_jac, global_rhs,BCs,ZeroDC)
       
        return global_jac, global_rhs, global_svars 
    
    def set_BCs(self,gjac,grhs,BCs,ZeroDC=True):
        flag=1.
        for bc in BCs:
            if bc[2]=="DC":
                if ZeroDC==True: flag=0.
                grhs[0,bc[0]]=-bc[1]*flag
                gjac[bc[0],:]=0
                gjac[bc[0],bc[0]]=1.
            elif bc[2]=="NM":
                grhs[0,bc[0]]+=-bc[1]
            elif bc[2]=="PR":
                Fl=np.zeros(self.nperbc)
                for i in range(self.nperbc//self.ndim):
                    DFij=np.array(bc[1])
                    iB=self.perbc[i][0];iA=self.perbc[i][1]
                    DX=np.array(self.coords[iB]-self.coords[iA]).T
                    Fl[self.ndim*i:self.ndim*(i+1)]=DFij@DX
                grhs[0,-self.nperbc:]+=-Fl
        return gjac, grhs 
        
    def initialize_matrix(self,n,m):
        if self.sparse:
            return sp.lil_matrix((n,m))
        else:
            return np.zeros((n,m))
    
    def get_avg_stress(self):
        if self.perbc==None:
            print("This function is only supported for periodic boundary conditions.")
            return np.zeros((self.ndim,self.ndim))
        else:
            sigmaij=0
            for i in range(self.nperbc//self.ndim):
                ti=self.Uglobal[self.ndofs//self.ndim+i]
                iB=self.perbc[i][1]
                iA=self.perbc[i][0]
                xi=self.coords[iB]-self.coords[iA]
                sigmaij+=np.tensordot(ti,xi,axes=0)
            return sigmaij



class solver:
    
    def __init__(self,lattice):
        self.rhstol=1.e-4
        self.dutol=1.e-4
        self.maxiter=50
        self.lattice=lattice
    
    def solve(self,BCs,svarsM,add_svarsM):
        DU,svars,success,niter=self.do_increment(BCs)
        if success==True:
            self.lattice.Uglobal+=DU
            self.lattice.svars=svars.copy()
            if self.lattice.perbc!=None:
                svarsM[:6] = Tensor_to_Voigt(self.lattice.get_avg_stress())
                svarsM[12:12+self.lattice.nbars] = self.lattice.svars[:,0] # micro-stresses
                svarsM[12+self.lattice.nbars:12+2*self.lattice.nbars] = self.lattice.svars[:,4] # micro-strain
                svarsM[12+2*self.lattice.nbars:12+3*self.lattice.nbars] = self.lattice.svars[:,2] # micro additional-z
                svarsM[12+3*self.lattice.nbars:-3] = self.lattice.svars[:,7] # lengths/elngations
                svarsM[-3] = np.sum(self.lattice.svars[:,5])/svarsM[-1] # total energy density
                svarsM[-2] = np.sum(self.lattice.svars[:,6])/svarsM[-1] # total dissipation rate density
                add_svarsM = self.lattice.Uglobal[:self.lattice.ndofs//self.lattice.ndim].copy()
        return success,svarsM,add_svarsM
    

    def do_increment(self,BCs):
        success=True
        niter=0
        
        #k indicates increments, i iterations
        DU_k1i1=np.zeros(self.lattice.ndofs+self.lattice.nperbc)
        DU_k1i=np.zeros(self.lattice.ndofs+self.lattice.nperbc)
        svars_k1i1=self.lattice.svars.copy()
        svars_k1i=svars_k1i1.copy()
        gjac_k1i1,grhs_k1i1,svars_k1i1=self.lattice.assemble_global_jac_rhs(
                                DU_k1i.reshape(
                                    (self.lattice.ndofs+self.lattice.nperbc)//self.lattice.ndim,
                                    self.lattice.ndim),
                                svars_k1i, BCs,ZeroDC=False)
        rhsnormki0=1.#np.linalg.norm(grhs_k1i1)
        #d indicates iterations, D increment
        dDU_k1i1 = scsolve(gjac_k1i1, grhs_k1i1.squeeze())
        DU_k1i1-=dDU_k1i1
        while True:
            niter+=1
            DU_k1i=DU_k1i1.copy()
            grhs_k1i=grhs_k1i1.copy()
            gjac_k1i1,grhs_k1i1,svars_k1i1=self.lattice.assemble_global_jac_rhs(
                                DU_k1i.reshape(
                                    (self.lattice.ndofs+self.lattice.nperbc)//self.lattice.ndim,
                                    self.lattice.ndim),
                                svars_k1i, BCs)
            
            rhsnormk1=np.linalg.norm(grhs_k1i1)
            dDU_k1i1 = scsolve(gjac_k1i1, grhs_k1i1.squeeze())
            DU_k1i1-=dDU_k1i1
            if np.linalg.norm(dDU_k1i1)<=self.dutol and rhsnormk1<=self.rhstol*rhsnormki0: break
            else:
                if niter>self.maxiter:
                    print("Maximum iterations reached.")
                    success=False
                    break
        return DU_k1i1.reshape((self.lattice.ndofs+self.lattice.nperbc)//self.lattice.ndim,
                                self.lattice.ndim),svars_k1i1,success,niter     
