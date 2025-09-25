import numpy as np
import torch

from network import *
from GNAR import *


def fourier_basis(x,n,P):
    return np.exp(1j*n*x*2*np.pi/P)

def cos_basis(x,n,P):
    return np.cos(x*2*np.pi*n/P)

def sin_basis(x,n,P):
    return np.sin(x*2*np.pi*n/P)
    
def haar_mother(x):
    return np.select([(0<=x)&(x<0.5),(0.5<=x)&(x<1)],[1,-1],0)

def haar_basis(x,n,k,T): 
    # n scale, k shift
    return 2**(n/2)*haar_mother(2**n*x/T-k)
    
from scipy.interpolate import interp1d

class DB:
    def __init__(self,vanishing_moments,T,T_ceil=None,j0=None):
        self.T = T
        if T_ceil is None:
            self.T_ceil = T
        else:
            self.T_ceil = T_ceil
        self.vanishing_moments = vanishing_moments
        self.support = 2*self.vanishing_moments
        match vanishing_moments:
            case 2:
                self.a = np.array([(1+np.sqrt(3))/(4*np.sqrt(2)), (np.sqrt(3)+3)/(4*np.sqrt(2)), (3-np.sqrt(3))/(4*np.sqrt(2)), (1-np.sqrt(3))/(4*np.sqrt(2))]).reshape(1,-1,1)
                self.b = np.array([(1-np.sqrt(3))/(4*np.sqrt(2)), (np.sqrt(3)-3)/(4*np.sqrt(2)), (3+np.sqrt(3))/(4*np.sqrt(2)), (-1-np.sqrt(3))/(4*np.sqrt(2))]).reshape(1,-1,1)
            case 4:
                self.a = (np.array([0.32580343,1.01094572,0.89220014,-0.03957503,-0.26450717,0.0436163,0.0465036,-0.01498699])/np.sqrt(2)).reshape(1,-1,1)
                self.b = (np.array([-0.01498699,-0.0465036,0.0436163,0.26450717,-0.03957503,-0.89220014,1.01094572,-0.32580343])/np.sqrt(2)).reshape(1,-1,1)
            case 8:
                self.a = np.array([0.05441584224308161,0.3128715909144659,0.6756307362980128,0.5853546836548691,-0.015829105256023893,-0.2840155429624281,0.00047248457399797254,0.128747426620186,
                                  -0.01736930100202211,-0.04408825393106472,0.013981027917015516,0.008746094047015655,-0.00487035299301066,-0.0003917403729959771,0.0006754494059985568,-0.00011747678400228192]).reshape(1,-1,1)
                self.b = np.array([-0.00011747678400228192,-0.0006754494059985568,-0.0003917403729959771,0.00487035299301066,0.008746094047015655,-0.013981027917015516,-0.04408825393106472,
                                  0.01736930100202211,0.128747426620186,-0.00047248457399797254,-0.2840155429624281,0.015829105256023893,0.5853546836548691,-0.6756307362980128,0.3128715909144659,-0.05441584224308161]).reshape(1,-1,1)
                
        if j0 is None:
            self.j0 = int(np.ceil(np.log2(self.vanishing_moments)))+1
        else:
            self.j0 = j0
        self.Vs = [np.identity(self.T_ceil)] #father
        self.Ws = []#mother
        for i in range(int( np.log2(self.T_ceil)- self.j0)):
            self.Ws.append(self.generate_next(self.Vs[-1],self.b))
            self.Vs.append(self.generate_next(self.Vs[-1],self.a)) # The last V has 2^j0 basis 
        self.basis = np.concatenate((self.Ws+self.Vs[-1:]),axis=1)
        if self.T != self.T_ceil:
            self.basis = interp1d(np.arange(1,1+self.T_ceil),self.basis,axis=0)(np.linspace(1,self.T_ceil,self.T))

    def generate_next(self,V,coefs):
        N = V.shape[1]
        V_exp = np.zeros((self.T_ceil,int(N/2),self.support))
        for i in range(2):
            V_exp[:,:,i] = V[:,i::2]
        for i in range(2,self.support):
            V_exp[:,:,i] = np.concatenate((V[:,i::2],V[:,i%2:i-1:2]),axis=1)
        return (V_exp@coefs)[:,:,0]
        
class DB2(DB):
    def __init__(self,T,T_ceil=None):
        super().__init__(2,T,T_ceil)

class DB4(DB):
    def __init__(self,T,T_ceil=None):
        super().__init__(4,T,T_ceil)

class DB8(DB):
    def __init__(self,T,T_ceil=None):
        super().__init__(8,T,T_ceil)
    
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def hard_thresh(x, a):
    return np.where(np.abs(x) > a, x, 0)

def soft_thresh(x, a):
    return np.where(np.abs(x) > a, np.sign(x)*(np.abs(x)-a), 0)
    
class TVGNAR(NetworkModel):
    def __init__(self, alpha_order, beta_order, basis_order=None, basis_order_min= None, intercept=True, global_intercept=False, tv_intercept=True, family = "haar",gpu=False):
        super().__init__(alpha_order, beta_order, intercept, global_intercept)
        self.basis_order = basis_order
        self.basis_order_min = basis_order_min
        self.tv_intercept = tv_intercept
        self.family = family
        self.gpu = gpu
        self.num_coefs = sum(self.alpha_orders)+sum(self.beta_order)

    def generate_basis_fourier(self):
        t = np.arange(1+self.alpha_order,self.T+1)
        self.cos_basis = cos_basis(t.reshape(-1,1),np.arange(1,self.basis_order+1).reshape(1,-1),self.T-self.alpha_order)
        self.sin_basis = sin_basis(t.reshape(-1,1),np.arange(1,self.basis_order+1).reshape(1,-1),self.T-self.alpha_order)
        return np.concatenate([np.ones((t.shape[0],1)),self.cos_basis,self.sin_basis],axis=1)

    def generate_basis_haar(self):
        t= np.arange(1,self.T+1-self.alpha_order)
        # shape = (len(t),1+sum_n 2^n = 2^(n+1))
        t[-1] -= 1e-10 #prevents the last t to be evaluated to 0
        self.num_father = 1
        return np.concatenate([np.ones((t.shape[0],1))]+[haar_basis(t.reshape(-1,1),n,np.arange(2**n).reshape(1,-1),self.T-self.alpha_order) 
                                                         for n in range(0,self.basis_order+1)],axis=1)#/(self.T-self.alpha_order)

    def generate_basis_db(self,vanishing_moments,basis_order_min):
        db = DB(vanishing_moments, T=self.T-self.alpha_order, T_ceil=int(2**np.ceil(np.log2(self.T-self.alpha_order))), j0=basis_order_min)
        basis = db.basis[:,-2**(1+self.basis_order):][:,::-1]#minimum 2, ::-1 gives a reversed order
        self.basis_order_min = db.j0
        self.num_father = 2**db.j0
        return basis
        
    
    def generate_basis(self):
        if self.family[:2] == "db":
            return self.generate_basis_db(int(self.family[2:]),self.basis_order_min)
        match self.family:
            case "fourier":
                return self.generate_basis_fourier()
            case "haar":
                return self.generate_basis_haar()

    def fit(self,network,vts,thresh="soft",mad_level=1,thresh_const_mult=1):
        self.network = network
        self.vts = vts
        self.vts_end = vts[-self.alpha_order:,:]
        self.T,self.N = vts.shape
        
        if self.basis_order is None:
            self.basis_order = np.floor(np.log2(np.sqrt((self.T-self.alpha_order)*self.N))).astype(int)
            if self.intercept:
                if not self.global_intercept:
                    self.intercept_basis_order = np.floor(np.log2(np.sqrt(self.T-self.alpha_order))).astype(int) #if non-global intercept, resolution of the intercept is lower
                else:
                    self.intercept_basis_order = self.basis_order 
        else:
            self.intercept_basis_order = self.basis_order
            
        self.y = vts[self.alpha_order:,:].flatten("F")
        self.X = self.transformVTS(vts)

        self.basis = self.generate_basis() # shape = (T-p, 1+sum_n 2^n = 2^(n+1))

        if self.intercept:
            if self.tv_intercept:
                if not self.global_intercept:
                    self.intercept_basis = self.basis[:,:2**(1+self.intercept_basis_order)]
                    self.X_basis = np.hstack((
                        (self.X[:,:-self.N,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1),
                        (self.X[:,-self.N:,np.newaxis]*np.tile(self.intercept_basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1)
                    ))
    
                else:
                    self.X_basis = (self.X[:,:,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1) # shape = (N(T-p), 2^(n+1)*#coef)
            else:
                self.X_basis = np.hstack((
                        (self.X[:,:self.num_coefs,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1),
                        self.X[:,self.num_coefs:]))
        else:
            self.X_basis = (self.X[:,:,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1)
        
        if self.gpu:
            X_basis_gpu = torch.Tensor(self.X_basis).cuda()
            y_gpu = torch.Tensor(self.y).cuda()
            lstsq = torch.linalg.lstsq(X_basis_gpu,y_gpu)
            basis_coefs, self.res = lstsq.solution.cpu().numpy(), lstsq.residuals.cpu().numpy()
        else:
            basis_coefs, self.res = np.linalg.lstsq(self.X_basis, self.y, rcond=None)[:2]
            #basis_coefs = np.linalg.solve(self.X_basis.T@self.X_basis, self.X_basis.T@self.y)
        
        if not self.intercept:
            self.basis_coefs = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F") # shape = (number of basis, number of coefficients)
        else:
            if self.tv_intercept:
                if not self.global_intercept:
                    self.basis_coefs = basis_coefs[:-self.intercept_basis.shape[1]*self.N].reshape(self.basis.shape[1], self.num_coefs, order="F")
                    self.intercept_basis_coefs = basis_coefs[-self.intercept_basis.shape[1]*self.N:].reshape(self.intercept_basis.shape[1], self.N, order="F")
                else:
                    basis_coefs = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
                    self.basis_coefs,self.intercept_basis_coefs = basis_coefs[:,:self.num_coefs], basis_coefs[:,self.num_coefs:]
            else:
                self.basis_coefs = basis_coefs[:self.basis.shape[1]*self.num_coefs].reshape(self.basis.shape[1], self.num_coefs, order="F")
                self.intercept_coefs = basis_coefs[self.basis.shape[1]*self.num_coefs:]
        
        if (self.family == "haar" or self.family == "db2" or self.family == "db4" or self.family == "db8") and thresh is not None:
            self.wavelet_thresh(thresh,mad_level,thresh_const_mult)

        self.last_coefs = self.get_coefs()[-1]
        if self.intercept:
            if self.tv_intercept:
                self.last_coefs = np.concatenate((self.last_coefs,self.get_intercept()[-1]))
            else:
                self.last_coefs = np.concatenate((self.last_coefs,self.get_intercept()))
            
    def get_coefs(self):
        return self.basis@self.basis_coefs

    def get_intercept(self):
        if not self.intercept:
            return None
        else:
            if self.tv_intercept:
                if not self.global_intercept:
                    return self.intercept_basis@self.intercept_basis_coefs
                else:
                    return self.basis@self.intercept_basis_coefs
            else:
                return self.intercept_coefs

    def predict(self, length, nodes=None, vts_end =None):
        if vts_end is None:
            vts_end = self.vts_end
        vts_pred = np.zeros((length+self.alpha_order,self.network.size))
        vts_pred[:self.alpha_order,:] = vts_end
        for i in range(length):
            vts_pred[self.alpha_order+i,:] = self.transformVTS(vts_pred[i:self.alpha_order+i+1,:])@self.last_coefs
        if nodes is None:
            return vts_pred[self.alpha_order:,:]
        else:
            return vts_pred[self.alpha_order:,nodes]

    def simulate(self, network, initial_vts, length, coefs, error_cov_mat):
        self.network = network
        l = len(initial_vts)
        vts_sim = np.zeros((length+l,self.network.size))
        vts_sim[:l,:] = initial_vts
        for i in range(length):
            vts_sim[l+i,:] = self.transformVTS(vts_sim[l+i-self.alpha_order:l+i+1,:])@coefs[i,:] + multivariate_normal(
                cov=error_cov_mat).rvs(1)
        return vts_sim[l:]

    def wavelet_thresh(self, thresh, mad_level=1,thresh_const_mult=1):
        index = np.sum([2**(self.basis_order-n) for n in range(mad_level)])
        self.mad = mad(self.basis_coefs[-index:],axis=0)
        self.thresh_const = thresh_const_mult*np.sqrt(2*np.log(self.basis_coefs.shape[0]-self.num_father))
        match thresh:
            case "hard":
                thresh_fun = hard_thresh
            case "soft":
                thresh_fun = soft_thresh
        self.basis_coefs[self.num_father:] = thresh_fun(self.basis_coefs[self.num_father:], self.mad*self.thresh_const)
        if self.intercept and self.tv_intercept:
            index = np.sum([2**(self.intercept_basis_order-n) for n in range(mad_level)])
            self.intercept_basis_coefs[self.num_father:] = thresh_fun(self.intercept_basis_coefs[self.num_father:], mad(self.intercept_basis_coefs[-index:],axis=0)*self.thresh_const)
        
def TVGNAR_sim(network, alpha_order, beta_order, coefs, intercept, global_intercept, error_cov_mat,length,burn_in=100):
    vts = np.zeros((length+burn_in+alpha_order,network.size))
    vts[:alpha_order] = norm(loc=0).rvs(size=(alpha_order,network.size))
    model = TVGNAR(alpha_order, beta_order, basis_order=1,intercept=intercept,global_intercept=global_intercept)
    vts[alpha_order:] = model.simulate(network,vts[:alpha_order],length+burn_in,coefs,error_cov_mat)
    return vts[burn_in+alpha_order:]
    
