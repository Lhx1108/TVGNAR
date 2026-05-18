import gc
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from network import *
from GNAR import *

from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, norm


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
    

class DB:
    """
    Builds a Daubechies wavelet basis matrix.

    Supported wavelets: db2, db4, db8

    The class caches previously computed bases so that repeated calls with
    the same parameters are faster.
    """
    pre_interp_basis_cache = {}

    def __init__(self,vanishing_moments,T,T_ceil=None,j0=None):
        """
        Initialize a Daubechies wavelet basis.
        
        Parameters:
        
        vanishing_moments : int
            Number of vanishing moments. Only 2, 4, and 8 are supported.

        T : int
            Desired output length of the basis.

        T_ceil : int, optional
            Length used internally to build the basis.
            If not provided, T_ceil = T.

        j0 : int, optional
            Coarsest scale level.
            If not provided, it is chosen automatically based on
            the number of vanishing moments.

        Attributes:
        
        basis : ndarray
            The generated wavelet basis matrix.

        a : ndarray
            Low-pass filter coefficients.

        b : ndarray
            High-pass filter coefficients.

        Vs : list
            Scaling-function basis matrices at each level.

        Ws : list
            Wavelet basis matrices at each level.
        """
        self.T = T
        if T_ceil is None:
            self.T_ceil = T
        else:
            self.T_ceil = T_ceil
        self.vanishing_moments = vanishing_moments
        self.support = 2*self.vanishing_moments

        if j0 is None:
            self.j0 = int(np.ceil(np.log2(self.vanishing_moments)))+1
        else:
            self.j0 = j0

        cache_key = (self.vanishing_moments,self.T_ceil,self.j0)

        if cache_key in DB.pre_interp_basis_cache:
            self.Vs = None
            self.Ws = None
            self.basis = DB.pre_interp_basis_cache[cache_key].copy()
        else:
            match vanishing_moments:
                case 2:
                    self.a = np.array([
                        (1+np.sqrt(3))/(4*np.sqrt(2)),
                        (np.sqrt(3)+3)/(4*np.sqrt(2)),
                        (3-np.sqrt(3))/(4*np.sqrt(2)),
                        (1-np.sqrt(3))/(4*np.sqrt(2))
                    ],dtype=np.float64).reshape(1,-1,1)
                    self.b = np.array([
                        (1-np.sqrt(3))/(4*np.sqrt(2)),
                        (np.sqrt(3)-3)/(4*np.sqrt(2)),
                        (3+np.sqrt(3))/(4*np.sqrt(2)),
                        (-1-np.sqrt(3))/(4*np.sqrt(2))
                    ],dtype=np.float64).reshape(1,-1,1)
                case 4:
                    self.a = (np.array([
                        0.32580343,1.01094572,0.89220014,-0.03957503,
                        -0.26450717,0.0436163,0.0465036,-0.01498699
                    ],dtype=np.float64)/np.sqrt(2)).reshape(1,-1,1)
                    self.b = (np.array([
                        -0.01498699,-0.0465036,0.0436163,0.26450717,
                        -0.03957503,-0.89220014,1.01094572,-0.32580343
                    ],dtype=np.float64)/np.sqrt(2)).reshape(1,-1,1)
                case 8:
                    self.a = np.array([
                        0.05441584224308161,0.3128715909144659,0.6756307362980128,0.5853546836548691,
                        -0.015829105256023893,-0.2840155429624281,0.00047248457399797254,0.128747426620186,
                        -0.01736930100202211,-0.04408825393106472,0.013981027917015516,0.008746094047015655,
                        -0.00487035299301066,-0.0003917403729959771,0.0006754494059985568,-0.00011747678400228192
                    ],dtype=np.float64).reshape(1,-1,1)
                    self.b = np.array([
                        -0.00011747678400228192,-0.0006754494059985568,-0.0003917403729959771,0.00487035299301066,
                        0.008746094047015655,-0.013981027917015516,-0.04408825393106472,0.01736930100202211,
                        0.128747426620186,-0.00047248457399797254,-0.2840155429624281,0.015829105256023893,
                        0.5853546836548691,-0.6756307362980128,0.3128715909144659,-0.05441584224308161
                    ],dtype=np.float64).reshape(1,-1,1)
                case _:
                    raise ValueError("Only db2, db4, db8 are supported.")

            self.Vs = [np.identity(self.T_ceil,dtype=np.float64)]
            self.Ws = []

            for i in range(int(np.log2(self.T_ceil)-self.j0)):
                self.Ws.append(self.generate_next(self.Vs[-1],self.b))
                self.Vs.append(self.generate_next(self.Vs[-1],self.a))

            self.basis = np.concatenate((self.Ws+self.Vs[-1:]),axis=1)
            self.basis = np.ascontiguousarray(self.basis,dtype=np.float64)

            DB.pre_interp_basis_cache[cache_key] = self.basis.copy()

        if self.T != self.T_ceil:
            self.basis = interp1d(
                np.arange(1,1+self.T_ceil),
                self.basis,
                axis=0
            )(np.linspace(1,self.T_ceil,self.T))

        self.basis = np.ascontiguousarray(self.basis,dtype=np.float64)

    def generate_next(self,V,coefs):
        N = V.shape[1]
        V_exp = np.zeros((self.T_ceil,int(N/2),self.support),dtype=np.float64)
        for i in range(2):
            V_exp[:,:,i] = V[:,i::2]
        for i in range(2,self.support):
            V_exp[:,:,i] = np.concatenate((V[:,i::2],V[:,i%2:i-1:2]),axis=1)
        return (V_exp@coefs)[:,:,0]

    @classmethod
    def clear_cache(cls):
        cls.pre_interp_basis_cache.clear()
        
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
    def __init__(self, alpha_order, beta_order, basis_order=None, intercept_basis_order=None, basis_order_min= None, intercept=True, global_intercept=False, tv_intercept=True, family = "haar",gpu=False,boundary_mode="periodic",boundary_pad=None):
        """
        Initialize a time-varying GNAR model.
    
        Parameters
        ----------
        alpha_order : int
            Max autoregressive lag.
    
        beta_order : list[int]
            Max stage of neighbourhood to include for each lag.
    
        basis_order : int, optional
            Finest wavelet resolution (j^* in the paper) used for fitting.
            If not provided, it is chosen automatically during fitting.
    
        intercept_basis_order : int, optional
            Finest wavelet resolution for the intercept.
            If not provided, it is chosen automatically based on "basis_order".
    
        basis_order_min : int, optional
            Coarest wavelet resolution used for Daubechies wavelets.
            If not provided, it is chosen automatically based on the DB class
    
        intercept : bool, default=True
            Whether to include an intercept term.
    
        global_intercept : bool, default=False
            Whether to use a global intercept across all nodes.
    
        tv_intercept : bool, default=True
            Whether the intercept should be time-varying.
    
        family : str, default="haar"
            Basis family to use. Supported examples include "haar",
            "fourier", "db2", "db4", and "db8".
    
        gpu : bool, default=False
            Whether to use GPU computation with PyTorch.
    
        boundary_mode : {"periodic", "reflect"}, default="periodic"
            Boundary handling method for fitting.
            "reflect" is recommend for forecasting tasks.
    
        boundary_pad : int, optional
            Number of reflected observations to add when using boundary_mode="reflect".
        """
        super().__init__(alpha_order, beta_order, intercept, global_intercept)
        self.basis_order = basis_order
        self.intercept_basis_order = intercept_basis_order
        self.basis_order_min = basis_order_min
        self.tv_intercept = tv_intercept
        self.family = family
        self.gpu = gpu
        self.boundary_mode = boundary_mode
        self.boundary_pad = boundary_pad
        self.num_coefs = sum(self.alpha_orders)+sum(self.beta_order)

    def generate_basis_fourier(self):
        """
        Internal method to generate fourier basis.
        """
        t = np.arange(1+self.alpha_order,self.T+1)
        self.cos_basis = cos_basis(t.reshape(-1,1),np.arange(1,self.basis_order+1).reshape(1,-1),self.T-self.alpha_order)
        self.sin_basis = sin_basis(t.reshape(-1,1),np.arange(1,self.basis_order+1).reshape(1,-1),self.T-self.alpha_order)
        return np.concatenate([np.ones((t.shape[0],1)),self.cos_basis,self.sin_basis],axis=1)

    def generate_basis_haar(self):
        """
        Internal method to generate Haar basis.
        """
        t= np.arange(1,self.T+1-self.alpha_order,dtype=float)
        # shape = (len(t),1+sum_n 2^n = 2^(n+1))
        t[-1] -= 1e-10 #prevents the last t to be evaluated to 0
        self.num_father = 1
        return np.concatenate([np.ones((t.shape[0],1))]+[haar_basis(t.reshape(-1,1),n,np.arange(2**n).reshape(1,-1),self.T-self.alpha_order) 
                                                         for n in range(0,self.basis_order+1)],axis=1)#/(self.T-self.alpha_order)

    def generate_basis_db(self,vanishing_moments,basis_order_min):
        """
        Internal method to generate DB basis.
        """
        db = DB(vanishing_moments, T=self.T-self.alpha_order, T_ceil=int(2**np.ceil(np.log2(self.T-self.alpha_order))), j0=basis_order_min)
        basis = db.basis[:,-2**(1+self.basis_order):][:,::-1].copy()#minimum 2, ::-1 gives a reversed order
        self.basis_order_min = db.j0
        self.num_father = 2**db.j0
        return np.ascontiguousarray(basis,dtype=np.float64)
        
    
    def generate_basis(self):
        """
        Internal method to generate basis based on the value of self.family.
        """
        if self.family[:2] == "db":
            return self.generate_basis_db(int(self.family[2:]),self.basis_order_min)
        match self.family:
            case "fourier":
                return self.generate_basis_fourier()
            case "haar":
                return self.generate_basis_haar()
            case _:
                raise ValueError("Unknown family.")

    def fit(self,network,vts,thresh=None,tau="auto",refit=False,sigma_bandwidth=None,refit_endpoint_only=False,endpoint_tol=1e-12,ind_thresh=True,mad_level=1):
        """
        Fit the time-varying GNAR model to vector time series data.
    
        Parameters
        ----------
        network : Network object
            Network object used by the GNAR model. If beta_order is not zeros,
            network.size must match vts.shape[1].
    
        vts : ndarray
            Vector time series data with shape (T, N), where T is the number
            of time points and N is the number of nodes.
    
        thresh : {None, "hard", "soft"}, default=None
            Optional wavelet thresholding method.
    
        tau : float or "auto", default="auto"
            Threshold multiplier. If "auto", use the default value in the paper.
    
        refit : bool, default=False
            Whether to refit the model after thresholding.
    
        sigma_bandwidth : int, optional
            Bandwidth used when estimating empirical variance of raw coefficients.
            If not provided, automatically chosen based on the length of the time series.
    
        refit_endpoint_only : bool, default=False
            Whether to refit only coefficients that affect the final time point (see the paper).
    
        endpoint_tol : float, default=1e-12
            Tolerance used to decide whether a basis coefficient affects the
            endpoint.
    
        ind_thresh : bool, default=True
            Whether to use individual thresholding based on empirical variance0.
            If False, use universal thresholding based on MAD.
    
        mad_level : int, default=1
            Number of wavelet levels used for computing MAD used in universal thresholding.
        """
        if thresh not in (None,"hard","soft"):
            raise ValueError("thresh must be one of: None, 'hard', 'soft'.")
        if self.boundary_mode not in ("periodic","reflect"):
            raise ValueError("boundary_mode must be periodic or reflect.")

        self.network = network
        self.vts = vts
        self.vts_end = vts[-self.alpha_order:,:]
        self.T,self.N = vts.shape
        self.coef_eval_index = self.T-self.alpha_order-1
        
        if self.basis_order is None:
            self.basis_order = np.floor(np.log2(np.sqrt((self.T-self.alpha_order)*self.N))).astype(int)
            if self.intercept:
                if not self.global_intercept:
                    self.intercept_basis_order = np.floor(np.log2(np.sqrt(self.T-self.alpha_order))).astype(int)
                else:
                    self.intercept_basis_order = self.basis_order 
        else:
            if self.intercept_basis_order is None:
                self.intercept_basis_order = self.basis_order

        if self.family[:2] == "db" and self.boundary_mode == "reflect":
            if self.boundary_pad is None:
                boundary_pad = max(1,2*int(self.family[2:])-1)
            else:
                boundary_pad = max(0,int(self.boundary_pad))
            if boundary_pad > 0:
                index = np.arange(self.T,self.T+boundary_pad,dtype=np.int64)
                while np.any(index < 0) or np.any(index >= self.T):
                    index[index < 0] = -index[index < 0]-1
                    index[index >= self.T] = 2*self.T-1-index[index >= self.T]
                self.vts_fit = np.vstack((vts,vts[index,:]))
            else:
                self.vts_fit = vts.copy()
        else:
            self.vts_fit = vts.copy()

        self.T,self.N = self.vts_fit.shape
        self.y = self.vts_fit[self.alpha_order:,:].flatten("F")
        self.X = self.transformVTS(self.vts_fit)

        self.basis = np.ascontiguousarray(self.generate_basis(),dtype=np.float64)

        if self.intercept:
            if self.tv_intercept:
                if not self.global_intercept:
                    self.intercept_basis = self.basis[:,:2**(1+self.intercept_basis_order)]
                    self.X_basis = np.hstack((
                        (self.X[:,:-self.N,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1),
                        (self.X[:,-self.N:,np.newaxis]*np.tile(self.intercept_basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1)
                    ))
                else:
                    self.X_basis = (self.X[:,:,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1)
            else:
                self.X_basis = np.hstack((
                        (self.X[:,:self.num_coefs,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1),
                        self.X[:,self.num_coefs:]))
        else:
            self.X_basis = (self.X[:,:,np.newaxis]*np.tile(self.basis[:,np.newaxis,:],(self.N,1,1))).reshape(self.X.shape[0],-1)
        self.X_basis = np.ascontiguousarray(self.X_basis,dtype=np.float64)
        
        if self.gpu:
            X_basis_gpu = torch.as_tensor(self.X_basis,dtype=torch.float64,device="cuda")
            y_gpu = torch.as_tensor(self.y,dtype=torch.float64,device="cuda")
            Q,R = torch.linalg.qr(X_basis_gpu,mode="reduced")
            basis_coefs_gpu = torch.linalg.solve_triangular(R,(Q.mT@y_gpu)[:,None],upper=True).squeeze(1)
            resid = y_gpu-X_basis_gpu@basis_coefs_gpu
            basis_coefs, self.res = basis_coefs_gpu.cpu().numpy(), np.array([float(torch.dot(resid,resid).cpu())])
        else:
            basis_coefs, self.res = np.linalg.lstsq(self.X_basis,self.y,rcond=None)[:2]

        if ind_thresh:
            self.basis_coef_sigma(basis_coefs,sigma_bandwidth=sigma_bandwidth)

        if not self.intercept:
            self.basis_coefs = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
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
            self.wavelet_thresh(thresh,tau,refit,refit_endpoint_only,endpoint_tol,ind_thresh,mad_level)
        else:
            self.set_last_coefs()

    def basis_coef_sigma(self,basis_coefs,sigma_bandwidth=None):
        """
        Internal method for computing empirical variance of raw coefficients. 
        Use the variance estimator in the appendix.
        """
        if sigma_bandwidth is None:
            sigma_bandwidth = max(5,int(np.ceil(np.sqrt(self.T-self.alpha_order))))
        if not hasattr(self.network,"w_mats") and len(self.beta_order) > 0 and max(self.beta_order) > 0:
            raise ValueError("network must contain w_mats.")

        weight_mats = []
        lag_idx = []
        for i in range(1,self.alpha_order+1):
            weight_mats.append(np.eye(self.N,dtype=np.float64))
            lag_idx.append(i-1)
        for i,s_i in enumerate(self.beta_order,start=1):
            for r in range(1,s_i+1):
                weight_mats.append(np.asarray(self.network.w_mats[r],dtype=np.float64))
                lag_idx.append(i-1)
        lag_idx = np.asarray(lag_idx,dtype=np.int64)

        if self.gpu:
            basis = torch.as_tensor(np.ascontiguousarray(self.basis),dtype=torch.float64,device="cuda")
            vts = np.asarray(self.vts_fit,dtype=np.float64).copy()

            if self.intercept:
                if self.tv_intercept:
                    if not self.global_intercept:
                        intercept_basis_coefs = basis_coefs[-self.intercept_basis.shape[1]*self.N:].reshape(self.intercept_basis.shape[1],self.N,order="F")
                        vts[self.alpha_order:,:] -= self.intercept_basis@intercept_basis_coefs
                    else:
                        basis_coefs_mat = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
                        vts[self.alpha_order:,:] -= self.basis@basis_coefs_mat[:,self.num_coefs:]
                else:
                    vts[self.alpha_order:,:] -= basis_coefs[self.basis.shape[1]*self.num_coefs:]

            vts = torch.as_tensor(np.ascontiguousarray(vts),dtype=torch.float64,device="cuda")
            resid = torch.as_tensor(np.ascontiguousarray(self.y.reshape(self.T-self.alpha_order,self.N,order="F")-(self.X_basis@basis_coefs).reshape(self.T-self.alpha_order,self.N,order="F")),dtype=torch.float64,device="cuda")
            kernel = torch.arange(self.T-self.alpha_order,dtype=torch.float64,device="cuda")
            kernel = torch.clamp(1-torch.abs(kernel[:,None]-kernel[None,:])/float(sigma_bandwidth),min=0)
            kernel = kernel/kernel.sum(dim=1,keepdim=True)
            weight_mats = torch.as_tensor(np.ascontiguousarray(np.stack(weight_mats,axis=0)),dtype=torch.float64,device="cuda")
            resp_idx = torch.arange(self.alpha_order,self.T,dtype=torch.long,device="cuda")
            lagged_vts = [vts[resp_idx-i,:] for i in range(1,self.alpha_order+1)]

            resid_network = []
            lag_network = []
            lag_network_T = []
            for i in range(self.num_coefs):
                resid_network.append(resid@weight_mats[i])
                lag_network.append(lagged_vts[lag_idx[i]]@weight_mats[i].T)
                lag_network_T.append(lagged_vts[lag_idx[i]].T)

            Atilde_hat = torch.zeros((self.basis.shape[1]*self.num_coefs,self.basis.shape[1]*self.num_coefs),dtype=torch.float64,device="cuda")
            Btilde_hat = torch.zeros((self.basis.shape[1]*self.num_coefs,self.basis.shape[1]*self.num_coefs),dtype=torch.float64,device="cuda")

            for r in range(self.num_coefs):
                r_slice = slice(r*self.basis.shape[1],(r+1)*self.basis.shape[1])
                for q in range(r+1):
                    q_slice = slice(q*self.basis.shape[1],(q+1)*self.basis.shape[1])
                    design_weight = kernel@(torch.sum(lag_network[q]*lag_network[r],dim=1)/self.N)
                    resid_product = (resid_network[q]@lag_network_T[q])*(resid_network[r]@lag_network_T[r])
                    variance_weight = torch.sum((kernel@resid_product)*kernel,dim=1)/self.N
                    A_block = (basis.T@(basis*design_weight[:,None]))/(self.T-self.alpha_order)
                    B_block = (basis.T@(basis*variance_weight[:,None]))/(self.T-self.alpha_order)
                    Atilde_hat[q_slice,r_slice] = A_block
                    Btilde_hat[q_slice,r_slice] = B_block
                    if q != r:
                        Atilde_hat[r_slice,q_slice] = A_block.T
                        Btilde_hat[r_slice,q_slice] = B_block.T

            Atilde_hat = 0.5*(Atilde_hat+Atilde_hat.T)
            Btilde_hat = 0.5*(Btilde_hat+Btilde_hat.T)
            try:
                Atilde_inv = torch.linalg.solve(Atilde_hat,torch.eye(Atilde_hat.shape[0],dtype=torch.float64,device="cuda"))
            except RuntimeError:
                Atilde_inv = torch.linalg.pinv(Atilde_hat)
            basis_coef_cov = (Atilde_inv@Btilde_hat@Atilde_inv)/((self.T-self.alpha_order)*self.N)
            basis_coef_cov = basis_coef_cov.cpu().numpy()

        else:
            basis = np.ascontiguousarray(self.basis,dtype=np.float64)
            vts = np.asarray(self.vts_fit,dtype=np.float64).copy()

            if self.intercept:
                if self.tv_intercept:
                    if not self.global_intercept:
                        intercept_basis_coefs = basis_coefs[-self.intercept_basis.shape[1]*self.N:].reshape(self.intercept_basis.shape[1],self.N,order="F")
                        vts[self.alpha_order:,:] -= self.intercept_basis@intercept_basis_coefs
                    else:
                        basis_coefs_mat = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
                        vts[self.alpha_order:,:] -= self.basis@basis_coefs_mat[:,self.num_coefs:]
                else:
                    vts[self.alpha_order:,:] -= basis_coefs[self.basis.shape[1]*self.num_coefs:]

            resid = np.ascontiguousarray(self.y.reshape(self.T-self.alpha_order,self.N,order="F")-(self.X_basis@basis_coefs).reshape(self.T-self.alpha_order,self.N,order="F"),dtype=np.float64)
            kernel = np.arange(self.T-self.alpha_order,dtype=np.float64)
            kernel = np.maximum(1-np.abs(kernel[:,None]-kernel[None,:])/float(sigma_bandwidth),0)
            kernel /= kernel.sum(axis=1,keepdims=True)
            weight_mats = np.ascontiguousarray(np.stack(weight_mats,axis=0),dtype=np.float64)
            resp_idx = np.arange(self.alpha_order,self.T,dtype=np.int64)
            lagged_vts = [np.ascontiguousarray(vts[resp_idx-i,:],dtype=np.float64) for i in range(1,self.alpha_order+1)]

            resid_network = []
            lag_network = []
            lag_network_T = []
            for i in range(self.num_coefs):
                resid_network.append(resid@weight_mats[i])
                lag_network.append(lagged_vts[lag_idx[i]]@weight_mats[i].T)
                lag_network_T.append(lagged_vts[lag_idx[i]].T)

            Atilde_hat = np.zeros((self.basis.shape[1]*self.num_coefs,self.basis.shape[1]*self.num_coefs),dtype=np.float64)
            Btilde_hat = np.zeros((self.basis.shape[1]*self.num_coefs,self.basis.shape[1]*self.num_coefs),dtype=np.float64)

            for r in range(self.num_coefs):
                r_slice = slice(r*self.basis.shape[1],(r+1)*self.basis.shape[1])
                for q in range(r+1):
                    q_slice = slice(q*self.basis.shape[1],(q+1)*self.basis.shape[1])
                    design_weight = kernel@(np.sum(lag_network[q]*lag_network[r],axis=1)/self.N)
                    resid_product = (resid_network[q]@lag_network_T[q])*(resid_network[r]@lag_network_T[r])
                    variance_weight = np.sum((kernel@resid_product)*kernel,axis=1)/self.N
                    A_block = (basis.T@(basis*design_weight[:,None]))/(self.T-self.alpha_order)
                    B_block = (basis.T@(basis*variance_weight[:,None]))/(self.T-self.alpha_order)
                    Atilde_hat[q_slice,r_slice] = A_block
                    Btilde_hat[q_slice,r_slice] = B_block
                    if q != r:
                        Atilde_hat[r_slice,q_slice] = A_block.T
                        Btilde_hat[r_slice,q_slice] = B_block.T

            Atilde_hat = 0.5*(Atilde_hat+Atilde_hat.T)
            Btilde_hat = 0.5*(Btilde_hat+Btilde_hat.T)
            try:
                Atilde_inv = np.linalg.solve(Atilde_hat,np.eye(Atilde_hat.shape[0],dtype=np.float64))
            except np.linalg.LinAlgError:
                Atilde_inv = np.linalg.pinv(Atilde_hat)
            basis_coef_cov = (Atilde_inv@Btilde_hat@Atilde_inv)/((self.T-self.alpha_order)*self.N)

        basis_coef_cov = 0.5*(basis_coef_cov+basis_coef_cov.T)
        basis_coef_sigma = np.sqrt(np.maximum(np.diag(basis_coef_cov),np.finfo(np.float64).tiny))
        self.basis_coef_sigma_mat = basis_coef_sigma.reshape(self.basis.shape[1],self.num_coefs,order="F")

    def set_last_coefs(self):
        """
        Internal method for setting coefficients at the last time point.
        """
        self.last_coefs = self.get_coefs()[self.coef_eval_index]
        if self.intercept:
            if self.tv_intercept:
                self.last_coefs = np.concatenate((self.last_coefs,self.get_intercept()[self.coef_eval_index]))
            else:
                self.last_coefs = np.concatenate((self.last_coefs,self.get_intercept()))
            
    def get_coefs(self):
        """
        Returns the fitted time-varying coefficients.
        """
        return self.basis@self.basis_coefs

    def get_intercept(self):
        """
        Returns the fitted intercept.
        """
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
        """
        Predict future values from the fitted model.
    
        Parameters
        ----------
        length : int
            Number of future time points to predict.
    
        nodes : list[int] or ndarray, optional
            Specific node indices to return. If not provided, predictions for
            all nodes are returned.
    
        vts_end : ndarray, optional
            Recent observations used to start prediction. If not provided,
            the final "alpha_order" observations from the fitted data are used.
    
        Returns
        -------
        ndarray
            Predicted values. If nodes is None, the shape is
            (length, N). Otherwise, only the selected nodes are returned.
        """
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
        """
        Method for simulating TV-GNAR process.
        See doc on TVGNAR_sim.
        """
        self.network = network
        l = len(initial_vts)
        vts_sim = np.zeros((length+l,self.network.size))
        vts_sim[:l,:] = initial_vts
        for i in range(length):
            vts_sim[l+i,:] = self.transformVTS(vts_sim[l+i-self.alpha_order:l+i+1,:])@coefs[i,:] + multivariate_normal(
                cov=error_cov_mat).rvs(1)
        return vts_sim[l:]

    def wavelet_thresh(self, thresh, tau="auto", refit=False, refit_endpoint_only=False, endpoint_tol=1e-12, ind_thresh=True, mad_level=1):
        """
        Internal method for wavelet thresholding.
        """
        match thresh:
            case "hard":
                thresh_fun = hard_thresh
            case "soft":
                thresh_fun = soft_thresh
            case _:
                raise ValueError("thresh must be 'hard' or 'soft'.")
        if isinstance(tau,str):
            if tau.lower() != "auto":
                raise ValueError("tau must be numeric or 'auto'.")
            tau = np.sqrt(2*np.log(max(self.basis_coefs.shape[0]-self.num_father,1)))
        else:
            tau = float(tau)
    
        if ind_thresh:
            self.thresh_const = tau
            self.basis_coefs[self.num_father:] = thresh_fun(
                self.basis_coefs[self.num_father:],
                tau*self.basis_coef_sigma_mat[self.num_father:]
            )
        else:
            index = np.sum([2**(self.basis_order-n) for n in range(mad_level)])
            self.mad = mad(self.basis_coefs[-index:],axis=0)
            self.basis_coefs[self.num_father:] = thresh_fun(
                self.basis_coefs[self.num_father:],
                self.mad*tau
            )
    
        if self.intercept and self.tv_intercept:
            index = np.sum([2**(self.intercept_basis_order-n) for n in range(mad_level)])
            intercept_thresh_const = np.sqrt(2*np.log(max(self.intercept_basis_coefs.shape[0]-self.num_father,1)))
            self.intercept_basis_coefs[self.num_father:] = thresh_fun(
                self.intercept_basis_coefs[self.num_father:],
                mad(self.intercept_basis_coefs[-index:],axis=0)*intercept_thresh_const
            )
    
        if not self.intercept:
            basis_coefs = self.basis_coefs.reshape(-1,order="F")
        else:
            if self.tv_intercept:
                if not self.global_intercept:
                    basis_coefs = np.concatenate((self.basis_coefs.reshape(-1,order="F"),self.intercept_basis_coefs.reshape(-1,order="F")))
                else:
                    basis_coefs = np.hstack((self.basis_coefs,self.intercept_basis_coefs)).reshape(-1,order="F")
            else:
                basis_coefs = np.concatenate((self.basis_coefs.reshape(-1,order="F"),self.intercept_coefs))
    
        if not refit:
            resid = self.y-self.X_basis@basis_coefs
            self.res = np.array([float(resid@resid)])
            self.set_last_coefs()
            return
    
        if not self.intercept:
            keep = np.r_[np.ones((self.num_father,self.X.shape[1]),dtype=bool),self.basis_coefs[self.num_father:] != 0].reshape(-1,order="F")
        else:
            if self.tv_intercept:
                if not self.global_intercept:
                    keep = np.concatenate((
                        np.r_[np.ones((self.num_father,self.num_coefs),dtype=bool),self.basis_coefs[self.num_father:] != 0].reshape(-1,order="F"),
                        np.r_[np.ones((self.num_father,self.N),dtype=bool),self.intercept_basis_coefs[self.num_father:] != 0].reshape(-1,order="F")
                    ))
                else:
                    keep = np.r_[np.ones((self.num_father,self.X.shape[1]),dtype=bool),np.hstack((self.basis_coefs[self.num_father:] != 0,self.intercept_basis_coefs[self.num_father:] != 0))].reshape(-1,order="F")
            else:
                keep = np.concatenate((
                    np.r_[np.ones((self.num_father,self.num_coefs),dtype=bool),self.basis_coefs[self.num_father:] != 0].reshape(-1,order="F"),
                    np.ones(self.N,dtype=bool)
                ))
    
        if refit_endpoint_only:
            endpoint = np.zeros(self.X_basis.shape[1],dtype=bool)
            endpoint[:self.basis.shape[1]*self.num_coefs] = np.tile(
                (np.abs(self.basis[self.coef_eval_index,:]) > endpoint_tol).reshape(-1,1),
                (1,self.num_coefs)
            ).reshape(-1,order="F")
            if self.intercept:
                if self.tv_intercept:
                    if not self.global_intercept:
                        endpoint[self.basis.shape[1]*self.num_coefs:] = np.tile(
                            (np.abs(self.intercept_basis[self.coef_eval_index,:]) > endpoint_tol).reshape(-1,1),
                            (1,self.N)
                        ).reshape(-1,order="F")
                    else:
                        endpoint = np.tile(
                            (np.abs(self.basis[self.coef_eval_index,:]) > endpoint_tol).reshape(-1,1),
                            (1,self.X.shape[1])
                        ).reshape(-1,order="F")
                else:
                    endpoint[self.basis.shape[1]*self.num_coefs:] = True
            refit_mask = keep & endpoint
            fixed_mask = keep & (~refit_mask)
            if np.any(refit_mask):
                if self.gpu:
                    X_basis_gpu = torch.as_tensor(self.X_basis[:,refit_mask],dtype=torch.float64,device="cuda")
                    y_gpu = torch.as_tensor(self.y-self.X_basis[:,fixed_mask]@basis_coefs[fixed_mask],dtype=torch.float64,device="cuda")
                    Q,R = torch.linalg.qr(X_basis_gpu,mode="reduced")
                    basis_coefs[refit_mask] = torch.linalg.solve_triangular(R,(Q.mT@y_gpu)[:,None],upper=True).squeeze(1).cpu().numpy()
                else:
                    basis_coefs[refit_mask] = np.linalg.lstsq(
                        self.X_basis[:,refit_mask],
                        self.y-self.X_basis[:,fixed_mask]@basis_coefs[fixed_mask],
                        rcond=None
                    )[0]
        else:
            if np.any(keep):
                if self.gpu:
                    X_basis_gpu = torch.as_tensor(self.X_basis[:,keep],dtype=torch.float64,device="cuda")
                    y_gpu = torch.as_tensor(self.y,dtype=torch.float64,device="cuda")
                    Q,R = torch.linalg.qr(X_basis_gpu,mode="reduced")
                    basis_coefs_refit = torch.linalg.solve_triangular(R,(Q.mT@y_gpu)[:,None],upper=True).squeeze(1).cpu().numpy()
                else:
                    basis_coefs_refit = np.linalg.lstsq(self.X_basis[:,keep],self.y,rcond=None)[0]
                basis_coefs = np.zeros_like(basis_coefs)
                basis_coefs[keep] = basis_coefs_refit
            else:
                basis_coefs = np.zeros_like(basis_coefs)
    
        resid = self.y-self.X_basis@basis_coefs
        self.res = np.array([float(resid@resid)])
        if not self.intercept:
            self.basis_coefs = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
        else:
            if self.tv_intercept:
                if not self.global_intercept:
                    self.basis_coefs = basis_coefs[:self.basis.shape[1]*self.num_coefs].reshape(self.basis.shape[1], self.num_coefs, order="F")
                    self.intercept_basis_coefs = basis_coefs[self.basis.shape[1]*self.num_coefs:].reshape(self.intercept_basis.shape[1], self.N, order="F")
                else:
                    basis_coefs = basis_coefs.reshape(self.basis.shape[1],self.X.shape[1],order="F")
                    self.basis_coefs,self.intercept_basis_coefs = basis_coefs[:,:self.num_coefs], basis_coefs[:,self.num_coefs:]
            else:
                self.basis_coefs = basis_coefs[:self.basis.shape[1]*self.num_coefs].reshape(self.basis.shape[1], self.num_coefs, order="F")
                self.intercept_coefs = basis_coefs[self.basis.shape[1]*self.num_coefs:]
        self.set_last_coefs()

    def __del__(self):
        """
        Memory control for repeated large experiments.
        """
        try:
            for name in (
                "X_basis", "X", "y", "basis", "intercept_basis",
                "basis_coef_sigma_mat", "basis_coefs",
                "intercept_basis_coefs", "intercept_coefs"
            ):
                if hasattr(self, name):
                    delattr(self, name)
        except Exception:
            pass
        
        
def TVGNAR_sim(network, alpha_order, beta_order, coefs, intercept, global_intercept, error_cov_mat,length,burn_in=100):
    """
    Returns a simulated sample of TV-GNAR process.

    Parameters
    ----------
    network : Network object
        Network object used for simulating the TV-GNAR process.

    alpha_order : int
        Max autoregressive lag.
    
    beta_order : list[int]
        Max stage of neighbourhood to include for each lag.

    coefs : ndarray
        Coefficient array with shape (T, P).
        The first dimension must match length + burn_in.
        The second dimension must be in the order 
        (alpha_1, beta_1,1, ..., beta_1,s_1, ..., alpha_p, beta_p,1, ..., beta_p,s_p, intercept(s)).

    intercept : bool
        Whether to include an intercept term.

    global_intercept : bool
        Whether to use a global intercept across all nodes.

    error_cov_mat : ndarray
        Covariance matrix for the innovation. Must match network.size.

    length : int
        Length of returned simulated time series.

    burn_in : int, default=100
        Length of burn-in period.
        
    """
    vts = np.zeros((length+burn_in+alpha_order,network.size))
    vts[:alpha_order] = norm(loc=0).rvs(size=(alpha_order,network.size))
    model = TVGNAR(alpha_order, beta_order, basis_order=1,intercept=intercept,global_intercept=global_intercept)
    vts[alpha_order:] = model.simulate(network,vts[:alpha_order],length+burn_in,coefs,error_cov_mat)
    return vts[burn_in+alpha_order:]