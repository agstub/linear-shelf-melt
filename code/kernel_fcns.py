# this file contains the integral kernel functions that are used for computing
# the elevation and velocity solutions

import numpy as np
from params import Nt,eps,delta,dt
from scipy.signal import fftconvolve

#---------------------------convolution operator--------------------------------
def conv(a,b):
    return dt*fftconvolve(a,b,mode='full',axes=0)[0:Nt,:,:]

#-----------------------------indicator function--------------------------------
# used to trunctate the integral kernels at a small k=k_min
def ind(k,k_min):
    return 0.5*(np.sign(k-k_min)+1)

#------------------------Functions relevant to kernels--------------------------
def R(k):
    # relaxation function for floating ice
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    R1 =  np.exp(4*n) + (4*n)*np.exp(2*n) - 1 
    D = n*(np.exp(4*n) -2*(1+2*n**2)*np.exp(2*n) + 1)
    f0 = D/R1
    f = 1/(eps+f0)
    return f

def B(k):
    # buoyancy transfer function for floating ice
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    B1 =  2*(n+1)*np.exp(3*n) + 2*(n-1)*np.exp(n)
    D = n*(np.exp(4*n) -2*(1+2*n**2)*np.exp(2*n) + 1)
    f0 = D/B1
    f =1/(eps+f0)
    return f

def Lamda_p(k,kx,alpha):
    # expression for the larger eigenvalue in the problem
    R_ = R(k)
    B_ = B(k)
    chi = (1-delta)*R_
    mu = np.sqrt(4*delta*(B_)**2 + chi**2)
    Lp = -0.5*(delta+1)*R_+0.5*mu-1j*(2*np.pi*kx)*alpha
    return Lp

def Lamda_m(k,kx,alpha):
    # expression for the smaller eigenvalue in the problem
    R_ = R(k)
    B_ = B(k)
    chi = (1-delta)*R_
    mu = np.sqrt(4*delta*(B_)**2 + chi**2)
    Lm = -1j*(2*np.pi*kx)*alpha-0.5*(delta+1)*R_-0.5*mu
    return Lm

def Uh(k,z):
    # horizontal velocity reponse function that multiplies the upper surface
    # elevation, where z (z=0 is base, z=1 is surface) is the depth
    n = 2*np.pi*k
    R = z*(2*n*(np.exp(2*n) +np.exp(2*n*z)))-(1-z)*(np.exp(2*n*(z+1)) +np.exp(2*n)-np.exp(2*n*z)-1)
    D = np.exp(n*z)*(np.exp(4*n)-2*(1+2*n**2)*np.exp(2*n) +1)/n
    f0 = D/(np.exp(n)*R)
    f = 1/(1e-5*eps+f0)
    return f

def Us(k,z):
    # horizontal velocity reponse function that multiplies the lower surface
    # elevation, where z (z=0 is base, z=1 is surface) is the depth
    n = 2*np.pi*k
    R = z*(np.exp(4*n)+np.exp(2*n*(z+1)) -np.exp(2*n)-np.exp(2*n*z))-(1-z)*2*n*np.exp(2*n)*(np.exp(2*n*z)+1)
    D = np.exp(n*z)*(np.exp(4*n)-2*(1+2*n**2)*np.exp(2*n) +1)/n
    f0 = D/R
    f = 1/(1e-5*eps+f0)
    return f


def Wh(k,z):
    # vertical velocity reponse function that multiplies the upper surface
    # elevation, where z (z=0 is base, z=1 is surface) is the depth
    n = 2*np.pi*k
    R = -(n*z*(2*n*np.exp(2*n) + np.exp(2*n) - 1) + n*np.exp(2*n) + n + (-n*z*(2*n + np.exp(2*n) - 1) + n*np.exp(2*n) + n + np.exp(2*n) - 1)*np.exp(2*n*z) + np.exp(2*n) - 1)*np.exp(-n*(z - 1))
    D = n*(np.exp(4*n) -2*(1+2*n**2)*np.exp(2*n) + 1)
    f0 = D/R
    f = 1/(eps+f0)
    return f

def Ws(k,z):
    # vertical velocity reponse function that multiplies the lower surface
    # elevation, where z (z=0 is base, z=1 is surface) is the depth
    n = 2*np.pi*k
    R = -((-2*n**2 + n*z*(2*n + np.exp(2*n) - 1) + 2*n + np.exp(2*n) - 1)*np.exp(2*n) + (2*n**2*np.exp(2*n) - n*z*(2*n*np.exp(2*n) + np.exp(2*n) - 1) + 2*n*np.exp(2*n) + np.exp(2*n) - 1)*np.exp(2*n*z))*np.exp(-n*z)
    D = n*(np.exp(4*n) -2*(1+2*n**2)*np.exp(2*n) + 1)
    f0 = D/R
    f = 1/(eps+f0)
    return f

#------------------------------ Kernels-----------------------------------------
def ker_h(t,k,kx,alpha):
    # kernel for computing the upper surface elevation when the melt-rate forcing
    # is time-dependent
    R_ = R(k)
    B_ = B(k)
    chi = (1-delta)*R_
    mu = np.sqrt(4*delta*(B_)**2 + chi**2)
    Lp = Lamda_p(k,kx,alpha)
    Lm = Lamda_m(k,kx,alpha)

    ker0 = (delta*B_/mu)*np.exp(Lp*t)
    ker1 = (delta*B_/mu)*np.exp(Lm*t)
    K = ker1-ker0
    return K

def ker_s(t,k,kx,alpha):
    # kernel for computing the lower surface elevation when the melt-rate forcing
    # is time-dependent
    R_ = R(k)
    B_ = B(k)
    chi = (1-delta)*R_
    mu = np.sqrt(4*delta*(B_)**2 + chi**2)
    Lp = Lamda_p(k,kx,alpha)
    Lm = Lamda_m(k,kx,alpha)

    ker0 = (1/(2*mu))*(mu-chi)*np.exp(Lm*t)
    ker1 = (1/(2*mu))*(mu+chi)*np.exp(Lp*t)
    K = ker0+ker1
    return K
