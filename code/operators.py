#-------------------------------------------------------------------------------
# this file defines time-dependent and steady-state solution operators for the
# upper and lower surface elevations and (depth-dependent) velocities.
#-------------------------------------------------------------------------------
import numpy as np
from kernel_fcns import B, R, Uh, Us, Wh, Ws, conv, ker_h, ker_s
from params import delta, eps, k, kx, t
from scipy.fft import fft2, ifft2


#---------------------Ice-surface elevation solution operators------------------
def compute_h(m,alpha):
    # solution operator tha returns the upper-surface elevation h given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the time-dependent version
    m_ft = fft2(m)
    h_ft = conv(ker_h(t,k,kx,alpha),m_ft)
    h_ft[k<10*k.min()] = 0
    return ifft2(np.nan_to_num(h_ft)).real


def compute_s(m,alpha):
    # solution operator tha returns the lower-surface elevation s given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the time-dependent version
    m_ft = fft2(m)
    s_ft = conv(ker_s(t,k,kx,alpha),m_ft)
    s_ft[k<10*k.min()] = 0
    return ifft2(np.nan_to_num(s_ft)).real

def compute_u(h,s,z):
    # function for computing the horizontal velocity in the x direction (u)
    # at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    u_ft = ((1j*2*np.pi*kx)/(2*np.pi*k)**2)*(Uh(k,z)*h_ft+Us(k,z)*delta*s_ft)
    u_ft[k<10*k.min()] = 0
    return ifft2(np.nan_to_num(u_ft)).real

def compute_v(h,s,z):
    # function for computing the horizontal velocity in the y direction (v)
    # at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    v_ft = ((1j*2*np.pi*ky)/(2*np.pi*k)**2)*(Uh(k,z)*h_ft+Us(k,z)*delta*s_ft)
    v_ft[k<10*k.min()] = 0
    return ifft2(np.nan_to_num(v_ft)).real

def compute_w(h,s,z):
    # function for computing the vertical velocity (w) at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    w_ft = Wh(k,z)*h_ft + Ws(k,z)*delta*s_ft
    w_ft[k<10*k.min()] = 0
    return ifft2(np.nan_to_num(w_ft)).real


def h_steady(m,alpha):
     # solution operator tha returns the upper-surface elevation h given the advection
     # parameter alpha (a scalar) and the melt-rate field (m)
     #
     # note: this is the steady-state version, so the melt-rate m should not depend
     # on t (I retain a time index for simplicity, for now; each time-slice is identical of course)
     R_ = R(k)
     B_ = B(k)
     m_ft = fft2(m)
     c0 = delta*(R_**2-B_**2)+2*np.pi*kx*1j*alpha*(delta+1)*R_-((2*np.pi*kx)**2)*(alpha**2)
     f0 = c0/B_
     f = 1/(1e-5*eps+f0)
     h_e_ft = -delta*f*m_ft
     h_e_ft[k<10*k.min()] = 0
     return ifft2(np.nan_to_num(h_e_ft)).real

def s_steady(m,alpha):
     # solution operator tha returns the upper-surface elevation h given the advection
     # parameter alpha (a scalar) and the melt-rate field (m)
     #
     # note: this is the steady-state version, so the melt-rate m should not depend
     # on t (I retain a time index for simplicity, for now; each time-slice is identical of course)
     R_ = R(k)
     B_ = B(k)
     m_ft = fft2(m)
     c0 = delta*(R_**2-B_**2)+2*np.pi*kx*1j*alpha*(delta+1)*R_-((2*np.pi*kx)**2)*(alpha**2)
     f0 = c0 / ((R_+2*np.pi*kx*1j*alpha))
     f = 1/(1e-5*eps+f0)
     s_e_ft = f*m_ft
     s_e_ft[k<10*k.min()] = 0
     return ifft2(np.nan_to_num(s_e_ft)).real