#-------------------------------------------------------------------------------
# this file defines time-dependent and steady-state solution operators for the
# upper and lower surface elevations and (depth-dependent) velocities.
#-------------------------------------------------------------------------------
import numpy as np
from kernel_fcns import ker_h,ker_s,conv,R,B,ind,Uh,Us,Wh,Ws
from params import t,k,kx,ky,dx,Nt,t_final,delta
from scipy.fft import ifft2,fft2

#---------------------Ice-surface elevation solution operators------------------
def compute_h(t,k,kx,alpha,m):
    # solution operator tha returns the upper-surface elevation h given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the time-dependent version
    m_ft = fft2(m)
    h_ft = conv(ker_h(t,k,kx,alpha),m_ft)
    return ifft2(h_ft).real


def compute_s(t,k,kx,alpha,m):
    # solution operator tha returns the lower-surface elevation s given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the time-dependent version
    m_ft = fft2(m)
    s_ft = conv(ker_s(t,k,kx,alpha),m_ft)
    return ifft2(s_ft).real

def h_steady(k,alpha,m):
    # solution operator tha returns the upper-surface elevation h given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the steady-state version, so the melt-rate m should not depend
    # on t (I retain a time index for simplicity, for now; each time-slice is identical of course)
    R_ = R(k)
    B_ = B(k)
    m_ft = fft2(m)
    c0 = delta*(R_**2-B_**2)+2*np.pi*kx*1j*alpha*(delta+1)*R_-((2*np.pi*kx)**2)*(alpha**2)
    h_e_ft = -delta*B_*m_ft/c0
    return ifft2(np.nan_to_num(h_e_ft*ind(k,1e-4))).real


def s_steady(k,alpha,m):
    # solution operator tha returns the lower-surface elevation s given the advection
    # parameter alpha (a scalar) and the melt-rate field (m)
    #
    # note: this is the steady-state version, so the melt-rate m should not depend
    # on t (I retain a time index for simplicity, for now; each time-slice is identical of course)
    R_ = R(k)
    B_ = B(k)
    m_ft = fft2(m)
    c0 = delta*(R_**2-B_**2)+2*np.pi*kx*1j*alpha*(delta+1)*R_-((2*np.pi*kx)**2)*(alpha**2)
    s_e_ft = (R_+2*np.pi*kx*1j*alpha)*m_ft/c0
    return ifft2(np.nan_to_num(s_e_ft*ind(k,1e-4))).real

def compute_u(h,s,k,kx,z):
    # function for computing the horizontal velocity in the x direction (u)
    # at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    u_ft = ((1j*2*np.pi*kx)/(2*np.pi*k)**2)*(Uh(k,z)*h_ft+Us(k,z)*delta*s_ft)
    return ifft2(u_ft).real

def compute_v(h,s,k,ky,z):
    # function for computing the horizontal velocity in the y direction (v)
    # at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    v_ft = ((1j*2*np.pi*ky)/(2*np.pi*k)**2)*(Uh(k,z)*h_ft+Us(k,z)*delta*s_ft)
    return ifft2(v_ft).real

def compute_w(h,s,k,z):
    # function for computing the vertical velocity (w) at a depth z
    h_ft = fft2(h)
    s_ft = fft2(s)
    w_ft = Wh(k,z)*h_ft + Ws(k,z)*delta*s_ft
    return ifft2(w_ft).real
