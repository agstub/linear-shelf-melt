#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCSx-see README
#------------------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import smb
from dolfinx import io
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.mesh import create_rectangle, refine
from mesh_routine import get_surfaces, get_vel, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nt, save_vtk, t_f, t_r
from scipy.signal import convolve
from stokes import stokes_solve,eta


def solve(a,m):

    # generate mesh
    p0 = [-L/2.0,0.0]
    p1 = [L/2.0,H]
    domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

    # Define arrays for saving surfaces
    h_i,s_i,x = get_surfaces(domain)
    nx = x.size
    h = np.zeros((nx,nt))
    s = np.zeros((nx,nt))

    t = np.linspace(0,t_f, nt)
    # # Begin time stepping
    for i in range(nt):

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        t_i = t[i]

        # Solve the Stoke problem for w = (u,p)
    
        sol = stokes_solve(domain)
        if i == 0:
            u,w,x_vel,z_vel0 = get_vel(sol,domain)

        # # uncomment to print the mean and minimum of the viscosity: 
        # eta_expr = eta(sol.sub(0))
        # V = FunctionSpace(domain, ("CG", 1))
        # eta_fcn = Function(V)
        # eta_fcn.interpolate(Expression(eta_expr, V.element.interpolation_points()))
        # eta_mean = eta_fcn.x.array.mean()
        # eta_min = eta_fcn.x.array.min()
        # print('i='+str(i+1)+': eta_mean = '+'{:.2e}'.format(eta_mean)+',    eta_min = '+'{:.2e}'.format(eta_min)+'\r',end='')
       


        # Move the mesh 
        domain = move_mesh(sol,domain,t_i,a,m)
       
        h_i,s_i,x = get_surfaces(domain)
  
        h[:,i] = h_i
        s[:,i] = s_i

        # save the stokes solution 
        if save_vtk == True:
            with io.VTXWriter(domain.comm, "../results/output.bp", [sol.sub(0).sub(1)]) as vtx:
                vtx.write(0.0)

    # extract the velocity solution
    u,w,x_vel,z_vel = get_vel(sol,domain)
    
    return h,s,u,w,x,x_vel,z_vel0