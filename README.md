# linear-shelf-melt
[![DOI](https://zenodo.org/badge/455572763.svg)](https://zenodo.org/badge/latestdoi/455572763)

Small-perturbation (linearized) model for topographic response of ice shelves to localized melting. Also includes finite element code to solve
the fully nonlinear problem. This code accompanies the manuscript:

> Stubblefield, A.G., Wearing, M.G., & Meyer, C.R. (2023). Linear analysis of ice-shelf topography response to basal melting and freezing. https://eartharxiv.org/repository/view/5330/ 

# Running the code
This repository contains a simple model for ice-shelves undergoing localized basal melting or freezing perturbations. 
The figures in the manuscript can be reproduced with the Jupyter notebooks in the "notebooks" directory.
A derivation and analysis of the model is outlined in the first two notebooks.  The final notebook "4_nonlinear.ipynb" relies on FEniCSx (https://fenicsproject.org) code in the "nonlinear-model" directory. This notebook can be run via Docker (https://www.docker.com) with the command:
`docker run --init -ti -p 8888:8888 -v $(pwd):/home/fenics/shared -w /home/fenics/shared dolfinx/lab:stable`

# Code organization
## Linear model
The "linear-model" directory contains:
1. **params.py**: sets the physical and numerical parameters
2. **kernel_fcns.py**: defines the functions that appear in the solution operators
3. **operators.py**: defines steady-state and time-dependent solution operators

Examples for running the code are provided in the Jupyter notebooks ("notebooks" directory).

## Nonlinear model
The "nonlinear-model" directory is essentially a fork of the repo https://github.com/agstub/grounding-line-methods that has been translated into Dolfinx.
The model is organized in 6 python files in the *source* directory as follows.

1. **params.py** contains all of the model parameters and model options.

3. **stokes.py** contains the Stokes system solver and related functions.

4. **mesh_routine.py** contains functions that solve the surface kinematic equations, move the mesh,
    and some post-processing functions.

5. **bdry_conds.py** contains functions that mark the mesh boundary and apply boundary conditions.

6. **smb.py** contains surface mass balance functions: basal melting/freezing rate
and surface accumulation/ablation rate.

7. **main.py** runs the model. It contains the time-stepping loop that
calls the Stokes solver and mesh-related functions at each timestep, and returns the output.
