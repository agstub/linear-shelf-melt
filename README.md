# linear-shelf-melt
small-perturbation model for sub-ice-shelf channels and keels

This repository contains a simple model for ice-shelves undergoing localized basal melting or freezing perturbations. A derivation and analysis of the model is provided in the "notes" directory. The figures in the notes can be reproduced with the Jupyter notebooks in the "notebooks" directory. 

The "code" directory contains:
1. **params.py**: sets the physical and numerical parameters
2. **kernel_fcns.py**: defines the functions that appear in the solution operators
3. **operators.py**: defines steady-state and time-dependent solution operators
4. **algebra.py**: uses SymPy to derive some of the functions appearing in the solution operators (this is best read alongside the notes).  

Examples for running the code are provided in the Jupyter notebooks ("notebooks" directory).
