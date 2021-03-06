# linear-shelf-melt
small-perturbation model for sub-ice-shelf channels and keels

This repository contains a simple model for ice-shelves undergoing localized basal melting or freezing perturbations. A derivation and analysis of the model is provided in the "notes" directory. The figures in the notes can be reproduced with the Jupyter notebooks in the "notebooks" directory.

The "code" directory contains:
1. **params.py**: sets the physical and numerical parameters
2. **kernel_fcns.py**: defines the functions that appear in the solution operators
3. **operators.py**: defines steady-state and time-dependent solution operators
4. **algebra.py**: uses SymPy to derive some of the functions appearing in the solution operators (this is best read alongside the notes).  

The code dependencies are listed in requirements.txt.

Examples for running the code are provided in the Jupyter notebooks ("notebooks" directory).

The Jupyter notebooks can be run with Binder here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/agstub/linear-shelf-melt/HEAD?labpath=%2Fnotebooks%2F1_math.ipynb)
(Note: to run the second notebook in Binder, make sure the other notebooks are not running. Otherwise the notebook will run out of memory)
