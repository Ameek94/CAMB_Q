#!/bin/bash

# cobaya-run spline_5.yaml --resume

# cobaya-run spline_5.yaml --minimize

# mpirun -n 2 cobaya-run spline_4_free.yaml -r

# cobaya-run spline_4_free_minimize.yaml

# export OMP_NUM_THREADS=8

cobaya-run Planck_lite_BAO_SN_CPL.yaml

cobaya-run Planck_lite_BAO_SN_LCDM.yaml

# python spline_minimize_only_DE.py bobyqa --maxfun 500 --nrestart 4

# python spline_minimize_only_DE.py scipy --maxfun 500 --nrestart 4

# python minimize_spline.py bobyqa 250 25

# python minimize_spline.py scipy 200 10

# python nautilus_spline.py