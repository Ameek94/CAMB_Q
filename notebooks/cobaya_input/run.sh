#!/bin/bash

# cobaya-run spline_5.yaml --resume

# cobaya-run spline_5.yaml --minimize

# mpirun -n 2 cobaya-run spline_4_free.yaml -r

# cobaya-run spline_4_free_minimize.yaml

export OMP_NUM_THREADS=8

python minimize_spline.py bobyqa 250 25

python minimize_spline.py scipy 200 10

python nautilus_spline.py