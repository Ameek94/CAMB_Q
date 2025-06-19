#!/bin/bash

# cobaya-run spline_5.yaml --resume

# cobaya-run spline_5.yaml --minimize

mpirun -n 2 cobaya-run spline_4_free.yaml -r

cobaya-run spline_4_free_minimize.yaml