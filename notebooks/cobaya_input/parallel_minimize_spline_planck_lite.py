# mpi_spline_opt.py
import argparse
import sys, os
import time

import numpy as np
from scipy.optimize import basinhopping
from scipy.stats.qmc import Sobol
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import pybobyqa
import faulthandler; faulthandler.enable()
from plot_utils import plot_spline_quintessence

# --- MPI import ---
from mpi4py import MPI

def log(msg, start_time=None):
    """ Simple logger with elapsed time. """
    now = time.time() - start_time if start_time else 0.0
    print(f"[+{now:6.1f}s] {msg}", flush=True)

def load_saved_best(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            first = f.readline().strip()
        _, val = first.split("=", 1)
        return float(val)
    except Exception:
        return None

def save_best(filename, best_f, param_labels, best_point):
    with open(filename, "w") as f:
        f.write(f"logpost = {best_f:.6f}\n\n")
        f.write(" ".join(param_labels) + "\n")
        f.write(" ".join(f"{v:.6g}" for v in best_point) + "\n")
    print(f"Wrote new best result to {filename}.")

def order_transform(unit_x, low=0., high=1., reverse=False):
    n = np.size(unit_x, axis=-1)
    idx = np.arange(n)
    term = np.power(1 - unit_x, 1/(n - idx))
    y = 1 - np.cumprod(term, axis=-1)
    if reverse:
        y = y[::-1]
    return y * (high - low) + low

def inverse_order_transform(unit_y, low=0., high=1., reverse=False):
    if reverse:
        unit_y = unit_y[::-1]
    n = np.size(unit_y, axis=-1)
    idx = np.arange(n)
    shifted = np.roll(unit_y, 1)
    shifted[0] = 0
    x = 1 - np.power((1 - unit_y)/(1 - shifted), n - idx)
    return x * (high - low) + low

def input_standardize(x, param_bounds):
    return (x - param_bounds[0])/(param_bounds[1] - param_bounds[0])

def input_unstandardize(x, param_bounds):
    return x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]

def prior(x, param_bounds, nspline=4):
    params = x.copy()
    phis = order_transform(params[1:nspline], reverse=False)
    Vs = order_transform(params[nspline:2*(nspline-1) + 1], reverse=True)
    x_phys = np.concatenate([[params[0]], phis, Vs, params[2*(nspline-1)+1:]])
    return input_unstandardize(x_phys, param_bounds)

def inverse_prior(x_phys, param_bounds, nspline=4):
    x = input_standardize(x_phys, param_bounds)
    phis = inverse_order_transform(x[1:nspline], reverse=False)
    Vs = inverse_order_transform(x[nspline:2*(nspline-1) + 1], reverse=True)
    x_inv = np.concatenate([[x[0]], phis, Vs, x[2*(nspline-1)+1:]])
    return x_inv

def loglikelihood(x, cobaya_model=None, param_list=None, param_bounds=None, nspline=4):
    try:
        x_phys = prior(x, param_bounds, nspline=nspline)
        pdict = dict(zip(param_list, x_phys))  # type: ignore
        res = cobaya_model.loglike(pdict, make_finite=True, return_derived=False)
    except Exception as e:
        print(f"Error in loglikelihood: {e}", file=sys.stderr)
        return -1e5
    return max(res, -1e5)

def make_monitor(fun, print_every=100):
    state = {'n_calls': 0, 'best_f': float('inf'), 'best_x': None}
    def wrap(x):
        f = fun(x)
        state['n_calls'] += 1
        if f < state['best_f']:
            state['best_f'], state['best_x'] = f, x.copy()
        if state['n_calls'] % print_every == 0:
            print(f"Eval #{state['n_calls']:4d} | f = {f:.6g} | best = {state['best_f']:.6g}")
        return f
    return wrap

def solve_basinhopping(x0, raw_func, maxfun, args=(), kwargs=None):
    kwargs = kwargs or {}
    func = lambda x: -raw_func(x, *args, **(kwargs or {}))
    bounds = [(0.,1.)] * len(x0)
    mon = make_monitor(func, print_every=10)
    res = basinhopping(
        mon, x0=x0, niter=1000, stepsize=0.4,
        minimizer_kwargs={
            'method': 'L-BFGS-B', 'bounds': bounds,
            'options': {'maxfun': maxfun}
        }
    )
    return -res.fun, res.x

def solve_bobyqa(x0, raw_func, maxfun, args=(), kwargs=None):
    kwargs = kwargs or {}
    func = lambda x: -raw_func(x, *args, **(kwargs or {}))
    bounds = [[0.]*len(x0), [1.]*len(x0)]
    res = pybobyqa.solve(
        func, x0=x0, bounds=bounds, maxfun=maxfun,
        seek_global_minimum=True, user_params={'init.run_in_parallel': False}
    )
    return -res.f, res.x

def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description="Run the solver with MPI‐parallel restarts."
    )
    parser.add_argument('solver', help="Name of the solver to use.")
    parser.add_argument('--maxfun', type=int, default=None,
                        help="Maximum number of function evaluations.")
    parser.add_argument('--nrestart', type=int, default=5,
                        help="Number of restarts.")
    args = parser.parse_args()

    solver   = args.solver
    maxfun   = args.maxfun
    nrestart = args.nrestart

    if rank == 0:
        print(f"Solver:   {solver}")
        print(f"Maxfun:   {maxfun!r}")
        print(f"Restarts: {nrestart}")

    nspline = 4
    start = time.time()
    if rank == 0:
        log("Loading model...", start)
    info = yaml_load(f'./spline_{nspline}_vector_cmb_all.yaml')
    cobaya_model = get_model(info)

    param_list   = list(cobaya_model.parameterization.sampled_params())
    param_bounds = np.array(
        cobaya_model.prior.bounds(confidence_for_unbounded=0.9999995)
    ).T

    if rank == 0:
        print(f"Parameter list:   {param_list}")
        print(f"Parameter bounds: {param_bounds.T}")

    # 1) Each rank builds its own initial-pool, seeded by rank
    seed = 12345 + rank
    rng = np.random.default_rng(seed)
    sob = Sobol(d=len(param_list), scramble=True, rng=rng)
    pool = sob.random(n=4)
    inits, vals = [], []
    for x in pool:
        v = loglikelihood(
            x,
            cobaya_model=cobaya_model,
            param_list=param_list,
            param_bounds=param_bounds,
            nspline=nspline
        )
        if v > -750:
            inits.append(x)
            vals.append(v)
            if len(inits) >= nrestart:
                break
    # if not enough valid points, pad with uniform
    while len(inits) < nrestart:
        inits.append(rng.random(len(param_list)))

    # 2) broadcast the list of inits to all ranks
    inits = comm.bcast(inits, root=0)
    # ensure length = nrestart
    inits = inits[:nrestart]

    # 3) each rank does its share of restarts
    local_best_f = -np.inf
    local_best_x = None

    for i in range(rank, nrestart, size):
        x0 = inits[i]
        if solver == 'scipy':
            fval, xnew = solve_basinhopping(
                x0, loglikelihood, maxfun,
                args=(), kwargs={
                    'param_bounds': param_bounds,
                    'cobaya_model': cobaya_model,
                    'param_list': param_list,
                    'nspline': nspline
                }
            )
        elif solver == 'bobyqa':
            fval, xnew = solve_bobyqa(
                x0, loglikelihood, maxfun,
                args=(), kwargs={
                    'param_bounds': param_bounds,
                    'cobaya_model': cobaya_model,
                    'param_list': param_list,
                    'nspline': nspline
                }
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if fval > local_best_f:
            local_best_f, local_best_x = fval, xnew.copy()

    # 4) gather all local bests to rank 0
    gathered = comm.gather((local_best_f, local_best_x), root=0)

    if rank == 0:
        # pick the global best
        best_f, best_x = max(gathered, key=lambda fx: fx[0]) # type: ignore
        best_phys = prior(best_x, param_bounds, nspline=nspline)
        pdict = dict(zip(param_list, best_phys))

        log(f"Global best f = {best_f:.6f}", start)
        print("Best params:")
        for k,v in pdict.items():
            print(f"  {k} = {v:.6f}")

        # save to file if improved
        fname = f"best_result_{nspline}_{solver}_{maxfun}_planck_lite.txt"
        old = load_saved_best(fname)
        if old is None or best_f > old:
            save_best(fname, best_f, param_list, best_phys)
        else:
            log(f"Best {best_f:.6f} <= stored {old:.6f}, not updating.", start)

        # final plot
        plot_spline_quintessence(
            param_dict=pdict,
            nspline=nspline,
            output_prefix="SplineQ_planck_lite"
        )

if __name__ == '__main__':
    main()