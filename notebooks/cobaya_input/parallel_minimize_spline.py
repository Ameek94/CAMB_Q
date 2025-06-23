# mpi_opt.py
import sys, os, time
import numpy as np
from scipy.optimize import basinhopping
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import pybobyqa
from mpi4py import MPI
import faulthandler; faulthandler.enable()

def log(msg,start_time=None, rank=0):
    # include rank and relative time
    now = time.time() - start_time
    print(f"[{rank:2d} | +{now:6.1f}s] {msg}", flush=True)

def load_saved_best(filename):
    """
    Reads the first line of filename, expecting it to be
      logpost = <float>
    Returns the float value, or None if the file does not exist
    or is malformed.
    """
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            first = f.readline().strip()
        tag, val = first.split("=", 1)
        return float(val)
    except Exception:
        return None

def save_best(filename, best_f, param_labels, best_point):
    """
    Overwrites filename with the new best_f and parameters.
    """
    with open(filename, "w") as f:
        f.write(f"logpost = {best_f:.6f}\n\n")
        f.write(" ".join(param_labels) + "\n")
        f.write(" ".join(f"{v:.6g}" for v in best_point) + "\n")
    print(f"[Rank 0] Wrote new best result to {filename}.")

def input_standardize(x,param_bounds):
    """
    Project from original domain to unit hypercube, X is N x d shaped, param_bounds are 2 x d
    """
    x =  (x - param_bounds[0])/(param_bounds[1] - param_bounds[0])
    return x

def input_unstandardize(x,param_bounds):
    """
    Project from unit hypercube to original domain, X is N x d shaped, param_bounds are 2 x d
    """
    x = x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]
    return x


def loglikelihood(x,param_bounds,cobaya_model):
    x = input_unstandardize(x, param_bounds)
    # try:
    res = cobaya_model.loglike(x, make_finite=True,return_derived=False)
    if res < -1e5:
        res = -1e5
    # except BaseException as e:
    #     res = -1e5
    return res

def make_monitoring_func(fun, worker_id, print_every=100):
    state = {
        'n_calls': 0,
        'best_f': float('inf'),
        'best_x': None,
    }
    def wrapped(x):
        f = fun(x)
        state['n_calls'] += 1
        # update best
        if f < state['best_f']:
            state['best_f'] = f
            state['best_x'] = x.copy()
        # periodic print
        if state['n_calls'] % print_every == 0:
            print(
                f"[Worker {worker_id}] "
                f"Eval #{state['n_calls']:4d} | "
                f"Current f = {f:.6g} | "
                f"Best f = {state['best_f']:.6g}"
            )
        return f
    return wrapped

def solve_basinhopping(x0, raw_func, maxfun, args = (), kwargs = {}, init_parallel=False,worker_id=0):

    # bind the extra args so fun(x) works correctly:
    func_with_args = lambda x: -raw_func(x, *args,**kwargs)
    bounds = [(0., 1.0)]*len(x0)

    # wrap the original objective
    monitored_func = make_monitoring_func(func_with_args, worker_id=worker_id, print_every=10)

    res = basinhopping(monitored_func,
                       x0=x0,
                       niter=1000,
                       stepsize=0.4,
                       minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxfun': maxfun}})
    
    print(f"Worker {worker_id} finished with best val = {-res.fun:.4f}.")

    return -res.fun, res.x

def worker_solve_bobyqa(x0, raw_func, maxfun, args = (), kwargs = {},
                        seek_global_minimum = True, init_parallel=False,worker_id=0):

    # bind the extra args so fun(x) works correctly:
    func_with_args = lambda x: -raw_func(x, *args,**kwargs)

    bounds = [[0.]*len(x0), [1.0]*len(x0)]  # bounds for each parameter
    print(f"Bounds: {bounds}")

    # wrap the original objective
    monitored_func = func_with_args #make_monitoring_func(func_with_args, worker_id=0, print_every=4)

    res = pybobyqa.solve(
        monitored_func,
        x0=x0,
        bounds=bounds,
        maxfun=maxfun,
        seek_global_minimum=seek_global_minimum,
        user_params={'init.run_in_parallel': init_parallel},
        print_progress=True,
    )
    print(f"Worker {worker_id} finished with {res.nf} function evaluations, best val = {res.f:.4f}.")

    return -res.f, res.x

def main():

    # check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python parallel_minimize_spline.py <solver> [maxfun]")
        print("Available solvers: 'scipy', 'bobyqa'")
        sys.exit(1)

    # --- MPI setup ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()



    # --- All ranks load the model independently ---
    # input_file = './spline_4_free_external.yaml'
    input_file = './spline_4_fixed_external.yaml'

    info = yaml_load(input_file)
    cobaya_model = get_model(info)
    confidence_for_unbounded = 0.9999995
    param_list = list(cobaya_model.parameterization.sampled_params())
    if rank == 0:
        print(f"Parameter list: {param_list}")
    # parameter setup
    param_bounds = np.array(
        cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
    ).T

    comm.Barrier()
    start_time = time.time()
    log("startup complete", start_time=start_time, rank=rank)


    # sample a pool of initial points on rank 0…
    if rank == 0:
        seeds = np.random.SeedSequence(12345).spawn(size-1)
        # param_dict = {'lengthscale': 0.197173, 'phi2': 0.125951, 'phi3': 0.056194, 'phi4': 0.1923, 'V2': 0.999935, 'V3': 0.774321, 'V4': 0.940254, 'omch2': 0.119289, 'ombh2': 0.022536, 'H0': 66.523051}
        param_dict = {'lengthscale': 0.123939, 'phi2': '0.283628', 'phi3': '0.421014', 'V2': '0.443599', 'V3': '0.705376', 'V4': '0.019240', 'omch2': '0.565322', 'ombh2': '0.333276', 'H0': '0.669330'}
        param_dict = {k: float(v) for k, v in param_dict.items() if k in param_list}
        val = cobaya_model.loglike(param_dict, make_finite=True, return_derived=False)
        inits = [input_standardize(np.array(list(param_dict.values())), param_bounds)]
        print(f"Initial point: {param_dict} with loglike = {val} at rank {rank}")
        for ss in seeds:
            rng = np.random.default_rng(ss)
            x0, val = cobaya_model.get_valid_point(1000, ignore_fixed_ref=False,
                                                   logposterior_as_dict=True)
            init_pt_dict = {k: f"{float(val):.4f}" for k, val in zip(param_list, x0)}
            print(f"Initial point: {init_pt_dict} with loglike = {val['loglikes']} at rank {rank}")
            inits.append(input_standardize(x0, param_bounds))
    else:
        inits = None

    # …then scatter one initial point to each rank
    x0_std = comm.scatter(inits, root=0)

    # pick solver from command‑line
    solver = str(sys.argv[1]) 
    maxfun = int(sys.argv[2]) 

    comm.Barrier()
    log("entering optimizer", start_time=start_time, rank=rank)

    if rank > 0:
        if solver == 'scipy':
            local_f, local_x = solve_basinhopping(
                x0=x0_std, raw_func=loglikelihood, maxfun=maxfun,
                kwargs={'param_bounds':param_bounds,'cobaya_model': cobaya_model},
                worker_id=rank)
        elif solver == 'bobyqa':
            local_f, local_x = worker_solve_bobyqa(
                x0=x0_std, raw_func=loglikelihood, maxfun=maxfun,
                kwargs={'param_bounds':param_bounds,'cobaya_model': cobaya_model},
                worker_id=rank)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    else:
        # Dummy result for rank 0
        local_f, local_x = -np.inf, np.zeros_like(x0_std)


    # # run the local optimization
    # if solver == 'scipy':
    #     local_f, local_x = solve_basinhopping(
    #         x0=x0_std,
    #         raw_func=loglikelihood,
    #         maxfun=maxfun,
    #         kwargs={'param_bounds': param_bounds,
    #                 'cobaya_model': cobaya_model},
    #         init_parallel=False,
    #         worker_id=rank
    #     )
    # elif solver == 'bobyqa':
    #     local_f, local_x = worker_solve_bobyqa(
    #         x0=x0_std,
    #         raw_func=loglikelihood,
    #         maxfun=maxfun,
    #         kwargs={'param_bounds': param_bounds,
    #                 'cobaya_model': cobaya_model},
    #         init_parallel=False,
    #         worker_id=rank
    #     )
    # else:
    #     raise ValueError(f"Unknown solver: {solver}")

    comm.Barrier()
    log("ended optimizer", start_time=start_time, rank=rank)


    # before gather:
    comm.Barrier()
    log("about to gather", start_time=start_time, rank=rank)

    # gather all (f, x) pairs to rank 0
    all_results = comm.gather((local_f, local_x), root=0)

    if rank == 0:
        # pick the global best
        print(f"\n=== RANK {rank} RESULTS ===")
        for i, (f, x) in enumerate(all_results):
            point = input_unstandardize(x, param_bounds)
            point_dict = {k: f"{float(val):.4f}" for k, val in zip(param_list, point)}
            print(f"Worker {i}: loglike = {f:.6f}, params = {point_dict}")

        best_f, best_x = max(all_results, key=lambda fx: fx[0])
        best_point = input_unstandardize(best_x, param_bounds)
        print(f"\n=== GLOBAL BEST ===")
        print(f"loglike = {best_f:.6f}")
        best_point_dict = {k: f"{float(val):.6f}" for k, val in zip(param_list, best_point)}
        print(f"Best param_dict = {best_point_dict}")
        # print(f"params  = {best_point}")

        filename = f"best_result_{solver}_{maxfun}.txt"

        # 1) load any previously saved best
        old_f = load_saved_best(filename)
        if old_f is None:
            print("[Rank 0] No existing result; saving this one.")
            save_best(filename, best_f, param_list, best_point)
        elif best_f > old_f:
            print(f"[Rank 0] New logpost {best_f:.6f} > old {old_f:.6f}; updating file.")
            save_best(filename, best_f, param_list, best_point)
        else:
            print(f"[Rank 0] Current best {best_f:.6f} is not better than stored {old_f:.6f}; leaving file unchanged.")



if __name__ == "__main__":
    main()
