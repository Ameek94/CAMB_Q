import sys, os, time
import numpy as np
from scipy.optimize import basinhopping
from scipy.stats.qmc import Sobol
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import pybobyqa
import faulthandler; faulthandler.enable()


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
    phis = order_transform(params[1:nspline-1], reverse=False)
    Vs = order_transform(params[nspline-1:2*(nspline-1)], reverse=True)
    x_phys = np.concatenate([[params[0]], phis, Vs, params[2*(nspline-1):]])
    return input_unstandardize(x_phys, param_bounds)


def loglikelihood(x, cobaya_model=None, param_list=None, param_bounds=None):
    x_phys = prior(x, param_bounds)
    pdict = dict(zip(param_list, x_phys)) # type: ignore
    res = cobaya_model.loglike(pdict, make_finite=True, return_derived=False)
    return max(res, -1e5)


def make_monitor(fun, print_every=100):
    state = { 'n_calls': 0, 'best_f': float('inf'), 'best_x': None }
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
    if len(sys.argv)<3:
        print("Usage: python file.py <solver> <maxfun> [nrestart]")
        sys.exit(1)

    solver, maxfun = sys.argv[1], int(sys.argv[2])
    nrestart = int(sys.argv[3]) if len(sys.argv)>3 else 5

    start = time.time()
    log("Loading model...", start)
    info = yaml_load('./spline_4_vector_cmb_lite.yaml')
    # print(info)

    cobaya_model = get_model(info)
    param_list = list(cobaya_model.parameterization.sampled_params())
    param_bounds = np.array(cobaya_model.prior.bounds(confidence_for_unbounded=0.9999995)).T

    # generate initial x0 in unit cube
    x0 = Sobol(d=len(param_list)).random(n=32)
    prev_best = [0.63488053, 0.63698126, 0.85711982, 0.29146541, 0.09060491, 0.78680306,
                  0.37451744, 0.48230537, 0.42946419]
    val = loglikelihood(np.array(prev_best), cobaya_model=cobaya_model, param_list=param_list, param_bounds=param_bounds)
    inits = [prev_best] # Start with a known good point
    vals = [val]
    i = 0
    for x in x0:
        val = loglikelihood(x, cobaya_model=cobaya_model, param_list=param_list, param_bounds=param_bounds)
        if val > -1e5:
            print(f"Got loglike {val:.4f} for x = {x}\n")
            phys_x = prior(x, param_bounds)
            phys_x_dict = {k: f"{float(v):.6f}" for k, v in zip(param_list, phys_x)}
            print(f"Initial point: {phys_x_dict} with loglike = {val:.4f}\n")
            inits.append(x)
            vals.append(val)
            i += 1
            if i >= 15:  # Limit to 15 initial points
                break
            # break

    best_idx = np.argmax(vals)
    initial_x = np.array(inits[best_idx]) if inits else np.random.uniform(0.0, 1.0, size=len(param_list))
    phys_x = prior(initial_x, param_bounds)
    phys_x_dict = {k: f"{float(v):.6f}" for k, v in zip(param_list, phys_x)}
    print(f"Best initial point: {phys_x_dict} with loglike = {vals[best_idx]:.4f}\n")

    best_f, best_x = -np.inf, None
    log(f"Starting {nrestart} sequential restarts", start)
    for i in range(1, nrestart+1):
        log(f"Restart {i}/{nrestart}, x0 = {dict(zip(param_list, prior(initial_x,param_bounds)))}", start)
        if solver=='scipy':
            fval, xnew = solve_basinhopping(initial_x, loglikelihood, maxfun,
                                           args=(), kwargs={'param_bounds':param_bounds,'cobaya_model':cobaya_model,'param_list':param_list})
        elif solver=='bobyqa':
            fval, xnew = solve_bobyqa(initial_x, loglikelihood, maxfun,
                                      args=(), kwargs={'param_bounds':param_bounds,'cobaya_model':cobaya_model,'param_list':param_list})
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if fval>best_f:
            best_f, best_x = fval, xnew.copy()
            log(f"New best f = {best_f:.6f}", start)
            log(f"Best x = {dict(zip(param_list, prior(best_x,param_bounds)))}, start")

        initial_x = best_x.copy()

    # summarize
    best_phys = prior(best_x, param_bounds)
    pdict = dict(zip(param_list, best_phys))
    log(f"Global best f = {best_f:.6f}", start)
    print("Best params:")
    for k,v in pdict.items(): print(f"  {k} = {v:.6f}")

    # save
    fname = f"best_result_{solver}_{maxfun}.txt"
    old = load_saved_best(fname)
    if old is None or best_f>old:
        save_best(fname, best_f, param_list, best_phys)
    else:
        log(f"Best {best_f:.6f} <= stored {old:.6f}, not updating.", start)

if __name__=='__main__':
    main()
