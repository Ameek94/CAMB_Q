import argparse
import sys, os, time
import numpy as np
from scipy.optimize import basinhopping
from scipy.stats.qmc import Sobol
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import pybobyqa
import faulthandler; faulthandler.enable()
from plot_utils_CMB import plot_spline_quintessence

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
    """ Transform the input x in the unit cube to the physical parameter space. """
    params = x.copy()
    # phis = order_transform(params[1:nspline], reverse=False)
    # Vs = order_transform(params[nspline:2*(nspline-1) + 1], reverse=True)
    # x_phys = np.concatenate([[params[0]], phis, Vs, params[2*(nspline-1)+1:]])
    return input_unstandardize(params, param_bounds)

def inverse_prior(x_phys, param_bounds, nspline=4):
    """ Transform x_phys from physical parameter space back to the unit cube. """
    x = input_standardize(x_phys, param_bounds)
    # phis = inverse_order_transform(x[1:nspline], reverse=False)
    # Vs = inverse_order_transform(x[nspline:2*(nspline-1) + 1], reverse=True)
    # x_inv = np.concatenate([[x[0]], phis, Vs, x[2*(nspline-1)+1:]])
    return x #x_inv

def loglikelihood(x, cobaya_model=None, param_list=None, param_bounds=None, nspline=4):
    # try:
    x_phys = prior(x, param_bounds, nspline=nspline)
    pdict = dict(zip(param_list, x_phys)) # type: ignore
    res = cobaya_model.loglike(pdict, make_finite=True, return_derived=False)
    # except Exception as e:
    #     print(f"Error in loglikelihood: {e}", file=sys.stderr)
    #     return -1e5
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
    parser = argparse.ArgumentParser(
        description="Run the solver with an optional maxfun and optional restarts."
    )
    # still a required positional
    parser.add_argument(
        'solver',
        help="Name of the solver to use."
    )
    # now flagged options for the others
    parser.add_argument(
        '--maxfun',
        type=int,
        default=None,
        help="Maximum number of function evaluations (default: None)."
    )
    parser.add_argument(
        '--nrestart',
        type=int,
        default=5,
        help="Number of restarts (default: 5)."
    )

    args = parser.parse_args()

    solver = args.solver
    maxfun = args.maxfun
    nrestart = args.nrestart

    # your code here...
    print(f"Solver:   {solver}")
    print(f"Maxfun:   {maxfun!r}")
    print(f"Restarts: {nrestart}")

    nspline = 6
    start = time.time()
    log("Loading model...", start)
    info = yaml_load(f'./spline_{nspline}_varyDEonly_camspec.yaml')
    # print(info)

    cobaya_model = get_model(info)
    param_list = list(cobaya_model.parameterization.sampled_params())
    param_bounds = np.array(cobaya_model.prior.bounds(confidence_for_unbounded=0.9999995)).T
    print(f"Parameter list: {param_list}")
    print(f"Parameter bounds: {param_bounds.T}")


    # init_dict = {'theta_i': 0.010243622782147037, 'omch2': 0.11980918131816978, 'ombh2': 0.022125090901215776, 'logA': 3.018102889940497, 'ns': 0.9672356987174313, 'H0': 65.41353581250422, 'tau': 0.04590253114007514,
    #              'A_planck': 0.9965852029071789, 'amp_143': 14.409007817323188, 'amp_217': 8.395881351576028, 'amp_143x217': 5.222027134842557, 'n_143': 1.173135564487783, 'n_217': 2.3545208086890126, 'n_143x217': 3.1000625465248763,
    #              'calTE': 1.0014945843015626, 'calEE': 1.0005730015262329}

    # init_dict = {'theta_i': 0.0009885953668691085, 'omch2': 0.11964380262588728, 'ombh2': 0.022138277627647945, 'logA': 3.01750523334186, 'ns': 0.9680009240856203, 'H0': 65.49170624708837, 'tau': 0.04613197783614444,
    #              'A_planck': 0.9964871694653109, 'amp_143': 14.555952934200967, 'amp_217': 8.652267529192365, 'amp_143x217': 5.470887868271735, 'n_143': 1.1217893731916149, 'n_217': 2.377302487384885, 'n_143x217': 2.9966079469388776,
    #              'calTE': 1.0006569941203427, 'calEE': 0.9996893207900899}

    # init_dict = {'theta_i': 0.0010099598757542928, 'omch2': 0.11962882212591071, 'ombh2': 0.022144114308432956, 'logA': 3.0177536395655737, 'ns': 0.967628742345485, 'H0': 65.48668908639127, 'tau': 0.04616516936868534,
    #              'A_planck': 0.9964958570822067, 'amp_143': 14.575008515614751, 'amp_217': 8.634454965611079, 'amp_143x217': 5.500131189307504, 'n_143': 1.1370519671640238, 'n_217': 2.2991420323633434, 'n_143x217': 2.9875168606080287,
    #              'calTE': 1.000495624737915, 'calEE': 0.9994876218820478}

    # init_dict = {'theta_i': 0.0010342121599348822, 'omch2': 0.1195772749450178, 'ombh2': 0.022155410535431425, 'logA': 3.0179798179645028, 'ns': 0.9675829876162794, 'H0': 65.48309596009558, 'tau': 0.046237407165564105, 'A_planck': 0.996536743217406, 'amp_143': 14.699369781271038, 'amp_217': 8.650486064354487, 'amp_143x217': 5.511177030385019, 'n_143': 1.1545521423283, 'n_217': 2.3008298782296595, 'n_143x217': 2.9423147473081683, 'calTE': 1.0002098871546643, 'calEE': 0.999388584783335}
    # camspec_dict = {
    # 'A_planck': 1.000945912,
    # 'amp_143': 18.90984584,
    # 'amp_217': 12.91572380,
    # 'amp_143x217': 9.802790167,
    # 'n_143': 0.8515425187,
    # 'n_217': 1.171328749,
    # 'n_143x217': 1.197817090,
    # 'calTE': 0.9970389173,
    # 'calEE': 0.9970339103
    # }
    # init_dict.update(camspec_dict)
    # init_dict['lengthscale'] =  0.09758613296694123

    # init_dict= {'theta_i': 0.0018322897564223852, 'omch2': 0.1191285987728817, 'ombh2': 0.022176933876205854, 'logA': 3.0161434343011306, 'ns': 0.9656884151982134, 'H0': 65.1496920706285, 'tau': 0.04280477991973508}

    # init_dict = {'theta_i': 0.0002831301783327904, 'omch2': 0.11781721822824441, 'ombh2': 0.022386695363833618, 'logA': 3.0517640806708104, 'ns': 0.9658905601530381, 'H0': 66.50135411326498, 'tau': 0.06151802778862622}
    # camspec_dict = {
    # 'A_planck': 1.000945912,
    # 'amp_143': 18.90984584,
    # 'amp_217': 12.91572380,
    # 'amp_143x217': 9.802790167,
    # 'n_143': 0.8515425187,
    # 'n_217': 1.171328749,
    # 'n_143x217': 1.197817090,
    # 'calTE': 0.9970389173,
    # 'calEE': 0.9970339103
    # }
    # init_dict.update(camspec_dict)

    phi_train = [0.,         0.05151515, 0.1030303,  0.15454545, 0.20606061, 0.25757576]
    V_train = [ 0.,         -0.03136744, -0.11752658, -0.23301456, -0.33374017, -0.36813454]

    init_dict = {f"phi{i+1}": phi for i, phi in enumerate(phi_train)}
    init_dict.update({f"V{i+1}": V for i, V in enumerate(V_train)})

    init_dict = {k: v for k, v in init_dict.items() if k in param_list}
    init_dict['lengthscale'] = 0.09758613296694123
    fixed_dict = {'theta_i': 0.00027267149410061794, 'omch2': 0.11784972610787393, 'ombh2': 0.022374480213129468, 'logA': 3.0518072593043093, 'ns': 0.966106880798031, 'H0': 66.49104389184664, 'tau': 0.06151589112478472, 'A_planck': 1.0010366935360766, 'amp_143': 18.99762773190983, 'amp_217': 12.928126350604574, 'amp_143x217': 9.81399847296, 'n_143': 0.8649054997605863, 'n_217': 1.1870886513227132, 'n_143x217': 1.2138697716380804, 'calTE': 0.9964663496117538, 'calEE': 0.9965020405605757}
    init_dict.update(fixed_dict)

    init = inverse_prior(np.array([init_dict[k] for k in param_list])
                        , param_bounds, nspline=nspline)
    val = loglikelihood(init, cobaya_model=cobaya_model, param_list=param_list,
                        param_bounds=param_bounds, nspline=nspline)
    print(f"Initial point: {init_dict} with loglike = {val:.4f}\n")

    inits, vals = [init], [val]

    n_init = 64

    cobaya_init = n_init
    # try:
    for i in range(cobaya_init):
        x0, logp_dict = cobaya_model.get_valid_point(max_tries=1000,logposterior_as_dict=True)
        unit_x = inverse_prior(x0, param_bounds,nspline=nspline)
        val = loglikelihood(unit_x, cobaya_model=cobaya_model, param_list=param_list,
                            param_bounds=param_bounds, nspline=nspline)
        # print(f"Got loglike {val:.4f} for x = {x0}\n")
        phys_x_dict = {k: f"{float(v):.6f}" for k, v in zip(param_list, x0)}
        print(f"Initial point: {phys_x_dict} with loglike = {val:.4f}\n")
        if val > -6000:
            inits.append(unit_x)
            vals.append(val)


    i = len(inits)
    print(f"Got {i-1} initial points from Cobaya model.\n")
    # generate initial x0 in unit cube
    x0 = Sobol(d=len(param_list)).random(n=n_init)
    for x in x0:
        val = loglikelihood(x, cobaya_model=cobaya_model, param_list=param_list,
                            param_bounds=param_bounds,nspline=nspline)
        phys_x = prior(x, param_bounds,nspline=nspline)
        phys_x_dict = {k: f"{float(v):.6f}" for k, v in zip(param_list, phys_x)}
        print(f"Initial point: {phys_x_dict} with loglike = {val:.4f}\n")
        if val > -6000:
            print(f"Got loglike {val:.4f} for x = {x}\n")
            inits.append(x)
            vals.append(val)
            i += 1
        if i>nrestart:  # Limit to nrestart initial points
            break
            # if i >= 15:  # Limit to 15 initial points
            #     break
            # break



    best_idx = np.argmax(vals)
    initial_x = np.array(inits[best_idx]) if inits else np.random.uniform(0.0, 1.0, size=len(param_list))
    phys_x = prior(initial_x, param_bounds,nspline=nspline)
    phys_x_dict = {k: f"{float(v):.6f}" for k, v in zip(param_list, phys_x)}
    print(f"Best initial point: {phys_x_dict} with loglike = {vals[best_idx]:.4f}\n")

    best_f, best_x = -np.inf, None

    idxs = np.argsort(vals)[::-1]
    inits = np.array(inits)[idxs]
    vals = np.array(vals)[idxs]
    print(f"Initial points sorted by loglike values: {vals}")

    # initial_x = prev.copy()  # Start with the previous best point
    log(f"Starting {nrestart} sequential restarts", start)
    for i in range(1, nrestart+1):
        log(f"Restart {i}/{nrestart}, x0 = {dict(zip(param_list, prior(initial_x,param_bounds,nspline=nspline)))}, start")
        if solver=='scipy':
            fval, xnew = solve_basinhopping(initial_x, loglikelihood, maxfun,
                                           args=(), kwargs={'param_bounds':param_bounds,'cobaya_model':cobaya_model,
                                           'param_list':param_list, 'nspline':nspline})
        elif solver=='bobyqa':
            fval, xnew = solve_bobyqa(initial_x, loglikelihood, maxfun,
                                      args=(), kwargs={'param_bounds':param_bounds,'cobaya_model':cobaya_model,
                                      'param_list':param_list, 'nspline':nspline})
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if fval>best_f:
            best_f, best_x = fval, xnew.copy()
            log(f"New best f = {best_f:.6f}", start)
            log(f"Best x = {dict(zip(param_list, prior(best_x,param_bounds,nspline=nspline)))}, start")

        try:
            initial_x = x0[i] # best_x.copy()
        except:
            initial_x = best_x.copy()
        if i==nrestart-1:
            initial_x = best_x.copy()  # Last iteration uses the best found point

    # summarize
    best_phys = prior(best_x, param_bounds,nspline=nspline)
    pdict = dict(zip(param_list, best_phys))
    log(f"Global best f = {best_f:.6f}", start)
    print("Best params:")
    for k,v in pdict.items(): print(f"{k} = {v:.6f}")

    # save
    fname = f"best_result_{nspline}_{solver}_{maxfun}_Camspec_all.txt"
    old = load_saved_best(fname)
    if old is None or best_f>old:
        save_best(fname, best_f, param_list, best_phys)
    else:
        log(f"Best {best_f:.6f} <= stored {old:.6f}, not updating.", start)

    # phis = [0.,      0.03330754, 0.06661507, 0.09992261, 0.13323014]
    # Vphis = [ 0.,         -0.01143734, -0.04149601, -0.08128815, -0.1194607 ] gp

    # pysr
    # phi_train = [0.,         0.05151515, 0.1030303,  0.15454545, 0.20606061, 0.25757576]
    # V_train = [ 0.,         -0.03136744, -0.11752658, -0.23301456, -0.33374017, -0.36813454]

    phi_train = [0.0] + [pdict[f'phi{i+1}'] for i in range(1,nspline)]
    V_train = [0.0] + [pdict[f'V{i+1}'] for i in range(1,nspline)]

    pdict['phi_train'] = phi_train
    pdict['V_train'] = V_train
    # for i in range(nspline):
    #     pdict.pop(f'phi{i}', None)  # Remove phi_i if it exists
    #     pdict.pop(f'V{i}', None)
    pdict.pop('A_planck', None)  # Remove A_planck if it exists
    pdict['As'] = 1e-10 * np.exp(pdict.pop('logA',3.045))  # Convert logA to As

    # fixed_dict = {'theta_i': 0.00027267149410061794, 'omch2': 0.11784972610787393, 'ombh2': 0.022374480213129468, 'logA': 3.0518072593043093, 'ns': 0.966106880798031, 'H0': 66.49104389184664, 'tau': 0.06151589112478472, 'A_planck': 1.0010366935360766, 'amp_143': 18.99762773190983, 'amp_217': 12.928126350604574, 'amp_143x217': 9.81399847296, 'n_143': 0.8649054997605863, 'n_217': 1.1870886513227132, 'n_143x217': 1.2138697716380804, 'calTE': 0.9964663496117538, 'calEE': 0.9965020405605757}
    # pdict.update(fixed_dict)

    # CPL results
    cpl_chi2_cmb = 1.097029896e+04
    cpl_chi2_bao = 1.187946432e+01
    cpl_chi2_sn = 2.143726844e+01
    cpl_chi2 = cpl_chi2_cmb + cpl_chi2_bao + cpl_chi2_sn

    #LCDM results
    lcdm_chi2_cmb = 1.097258134e+04
    lcdm_chi2_bao = 1.692238778e+01
    lcdm_chi2_sn = 2.719041256e+01
    lcdm_chi2 = lcdm_chi2_cmb + lcdm_chi2_bao + lcdm_chi2_sn

    print(f"CPL chi2: {cpl_chi2:.6f}, LCDM chi2: {lcdm_chi2:.6f}, difference: {cpl_chi2 - lcdm_chi2:.6f}")
    q_chi2 = -2 * best_f
    print(f"Quintessence chi2: {q_chi2:.6f}, Delta LCDM: {q_chi2 - lcdm_chi2:.6f}, Delta CPL: {q_chi2 - cpl_chi2:.6f}")

    camb_keys = ['phi_train', 'V_train', 'lengthscale', 'As', 'ombh2', 'omch2', 'H0','ns']
    param_dict = {k: pdict[k] for k in camb_keys if k in pdict}

    plot_spline_quintessence(param_dict=param_dict,nspline=nspline,output_prefix=f"SplineQ_Camspec_all")


if __name__=='__main__':
    main()
