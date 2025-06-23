from functools import partial
import sys
import numpy as np
from scipy.optimize import minimize, basinhopping
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import pybobyqa
from nautilus import Sampler
import warnings
warnings.filterwarnings("ignore")
from mpi4py.futures import MPIPoolExecutor

# from loglike import cobaya_loglike


def order_transform(unit_x, low=0., high=1., reverse=False):
    """
    Transform the input vector x to a new order.
    If reverse is True, it will reverse the order of the elements.
    """
    n = np.size(unit_x, axis=-1)
    index = np.arange(n)
    inner_term = np.power(1 - unit_x, 1/(n - index))
    unit_y = 1 - np.cumprod(inner_term, axis=-1)
    if reverse:
        unit_y =  unit_y[::-1]
    return unit_y * (high - low) + low

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

def prior(x,param_bounds,nspline=4):
    params = x.copy()
    phis = params[1:nspline-1]
    phis = order_transform(phis,reverse=False)
    Vs = params[nspline-1: 2*(nspline-1)]
    # print(f"phis: {phis}, Vs: {Vs}")
    Vs = order_transform(Vs, reverse=True)
    x = np.concatenate([[params[0]], phis, Vs, params[2*(nspline - 1):]])
    return input_unstandardize(x, param_bounds)


def loglikelihood(x,cobaya_model=None,logprior_vol=0.,param_list=[]):
    param_dict = dict(zip(param_list, x))
    # try:
    res = cobaya_model.loglike(param_dict, make_finite=True,return_derived=False,)
    if res < -1e5:
        res = -1e5
    # except:
        # res = -1e5
    vals = {k: f'{v:.4f}' for k, v in param_dict.items()}
    # print(f"Parameters {vals} with loglike {res:.4f}")
    return res #+ logprior_vol


def main():
    nspline = 4
    input_file = f'./spline_{nspline}_vector_cmb_nautilus.yaml'
    info = yaml_load(input_file)
    cobaya_model = get_model(info)
    confidence_for_unbounded = 0.9999995
    param_list = list(cobaya_model.parameterization.sampled_params())
    print(f"Parameter list: {param_list}")
    param_bounds = np.array(
            cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
        ).T
    print(f"Parameter bounds: {param_bounds.T} and shape {param_bounds.shape}")
    logprior_vol = np.log(np.prod(param_bounds[1] - param_bounds[0]))
    param_labels = [cobaya_model.parameterization.labels()[k] for k in param_list]
    print(f"Parameter labels: {param_labels}")
    ndim = len(param_list)

    # x0, val = cobaya_model.get_valid_point(100, ignore_fixed_ref=False,
    #                                                logposterior_as_dict=True)
    # init_pt_dict = dict(zip(param_list, x0))
    # print(f"Initial point: {init_pt_dict} with loglike = {val['loglikes']}")


    prior_kwargs = {'param_bounds': param_bounds, 'nspline': nspline}
    likelihood_kwargs = {'cobaya_model': cobaya_model, 'logprior_vol': logprior_vol,
                         'param_list': param_list}


    sampler = Sampler(prior, loglikelihood, ndim, pass_dict=False,filepath=f'spline_quintessence.h5',
                      prior_kwargs=prior_kwargs, likelihood_kwargs=likelihood_kwargs,
                      resume=True,pool=None)

    sampler.run(verbose=True, f_live=0.05,n_like_max=int(1e5))
    print('log Z: {:.2f}'.format(sampler.log_z))

    samples, logl, logwt = sampler.posterior() # type: ignore
    best_idx = np.argmax(logl)
    best_point = samples[best_idx] # type: ignore
    best_param_dict = dict(zip(param_list, best_point))
    print(f"Best fit parameters: {best_param_dict}")

if __name__ == "__main__":
    main()