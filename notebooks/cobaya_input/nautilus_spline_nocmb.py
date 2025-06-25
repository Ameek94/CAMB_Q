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
from plot_utils import plot_spline_quintessence

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
    phis = order_transform(params[1:nspline], reverse=False)
    Vs = order_transform(params[nspline:2*(nspline-1) + 1], reverse=True)
    x_phys = np.concatenate([[params[0]], phis, Vs, params[2*(nspline-1)+1:]])
    return input_unstandardize(x_phys, param_bounds)

def inverse_prior(x_phys, param_bounds, nspline=4):
    """ Transform x_phys from physical parameter space back to the unit cube. """
    x = input_standardize(x_phys, param_bounds)
    phis = inverse_order_transform(x[1:nspline], reverse=False)
    Vs = inverse_order_transform(x[nspline:2*(nspline-1) + 1], reverse=True)
    x_inv = np.concatenate([[x[0]], phis, Vs, x[2*(nspline-1)+1:]])
    return x_inv

def loglikelihood(x, cobaya_model=None, param_list=None, param_bounds=None, nspline=4):
    pdict = dict(zip(param_list,x)) # type: ignore
    res = cobaya_model.logpost(pdict, make_finite=True)
    return max(res, -1e5)


def main():
    nspline = 4
    input_file = f'./spline_{nspline}_vector_nocmb_nautilus.yaml'
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

    prior_kwargs = {'param_bounds': param_bounds, 'nspline': nspline}
    likelihood_kwargs = {'cobaya_model': cobaya_model, 'param_bounds': param_bounds,
                         'param_list': param_list, 'nspline': nspline}

    n_live = 5000
    sampler = Sampler(prior, loglikelihood, ndim, pass_dict=False,filepath=f'spline_quintessence_{nspline}.h5',
                      prior_kwargs=prior_kwargs, likelihood_kwargs=likelihood_kwargs,n_live=n_live,
                      resume=True,pool=None)

    sampler.run(verbose=True, f_live=0.002,n_like_max=int(1e5))
    print('log Z: {:.2f}'.format(sampler.log_z))

    samples, logl, logwt = sampler.posterior() # type: ignore
    best_idx = np.argmax(logl)
    best_point = samples[best_idx] # type: ignore
    best_param_dict = dict(zip(param_list, best_point))
    print(f"Best fit parameters: {best_param_dict}")

    plot_spline_quintessence(best_param_dict,nspline=nspline, output_prefix='SplineQ_nautilus')

if __name__ == "__main__":
    main()