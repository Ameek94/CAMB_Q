from functools import partial
import logging
import sys
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from typing import Any, Callable, List,Optional, Tuple, Union, Dict
from nautilus import Sampler
import warnings
warnings.filterwarnings("ignore")
from mpi4py.futures import MPIPoolExecutor
from jaxbo.bo import BOBE
from jaxbo.bo_utils import plot_final_samples
from jaxbo.loglike import external_loglike
import time

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'

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

def prior(x,param_bounds,nspline=5):
    params = x.copy()
    Vs = params[1: nspline]
    # print(f"phis: {phis}, Vs: {Vs}")
    Vs = order_transform(Vs, reverse=True)
    x = np.concatenate([[params[0]], Vs, params[nspline:]])
    return input_unstandardize(x, param_bounds)

# class spline_loglike(external_loglike):

#         def __init__(self
#                  ,loglikelihood: Callable
#                  ,ndim: int
#                  ,param_list: Optional[list[str]] = None # ndim length
#                  ,param_labels: Optional[list[str]] = None# ndim length
#                  ,param_bounds: Optional[Union[list,np.ndarray]] = None # 2 x ndim shaped
#                  ,noise_std: float = 0.
#                  ,name: Optional[str] = None,
#                  vectorized: bool = False,
#                  minus_inf: float = -1e5,
#                  cobaya_model: Optional[Any] = None
#                  ) -> None:

#             super().__init__(loglikelihood=loglikelihood,
#                          ndim=ndim,
#                          param_list=param_list, param_labels=param_labels, param_bounds=param_bounds,
#                          noise_std=noise_std, name=name, vectorized=False, minus_inf=minus_inf)
#             self.cobaya_model = cobaya_model
#             self.param_list = param_list

#         def __call__(self, x: Union[np.ndarray, List[float]]) -> Union[np.ndarray, float]:
#             """
#             Evaluate the log-likelihood at point x.
#             """
#             # if x is 2d flatten it
#             if x.ndim == 2:
#                 x = x.flatten()

#             x = prior(x, self.param_bounds, nspline=4)  # Transform to original domain
#             param_dict = dict(zip(self.param_list, x)) # type: ignore
#             # try:
#             res = self.cobaya_model.loglike(param_dict, make_finite=True,return_derived=False,)
#             if res < self.minus_inf:
#                 res = self.minus_inf
#             return res

def loglikelihood(x,cobaya_model=None,param_list=[],param_bounds = None):
    param_dict = dict(zip(param_list, x))
    # try:
    if x.ndim == 2:
        x = x.flatten()
    param_dict = dict(zip(param_list, x))
    # print(f"Input x: {param_dict}")
    x = prior(x, param_bounds, nspline=4)  # Transform to original domain
    param_dict = dict(zip(param_list, x))
    # print(f"Transformed x: {param_dict}")
    res = cobaya_model.loglike(param_dict, make_finite=True,return_derived=False,)
    if res < -1e5:
        res = -1e5
    val_dict = {k: f'{v:.4f}' for k, v in param_dict.items()}
    print(f"Parameters {val_dict} with loglike {res:.4f}")
    return res

# def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
#     """
#     Computes the Expected Improvement at points X based on existing samples
#     X_sample and Y_sample, using a fitted Gaussian process gp.
#     """
#     mu, sigma = gp.predict(X, return_std=True)
#     sigma = sigma.reshape(-1, 1)
#     Y_best = np.max(Y_sample)

#     with np.errstate(divide='warn'):
#         improvement = mu - Y_best - xi
#         Z = improvement / sigma
#         ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
#         ei[sigma == 0.0] = 0.0
#     return ei

def main():

    nspline = 5
    input_file = f'./spline_{nspline}_fixed_vector_cmb.yaml'
    info = yaml_load(input_file)
    cobaya_model = get_model(info)
    rootlogger = logging.getLogger()
    rootlogger.handlers.pop()
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

    obj_func = partial(loglikelihood, cobaya_model=cobaya_model, param_list=param_list, param_bounds=param_bounds)

    my_loglike = external_loglike(loglikelihood=obj_func,
                                  ndim=ndim,
                                  param_list=param_list,
                                  param_labels=param_labels,
                                  param_bounds = None,
                                  name = f'spline_{nspline}_fixed')

    start = time.time()
    sampler = BOBE(n_cobaya_init=0, n_sobol_init =64,
        miniters=500, maxiters=1500,max_gp_size=1000,
        loglikelihood=my_loglike,
        resume=False,
        resume_file=f'{my_loglike.name}.npz',
        save=True,
        fit_step = 20, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        lengthscale_priors='DSLP',logz_threshold=10.,
        svm_threshold=150,svm_gp_threshold=2500,
        use_svm=True,svm_use_size=400,svm_update_step=5,minus_inf=-1e5,)


    gp, ns_samples, logz_dict = sampler.run()
    end = time.time()
    print(f"Total time taken = {end-start:.4f} seconds")

    best_idx = np.argmax(gp.train_y)
    best_point = gp.train_x[best_idx]  # type: ignore
    best_point = input_unstandardize(best_point.flatten(), param_bounds)
    best_param_dict = dict(zip(param_list, best_point))
    best_param_dict = {k: f'{float(v):.6f}' for k, v in best_param_dict.items()}
    print(f"Best fit parameters: {best_param_dict}")


if __name__ == "__main__":
    main()
