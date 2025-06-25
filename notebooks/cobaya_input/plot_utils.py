import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import camb
from camb import model, initialpower
from scipy.interpolate import UnivariateSpline

# configure matplotlib
font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


def plot_spline_quintessence(param_dict,
                             nspline=4,
                             zs=None,
                             wz_median_file='../wz_median.txt',
                             wz_ul_file='../wz_ul.txt',
                             wz_ll_file='../wz_ll.txt',
                             hz_median_file='../hz_hlcdm_median.txt',
                             hz_ul_file='../hz_hlcdm_ul.txt',
                             hz_ll_file='../hz_hlcdm_ll.txt',
                             output_prefix='SplineQ_nocmb'):
    """
    Generates summary plots for spline quintessence using CAMB.

    Parameters
    ----------
    param_dict : dict
        Must contain keys 'lengthscale', 'phi2', 'phi3', ..., 'phi{nspline}',
        'V2', ..., 'V{nspline}', 'ombh2', 'omch2', 'H0'.
    nspline : int
        Number of spline nodes (including phi0/V0 at z=0).
    zs : array-like, optional
        Redshift grid. If None, uses np.linspace(0,2.4,500).
    *_file : str
        Paths to data files for observational constraints.
    output_prefix : str
        Prefix to use when saving plot PDFs.
    """
    # redshift grid
    if zs is None:
        zs = np.linspace(0, 2.4, 500)
    scales = 1.0 / (1.0 + zs)

    # setup CAMB
    camb.set_feedback_level(level=0)

    # baseline LCDM
    ombh2          = 0.0223828
    omch2          = 0.1201075
    omk            = 0.
    hubble         = 67.32117
    pars_lcdm = camb.set_params(ombh2=ombh2, omch2=omch2, H0=hubble, dark_energy_model='fluid')
    results_lcdm = camb.get_results(pars_lcdm)
    cl_lcdm = results_lcdm.get_lensed_scalar_cls(CMB_unit='muK')
    ls = np.arange(cl_lcdm.shape[0])

    # spline quintessence
    # prepare phi and V training arrays
    phi_train = [param_dict[f'phi{i}'] for i in range(2, nspline+1)]
    V_train = [param_dict[f'V{i}'] for i in range(2, nspline+1)]
    phi_train = np.concatenate([[0.0], phi_train])
    V_train = np.concatenate([[0.0], V_train])

    # extract cosmological parameters
    ombh2 = param_dict['ombh2']
    omch2 = param_dict['omch2']
    H0 = param_dict['H0']
    lengthscale = param_dict['lengthscale']
    camb.set_feedback_level(level=2)
    pars_spline = camb.set_params(V0=1e-8,
                                  dark_energy_model='QuintessenceSpline',
                                  lengthscale=lengthscale,
                                  ombh2=ombh2,
                                  omch2=omch2,
                                  H0=H0,
                                  nspline=nspline,
                                  phi_train=phi_train,
                                  V_train=V_train)
    results_spline = camb.get_results(pars_spline)
    cl_spline = results_spline.get_lensed_scalar_cls(CMB_unit='muK')

    # start plotting
    fig, ax = plt.subplots(3, 2, figsize=(15, 12), layout='constrained')
    # axis labels
    ax[0,0].set_xlabel(r'$z$')
    ax[0,0].set_ylabel(r'$\Omega_{\rm DE}$')
    ax[1,0].set_xlabel(r'$\phi$')
    ax[1,0].set_ylabel(r'$V_{\phi}/V_0$')
    ax[2,0].set_xlabel(r'$z$')
    ax[2,0].set_ylabel(r'$\phi$')
    ax[0,1].set_xlabel(r'$\ell$')
    ax[0,1].set_ylabel(r'$ D_\ell^{TT} [\mu K^2]$')
    ax[1,1].set_xlabel(r'$\ell$')
    ax[1,1].set_ylabel(r'$\Delta D_\ell^{TT} [\mu K^2]$')
    ax[2,1].set_xlabel(r'$z$')
    ax[2,1].set_ylabel(r'$\dot{\phi}$')

    # energy densities vs z
    ax[0,0].plot(zs, results_spline.get_Omega('de', z=zs), label='Spline')
    ax[0,0].plot(zs, results_lcdm.get_Omega('de', z=zs), label='LCDM', color='k', ls='-.')
    ax[0,0].legend()

    # CMB power spectra
    ax[0,1].plot(ls[2:], cl_spline[2:,0], label='Spline')
    ax[0,1].plot(ls[2:], cl_lcdm[2:,0], label='LCDM', color='k', ls='-.')
    ax[1,1].plot(ls[2:], cl_spline[2:,0] - cl_lcdm[2:,0], label='Spline')
    ax[1,1].legend()

    # phi and V plots
    ev_phi = np.array(results_spline.get_dark_energy_phi_phidot(1/(1+zs))).T
    phis = np.linspace(0.0, np.max(ev_phi[:,0]), 100)
    Vphis = np.array(results_spline.get_dark_energy_Vphi(phis))
    Vphis /= Vphis[0]
    ax[1,0].plot(phis, Vphis)
    ax[1,0].scatter(phi_train, np.exp(V_train), color='red')

    # phi(z) and phidot
    ax[2,0].plot(zs, ev_phi[:,0], label='Spline')
    ax[2,1].plot(zs, ev_phi[:,1] / scales, label='Spline')
    ax[2,1].legend()

    fig.suptitle(f'Spline Quintessence, n = {nspline}')
    fig.savefig(f'{output_prefix}_{nspline}_summary.pdf', bbox_inches='tight')
    plt.close(fig)

    # w(z) and H(z)
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 4), layout='constrained')
    for a in ax2:
        a.set_xlabel(r'$z$')
    ax2[0].set_ylabel(r'$w_{\rm DE}(z)$')
    ax2[1].set_ylabel(r'$h(z)/h^{\rm LCDM}$')

    w_spline = np.array(results_spline.get_dark_energy_rho_w(1/(1+zs))).T
    print(f'Quintessence wde at z=0 = {w_spline[0,1]:.6f}')
    w_lcdm   = np.array(results_lcdm.get_dark_energy_rho_w(1/(1+zs))).T
    ax2[0].plot(zs, w_spline[:,1], label='Spline')
    ax2[0].plot(zs, w_lcdm[:,1], color='k', ls='-.', label='LCDM')

    # observational w(z)
    wz_m = np.loadtxt(wz_median_file, delimiter=',')
    wz_ul = np.loadtxt(wz_ul_file, delimiter=',')
    wz_ll = np.loadtxt(wz_ll_file, delimiter=',')
    f_m = UnivariateSpline(wz_m[:,0], wz_m[:,1], s=0)
    f_ul = UnivariateSpline(wz_ul[:,0], wz_ul[:,1], s=0)
    f_ll = UnivariateSpline(wz_ll[:,0], wz_ll[:,1], s=0)
    ax2[0].plot(zs, f_m(zs), color='C1', label='DESI+Union3')
    ax2[0].fill_between(zs, f_ll(zs), f_ul(zs), alpha=0.2)
    ax2[0].set_ylim(-2.0, 0.0)
    ax2[0].legend()

    # H(z)
    hz_lcdm = np.array(results_lcdm.h_of_z(zs)) / H0
    hz_spline = np.array(results_spline.h_of_z(zs)) / H0
    ax2[1].plot(zs, hz_spline / hz_lcdm, lw=1.5, label='Spline')
    ax2[1].axhline(1.0, ls='-.', color='k', label='LCDM')
    hz_m = np.loadtxt(hz_median_file, delimiter=',')
    hz_ul = np.loadtxt(hz_ul_file, delimiter=',')
    hz_ll = np.loadtxt(hz_ll_file, delimiter=',')
    f_hm = UnivariateSpline(hz_m[:,0], hz_m[:,1], s=0)
    f_hul = UnivariateSpline(hz_ul[:,0], hz_ul[:,1], s=0)
    f_hll = UnivariateSpline(hz_ll[:,0], hz_ll[:,1], s=0)
    ax2[1].plot(zs, f_hm(zs), color='C1', label='DESI+Union3')
    ax2[1].fill_between(zs, f_hul(zs), f_hll(zs), alpha=0.2)
    ax2[1].legend()

    fig2.savefig(f'{output_prefix}_wz_Hz_{nspline}.pdf', bbox_inches='tight')
    plt.close(fig2)

    print(f'Plots saved with prefix {output_prefix}, nspline={nspline}')
