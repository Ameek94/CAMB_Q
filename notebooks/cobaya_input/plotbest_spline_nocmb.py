import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
from scipy.interpolate import interp1d, UnivariateSpline
font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text',usetex=True)
import faulthandler; faulthandler.enable()

print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

import pandas as pd

def load_data_with_subset(filepath, keep_keys):
    """
    Loads a whitespace-delimited file whose first line (starting with '#')
    gives the real headers. Returns only the columns in keep_keys
    as a single dict (if one row) or a list of dicts (if many rows).
    """
    # 1) Read the header line yourself
    with open(filepath, 'r') as f:
        # Skip blank lines till we hit the header
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Expect header to start with '#'
            if line.startswith('#'):
                # drop the '#' and any leading spaces, then split on whitespace
                columns = line.lstrip('#').strip().split()
                break
        else:
            raise ValueError("No header line starting with '#' found.")

    # 2) Sanity‑check that the keys you want exist
    missing = set(keep_keys) - set(columns)
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    # 3) Use pandas to read the rest of the file, telling it:
    #    - names=columns: use our manually read names
    #    - skiprows=1: skip that first header line
    #    - delim_whitespace=True: split on any amount of space
    df = pd.read_csv(
        filepath,
        names=columns,
        skiprows=1,
        delim_whitespace=True,
        engine='python'
    )

    # 4) Subset to only the keys you care about
    df_sub = df[keep_keys]

    # 5) Convert to record(s)
    records = df_sub.to_dict(orient='records')
    return records[0] if len(records) == 1 else records

# Set initial parameters
pars = camb.set_params( ombh2=0.022, omch2=0.122, H0=67.2)
pars.Accuracy.AccuracyBoost=2.
pars.Accuracy.BackgroundTimeStepBoost=2.
# print(pars)

# Spline Quintessence
# z1 = np.logspace(5,-2,500)
# z2 = np.linspace(0.01,0.)
# zs = np.concatenate((z1,z2))
zs = np.linspace(0,2.4,500)
scales = 1/(1+zs)
Ns = np.log(scales)

fig, ax = plt.subplots(3,2,figsize=(15,12),layout='constrained')
ax[0,1].set_ylabel(r'$ D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_ylabel(r'$\Delta D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_xlabel(r'$\ell$')
ax[0,1].set_xlabel(r'$\ell$')
ax[0,0].set_ylabel(r'$\Omega_{\rm DE}$')
ax[1,0].set_ylabel(r'$V_{\phi}/V_0$')
ax[2,0].set_ylabel(r'$\phi$')
ax[0,0].set_xlabel(r'$z$')
ax[1,0].set_xlabel(r'$\phi$')
ax[2,0].set_xlabel(r'$z$')
ax[2,1].set_xlabel(r'$z$')
ax[2,1].set_ylabel(r'$\dot{\phi}$')

ombh2          = 0.0223828
omch2          = 0.1201075
omk            = 0.
hubble         = 67.32117

camb.set_feedback_level(level=0)
pars = camb.set_params( ombh2=ombh2, omch2=omch2, H0=hubble,dark_energy_model='fluid')
results_LCDM = camb.get_results(pars)
cl_LCDM = results_LCDM.get_lensed_scalar_cls(CMB_unit='muK')
ls_LCDM = np.arange(cl_LCDM.shape[0])

dark_energy_model  = 'QuintessenceSpline'
nspline = 4

# lengthscale = 0.276477
# phi2 = 0.124353
# phi3 = 0.345609
# phi4 = 0.393382
# V2 = -0.165591
# V3 = -0.218877
# V4 = -0.455565
# omch2 = 0.122605
# ombh2 = 0.021789
# H0 = 65.654470
# phi_train = [phi2, phi3,phi4]
# V_train = [V2, V3, V4]

param_dict = {'lengthscale': 0.3505253387749452, 'phi2': 0.15554633610320867, 'phi3': 0.24469561044753646, 'phi4': 0.3963848159429807, 'V2': -0.21592341134816806, 'V3': -0.4162189753600877, 'V4': -0.5303014860292089, 'omch2': 0.1205240492254077, 'ombh2': 0.02248387903932483, 'H0': 65.85170246559673}
phi_train = [param_dict['phi'+str(i)] for i in range(2,nspline+1)]
V_train = [param_dict['V'+str(i)] for i in range(2,nspline+1)]
phi_train = np.concatenate([[0.],phi_train])
V_train = np.concatenate([[0.],V_train])

print(f"phi train = {phi_train}")
print(f"V train = {V_train}")


# for key in param_dict:
# #     param_dict[key] = float(param_dict[key])

# print(f'Using best fit parameters: {param_dict}')

keys = ['lengthscale', 'ombh2', 'omch2', 'H0']
vals_dict = dict(zip(keys, [param_dict[key] for key in keys]))

camb.set_feedback_level(level=2)
pars = camb.set_params(V0= 1e-8,
                       dark_energy_model='QuintessenceSpline',
                       **vals_dict,
                       phi_train=phi_train, V_train=V_train, nspline=nspline)

# pars.set_accuracy(AccuracyBoost=4.)
# camb.model.AccuracyParams(AccuracyBoost=2.,BackgroundTimeStepBoost=2.)
results = camb.get_results(pars)
print(f'Spline Quintessence, thetamc = {results.cosmomc_theta():.6f}, Age of Universe = {results.physical_time(0.):.4f} Gyrs')
om = ['K', 'cdm', 'baryon', 'photon', 'neutrino' , 'nu', 'de']
omega0 = []
for param in om:
    omega0.append(results.get_Omega(param))
omdict = dict(zip(om,omega0))
print("Energy Densities: "+"".join(f"{key} = {value:.6f}, " for key, value in omdict.items()))
print('Sum of energy densities = {:.6f}\n'.format(sum(omega0)))
ax[0,0].plot(zs,results.get_Omega('de',z=zs),label='Spline')
cl = results.get_lensed_scalar_cls(CMB_unit='muK')
ls = np.arange(0,cl.shape[0])
ax[1,1].plot(ls[2:],cl[2:,0]-cl_LCDM[2:,0],label='Spline')
ax[0,1].plot(ls[2:],cl[2:,0],label='Spline')
ev_phi = np.array(results.get_dark_energy_phi_phidot(1/(1+zs))).T
phis = np.linspace(0.,max(ev_phi[:,0]),100)
Vphis = np.array(results.get_dark_energy_Vphi(phis))
# np.savetxt('spline_quintessence_Vphis.txt',np.array([phis,Vphis]).T,header='phi Vphi')
# phis = np.linspace(0,0.4,100)
# V_phi = np.array(results.get_dark_energy_Vphi(phis))
# V0 =  V_phi[0]
# V_phi_realized = np.array(results.get_dark_energy_Vphi(ev_phi[:,0]))
# print(f"Vphi0 = {V_phi[0]:.4e}, Vphi_realized0 = {V_phi_realized[0]:.4e}")
# V_phi = V_phi / V0  # Normalize to V(0)
# V_phi_realized = V_phi_realized / V0  # Normalize to V(0)
Vphis = Vphis / Vphis[0]  # Normalize to V(0)
ax[2,0].plot(zs,ev_phi[:,0],label='Spline')
ax[2,1].plot(zs,ev_phi[:,1]/scales,label='Spline')
ax[1,0].plot(phis,Vphis)
ax[1,0].scatter(phi_train,np.exp(V_train),color='red')
# ax[1,0].plot(ev_phi[:,0],V_phi_realized,ls='--',color='C3')
print(f'LCDM: thetamc = {results.cosmomc_theta():.6f}')
om = ['K', 'cdm', 'baryon', 'photon', 'neutrino' , 'nu', 'de']
omega0 = []
for param in om:
    omega0.append(results_LCDM.get_Omega(param))
omdict = dict(zip(om,omega0))
print("Energy Densities: "+"".join(f"{key} = {value:.6f}, " for key, value in omdict.items()))
print('Sum of energy densities = {:.6f}\n'.format(sum(omega0)))
ax[0,1].plot(ls[2:],cl_LCDM[2:,0],label='LCDM',color='k',ls='-.')
ax[0,0].plot(zs,results_LCDM.get_Omega('de',z=zs),label=r'LCDM',color='k',ls='-.')

ax[0,0].legend()
# ax[1,0].legend()
fig.suptitle(f'Spline Quintessence, n = {nspline}')
# fig.tight_layout()
plt.savefig(f'SplineQ_{nspline}_summary_cmb.pdf',bbox_inches='tight')
# plt.show()

fig,ax = plt.subplots(1,2,figsize=(15,4),layout='constrained')

for x in ax:
    x.set_xlabel(r'$z$')
ax[0].set_ylabel(r'$w_{\rm DE}(z)$')
ax[1].set_ylabel(r'$h(z)/h^{\rm LCDM}$')

wde = np.array(results.get_dark_energy_rho_w(1/(1+zs))).T
print(f'Quintessence wde at z=0 = {wde[0,1]:.6f}')
ax[0].plot(zs,wde[:,1],label='Spline',color='C0')
wde = np.array(results_LCDM.get_dark_energy_rho_w(1/(1+zs))).T
ax[0].plot(zs,wde[:,1],color='k',ls='-.')
wz_m = np.loadtxt('../wz_median.txt',skiprows=0,delimiter=',')
wz_ul = np.loadtxt('../wz_ul.txt',skiprows=0,delimiter=',')
wz_ll = np.loadtxt('../wz_ll.txt',skiprows=0,delimiter=',')
f_ll = UnivariateSpline(wz_ll[:,0],wz_ll[:,1],s=0)
f_ul = UnivariateSpline(wz_ul[:,0],wz_ul[:,1],s=0)
f_m = UnivariateSpline(wz_m[:,0],wz_m[:,1],s=0)
ax[0].plot(zs,f_m(zs),color='C1',label=r'DESI+Union3')
ax[0].fill_between(zs,f_ll(zs),f_ul(zs),color='C1',alpha=0.2)
ax[0].set_ylim(-2.,0.)
ax[0].legend()

hz_LCDM = np.array(results_LCDM.h_of_z(zs))/hubble
hubble_spline = vals_dict['H0']
hz_spline = np.array(results.h_of_z(zs))/hubble_spline
ax[1].plot(zs,hz_spline/hz_LCDM,lw=1.5)
ax[1].axhline(1,ls='-.',color='k',label='LCDM')

hz_m = np.loadtxt('../hz_hlcdm_median.txt',skiprows=0,delimiter=',')
hz_ul = np.loadtxt('../hz_hlcdm_ul.txt',skiprows=0,delimiter=',')
hz_ll = np.loadtxt('../hz_hlcdm_ll.txt',skiprows=0,delimiter=',')
f_ll = UnivariateSpline(hz_ll[:,0],hz_ll[:,1],s=0)
f_ul = UnivariateSpline(hz_ul[:,0],hz_ul[:,1],s=0)
f_m = UnivariateSpline(hz_m[:,0],hz_m[:,1],s=0)
f_m2 = UnivariateSpline(hz_m[:,0],hz_m[:,1]) #smooth
ax[1].plot(zs,f_m(zs),color='C1',label=r'DESI+Union3')
ax[1].fill_between(zs,f_ul(zs),f_ll(zs),alpha=0.2,color='C1')


plt.savefig(f'SplineQ_wz_Hz_{nspline}_cmb.pdf',bbox_inches='tight')