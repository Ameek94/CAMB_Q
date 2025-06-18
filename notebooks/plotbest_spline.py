import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
from scipy.interpolate import interp1d
font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text',usetex=True)

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
z1 = np.logspace(5,-2,500)
z2 = np.linspace(0.01,0.)
zs = np.concatenate((z1,z2))
scales = 1/(1+zs)

fig, ax = plt.subplots(3,2,figsize=(15,9),layout='constrained')
ax[0,1].set_ylabel(r'$ D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_ylabel(r'$\Delta D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_xlabel(r'$\ell$')
ax[0,1].set_xlabel(r'$\ell$')
ax[0,0].set_ylabel(r'$\Omega_{\rm DE}$')
ax[1,0].set_ylabel(r'$w_{\rm DE}$')
ax[2,0].set_ylabel(r'$\phi$')
ax[0,0].set_xlabel(r'$z$')
ax[1,0].set_xlabel(r'$z$')
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
nspline = 5
keys = ["V2", "V3", "V4", "V5", "omch2", "H0","lengthscale"]
filepath = "cobaya_input/chains/spline_5.minimum.txt"
subset = load_data_with_subset(filepath, keys)
# print(subset)
    # => {'V2': 0.90901738, 'V3': 0.80273055, 'V4': 0.77630574,
    #     'V5': 0.9018205,   'omch2': 0.12074249, 'H0': 66.674175}

subset['V1'] = 1.01
subset['V6'] = 0.
subset['phi1'] = 0.
subset['phi2'] = 0.2
subset['phi3'] = 0.4
subset['phi4'] = 0.6
subset['phi5'] = 0.8
subset['phi6'] = 1.
nspline = 5
subset['nspline'] = nspline
print(subset)


camb.set_feedback_level(level=2)
pars = camb.set_params(V0= 1e-10,
                       dark_energy_model='QuintessenceSpline',**subset) 

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
wde = np.array(results.get_dark_energy_rho_w(1/(1+zs))).T
ax[1,0].semilogx(zs,wde[:,1])
ax[0,0].semilogx(zs,results.get_Omega('de',z=zs),label='Spline')
cl = results.get_lensed_scalar_cls(CMB_unit='muK')
ls = np.arange(0,cl.shape[0])
ax[1,1].plot(ls[2:],cl[2:,0]-cl_LCDM[2:,0],label='Spline')
ax[0,1].plot(ls[2:],cl[2:,0],label='Spline')
ev_phi = np.array(results.get_dark_energy_phi_phidot(1/(1+zs))).T
ax[2,0].semilogx(zs,ev_phi[:,0],label='Spline')
ax[2,1].semilogx(zs,ev_phi[:,1]/scales,label='Spline')

print(f'LCDM: thetamc = {results.cosmomc_theta():.6f}')
wde = np.array(results_LCDM.get_dark_energy_rho_w(1/(1+zs))).T
om = ['K', 'cdm', 'baryon', 'photon', 'neutrino' , 'nu', 'de']
omega0 = []
for param in om:
    omega0.append(results_LCDM.get_Omega(param))
omdict = dict(zip(om,omega0))
print("Energy Densities: "+"".join(f"{key} = {value:.6f}, " for key, value in omdict.items()))
print('Sum of energy densities = {:.6f}\n'.format(sum(omega0)))
ax[0,1].plot(ls[2:],cl_LCDM[2:,0],label='LCDM',color='k',ls='-.')
ax[0,0].semilogx(zs,results_LCDM.get_Omega('de',z=zs),label=r'LCDM',color='k',ls='-.')
ax[1,0].semilogx(zs,wde[:,1],color='k',ls='-.')

ax[0,0].legend()
fig.suptitle(f'Spline Quintessence, n = {nspline}')
# fig.tight_layout()
plt.savefig(f'spline_quintessence_{nspline}.pdf',bbox_inches='tight')
plt.show()