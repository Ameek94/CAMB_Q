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

fig, ax = plt.subplots(3,2,figsize=(15,9))
ax[0,1].set_ylabel(r'$ D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_ylabel(r'$\Delta D_\ell^{TT} [\mu {\rm K}^2]$')
ax[1,1].set_xlabel(r'$\ell$')
ax[0,1].set_xlabel(r'$\ell$')
ax[0,0].set_ylabel(r'$\Omega_{\rm DE}$')
ax[1,0].set_ylabel(r'$w_{\rm DE}$')
ax[2,0].set_ylabel(r'$\phi$')
ax[0,0].set_xlabel(r'$z$')
ax[1,0].set_xlabel(r'$z$')
ax[2,0].set_xlabel(r'$a$')
ax[2,1].set_xlabel(r'$a$')
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
phi1 = 0.
phi2 = 0.34 #0.2
phi3 = 0.15 #0.4
phi4 = 0.2 #0.6
V1 = 1.
V2 = 0.9 #0.95
V3 = 0.85 #0.90
V4 = 0.8 #0.85
# phi1 = 0.
# phi2 = 0.2
# phi3 = 0.4
# phi4 = 0.6
# # phi5 = 0.8
# # phi6 = 1.
# V1 = 1.
# V2 = 0.95
# V3 = 0.90
# V4 = 0.85
# V5 = 0.80
# V6 = 0.75
lengthscale = 0.36

camb.set_feedback_level(level=2)
pars = camb.set_params(ombh2=ombh2, omch2=omch2, H0=hubble,
                       V0= 1e-10, phi1=phi1, phi2=phi2, phi3=phi3, phi4=phi4,# phi5=phi5, phi6=phi6,
                       V1=V1, V2=V2, V3=V3, V4=V4, #V5=V5, V6=V6,
                       nspline=nspline,lengthscale=lengthscale,
                       dark_energy_model='QuintessenceSpline',) 

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
ax[2,0].semilogx(scales,ev_phi[:,0],label='Spline')
ax[2,1].semilogx(scales,ev_phi[:,1]/scales,label='Spline')

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
fig.suptitle('Spline Quintessence')
fig.tight_layout()
plt.savefig('testspline_quintessence.pdf',bbox_inches='tight')
plt.show()

# plot the spline potential

# plt.figure(figsize=(8,4))
# phis = np.linspace(0.,0.5,50)
# Vphi = np.array(results.get_dark_energy_Vphi(phis)) #.T

# plt.plot(phis,Vphi,label='Spline')
# plt.show()
