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

print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

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

fig, ax = plt.subplots(3,2,figsize=(15,12))
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
nspline = 4
phi_train = [0., 0.03600689, 0.10145765, 0.26488344] #, 0.4]
V_train = [ 0., -0.03056663, -0.19151474, -0.19157352] #, -0.3 ] #
lengthscale = 0.197173

camb.set_feedback_level(level=2)
pars = camb.set_params(ombh2=ombh2, omch2=omch2, H0=hubble,
                       V0= 1e-10, phi_train=phi_train, V_train=V_train,
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
ax[1,0].plot(zs,wde[:,1])
ax[0,0].plot(zs,results.get_Omega('de',z=zs),label='Spline')
cl = results.get_lensed_scalar_cls(CMB_unit='muK')
ls = np.arange(0,cl.shape[0])
ax[1,1].plot(ls[2:],cl[2:,0]-cl_LCDM[2:,0],label='Spline')
ax[0,1].plot(ls[2:],cl[2:,0],label='Spline')
ev_phi = np.array(results.get_dark_energy_phi_phidot(1/(1+zs))).T
ax[2,0].plot(zs,ev_phi[:,0],label='Spline')
ax[2,1].plot(zs,ev_phi[:,1]/scales,label='Spline')

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
ax[0,0].plot(zs,results_LCDM.get_Omega('de',z=zs),label=r'LCDM',color='k',ls='-.')
ax[1,0].plot(zs,wde[:,1],color='k',ls='-.')


wz_m = np.loadtxt('wz_median.txt',skiprows=0,delimiter=',')
wz_ul = np.loadtxt('wz_ul.txt',skiprows=0,delimiter=',')
wz_ll = np.loadtxt('wz_ll.txt',skiprows=0,delimiter=',')

f_ll = UnivariateSpline(wz_ll[:,0],wz_ll[:,1],s=0)
f_ul = UnivariateSpline(wz_ul[:,0],wz_ul[:,1],s=0)
f_m = UnivariateSpline(wz_m[:,0],wz_m[:,1],s=0)
ax[1,0].plot(zs,f_m(zs),color='k',label='DESI+Union3')
ax[1,0].fill_between(zs,f_ll(zs),f_ul(zs),color='k',alpha=0.2)
ax[1,0].set_ylim(-2.,0.)

ax[0,0].legend()
fig.suptitle('Spline Quintessence')
fig.tight_layout()
plt.savefig('testspline_quintessence.pdf',bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,4))
phis = np.linspace(-0.1,phi_train[-1],50)
Vphis = np.array(results.get_dark_energy_Vphi(phis))
print('VPhis',  Vphis)
# np.savetxt('spline_quintessence_Vphis.txt',np.array([phis,Vphis]).T,header='phi Vphi')
# phis = np.linspace(0,0.4,100)
# V_phi = np.array(results.get_dark_energy_Vphi(phis))
# V0 =  V_phi[0]
# V_phi_realized = np.array(results.get_dark_energy_Vphi(ev_phi[:,0]))
# print(f"Vphi0 = {V_phi[0]:.4e}, Vphi_realized0 = {V_phi_realized[0]:.4e}")
# V_phi = V_phi / V0  # Normalize to V(0)
# V_phi_realized = V_phi_realized / V0  # Normalize to V(0)
Vphis = Vphis / max(Vphis)  # Normalize to V(0)
plt.plot(phis,Vphis)
plt.scatter(phi_train,np.exp(V_train),color='red',label='Training points')
plt.axvline(max(ev_phi[:,0]),color='k',ls='--')
plt.show()


# plot the spline potential

# plt.figure(figsize=(8,4))
# phis = np.linspace(0.,0.5,50)
# Vphi = np.array(results.get_dark_energy_Vphi(phis)) #.T

# plt.plot(phis,Vphi,label='Spline')
# plt.show()
