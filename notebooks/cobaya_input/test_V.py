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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


phi, Vphi = np.loadtxt('phi_Vphis_old.txt', unpack=True)
Vphi = Vphi / Vphi.max()  # Normalize to max value

gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2)),
    n_restarts_optimizer=10,
    alpha=1e-6
)
print("Fitting Gaussian Process...")
phi_train = phi[::20]  # Use every 20th point for training
Vphi_train = Vphi[::20]  # Corresponding V(ϕ)
print(f"Training points: phi_train = {phi_train}, Vphi_train = {Vphi_train}, npoints = {len(phi_train)}")
gp.fit(phi_train[:, np.newaxis], Vphi_train)
print("GP length scale:", gp.kernel_.length_scale)
V_gp = gp.predict(phi[:, np.newaxis])


plt.plot(phi, Vphi, label=r'$V(\phi)$', color='blue')
plt.plot(phi, V_gp, label='GP Prediction', color='orange')
plt.scatter(phi_train, Vphi_train, color='red', label='Training Points', s=10)
plt.legend()
plt.show()


zs = np.linspace(0,2.4,500)
scales = 1/(1+zs)

fig,ax = plt.subplots(1,2,figsize=(15,4),layout='constrained')


ombh2          = 0.0223828
omch2          = 0.1201075
omk            = 0.
hubble         = 67.32117
camb.set_feedback_level(level=0)
pars = camb.set_params( ombh2=ombh2, omch2=omch2, H0=hubble,dark_energy_model='fluid')
results_LCDM = camb.get_results(pars)


camb.set_feedback_level(level=2)
nspline = len(phi_train)  # Number of spline points
vals_dict = {
    'H0': 66.5,
    'ombh2': 0.0224,
    'omch2': 0.1184}
pars = camb.set_params(V0= 1e-8,
                       dark_energy_model='QuintessenceSpline',
                       **vals_dict,
                       phi_train=phi_train, V_train=np.log(Vphi_train), nspline=nspline)

# pars.set_accuracy(AccuracyBoost=4.)
# camb.model.AccuracyParams(AccuracyBoost=2.,BackgroundTimeStepBoost=2.)
results = camb.get_results(pars)


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
plt.show()