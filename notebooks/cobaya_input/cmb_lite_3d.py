import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from scipy.linalg import inv, det
from cobaya.likelihood import Likelihood

# Speed of light in km/s
c = 299792.458

# mean = np.array([0.01041, 0.02223, 0.14208])
# cov = 1e-9 * np.array([
#         [0.006621, 0.12444, -1.1929],
#         [0.12444, 21.344, -94.001],
#         [-1.1929, -94.001, 1488.4]
#     ])
# # inv_cov = inv(cov)
# # log_norm = -0.5 * (3 * np.log(2 * np.pi) + np.log(det(cov)))
# rv = multivariate_normal(mean=mean, cov=cov)

# def cmb_lite_3d(**params_values):
#     ombh2 = params_values["ombh2"]
#     omch2 = params_values["omch2"]
#     thetastar = params_values["thetastar"]
#     return rv.logpdf([thetastar, ombh2, omch2 + ombh2])

class cmb_lite_3d(Likelihood):
    r"""
    Taken from https://arxiv.org/abs/2503.14738, https://arxiv.org/abs/2302.12911

    Multivariate Gaussian likelihood for:
      x = [theta_* , omega_b h^2 , omega_c h^2 + omega_b h^2]
    with mean and covariance:
      mu = [0.01041, 0.02223, 0.14208]
      C = 1e-9 * [[   0.006621,   0.12444,   -1.1929 ],
                  [   0.12444,  21.344  ,  -94.001 ],
                  [  -1.1929 , -94.001 , 1488.4   ]]
    """
    # Default mean and covariance (can override in YAML)
    mean = np.array([0.01041, 0.02223, 0.14208])
    cov = 1e-9 * np.array([
        [0.006621, 0.12444, -1.1929],
        [0.12444, 21.344, -94.001],
        [-1.1929, -94.001, 1488.4]
    ])
    inv_cov = inv(cov)
    log_norm = -0.5 * (3 * np.log(2 * np.pi) + np.log(det(cov)))

    def initialize(self):
        self.rv = multivariate_normal(mean=self.mean, cov=self.cov)

    def get_requirements(self):
        return {
            "thetastar": None,
            "ombh2": None,
            "omch2": None,
        }

    def logp(self, **params_values):
        ombh2 = params_values["ombh2"]
        omch2 = params_values["omch2"]
        thetastar = self.provider.get_param("thetastar")
        return self.log_likelihood(ombh2, omch2, thetastar)

    def log_likelihood(self, ombh2, omch2,thetastar):
        x = np.array([thetastar, ombh2, omch2+ombh2])
        # Multivariate Gaussian log-likelihood
        return self.rv.logpdf(x)  # or self.log_norm - 0.5 * d @ self.inv_cov @ d


    # def loglkl(self, **params_values):
    #     # Extract parameters
    #     H0 = params_values["H0"]
    #     ombh2 = params_values["ombh2"]
    #     omch2 = params_values["omch2"]
    #     omnuh2 = params_values.get("omnuh2", 0.0)

    #     # Compute theta
    #     theta = self.compute_theta(ombh2, omch2, omnuh2, H0)

    #     # Construct data vector
    #     x = np.array([theta, ombh2, omch2])
    #     d = x - self.mean

    #     # Multivariate Gaussian log-likelihood
    #     return self.log_norm - 0.5 * d @ self.inv_cov @ d

    # def compute_theta(self, ombh2, omch2, omnuh2, H0):
    #     # Total matter density today
    #     omega_m_h2 = ombh2 + omch2 + omnuh2
    #     h = H0 / 100.0
    #     omega_r_h2 = 4.15e-5  # photons+neutrinos
    #     Omega_m = omega_m_h2 / h**2
    #     Omega_r = omega_r_h2 / h**2
    #     Omega_lambda = 1.0 - Omega_m - Omega_r

    #     # 1) Redshift of last scattering (Hu & Sugiyama)
    #     zstar = 1048.0 * (1 + 0.00124 * ombh2**(-0.738)) * (
    #         1 + (0.0783 * ombh2**(-0.238) / (1 + 39.5 * ombh2**0.763))
    #         * (omega_m_h2 + ombh2) ** (0.560 / (1 + 21.1 * ombh2**1.81))
    #     )
    #     astar = 1.0 / (1.0 + zstar)

    #     # 2) Sound horizon at a* = integral 0->a* da (c_s / (a^2 H(a)))
    #     def dsound_da(a):
    #         R = 3e4 * a * ombh2
    #         cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
    #         return self.dtd_a(a, h, Omega_m, Omega_r, Omega_lambda) * cs
    #     rs, _ = quad(dsound_da, 1e-8, astar, epsabs=1e-8)

    #     # 3) Angular diameter distance to zstar
    #     DA = self.angular_diameter_distance(zstar, h, Omega_m, Omega_r, Omega_lambda)

    #     # theta* = rs / DA
    #     return rs / DA

    # def dtd_a(self, a, h, Om, Or, Ol):
    #     H0 = h * 100.0
    #     E = np.sqrt(Om / a**3 + Or / a**4 + Ol)
    #     return (c / H0) / (a**2 * E)

    # def angular_diameter_distance(self, z, h, Om, Or, Ol):
    #     H0 = h * 100.0
    #     def inv_E(zp):
    #         return 1.0 / np.sqrt(Om * (1 + zp)**3 + Or * (1 + zp)**4 + Ol)
    #     D_c, _ = quad(lambda zp: inv_E(zp), 0, z, epsabs=1e-8)
    #     D_c *= (c / H0)
    #     return D_c / (1.0 + z)
