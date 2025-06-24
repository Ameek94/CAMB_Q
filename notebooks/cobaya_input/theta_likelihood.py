import numpy as np
from scipy.integrate import quad
from cobaya.likelihood import Likelihood

# Speed of light in km/s\c = 299792.458

class ThetaLikelihood(Likelihood):
    r"""
    Gaussian likelihood for cosmomc_theta = r_s / D_A at recombination
    with mean = 1.04087 and sigma = 0.00031.
    """
    # Mean and error (can be overridden in .yaml)
    mean = 1.04087
    sigma = 0.00031

    def get_requirements(self):
        return {
            "H0": None,
            "ombh2": None,
            "omch2": None,
            "omnuh2": 0.0,
        }

    def loglkl(self, **params_values):
        # Extract cosmological parameters
        H0 = params_values["H0"]
        ombh2 = params_values["ombh2"]
        omch2 = params_values["omch2"]
        omnuh2 = params_values.get("omnuh2", 0.0)

        # Compute theta
        theta = self.compute_theta(ombh2, omch2, omnuh2, H0)

        # Gaussian log-likelihood
        return -0.5 * ((theta - self.mean) / self.sigma) ** 2

    def compute_theta(self, ombh2, omch2, omnuh2, H0):
        # Total matter density today
        omega_m_h2 = ombh2 + omch2 + omnuh2
        h = H0 / 100.0
        # Radiation density from photons + neutrinos
        omega_r_h2 = 4.15e-5  # includes N_eff = 3.046
        # Dimensionless densities
        Omega_m = omega_m_h2 / h**2
        Omega_r = omega_r_h2 / h**2
        Omega_lambda = 1.0 - Omega_m - Omega_r

        # 1) Compute redshift of last scattering (Hu & Sugiyama)
        zstar = 1048.0 * (1 + 0.00124 * ombh2**(-0.738)) * (
            1 + (0.0783 * ombh2**(-0.238) / (1 + 39.5 * ombh2**0.763))
            * (omega_m_h2 + ombh2) ** (0.560 / (1 + 21.1 * ombh2**1.81))
        )
        astar = 1.0 / (1.0 + zstar)

        # 2) Compute sound horizon at zstar: r_s = \int_0^{a*} da (c_s / (a^2 H(a)))
        def dsound_da(a):
            R = 3e4 * a * ombh2
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return self.dtd_a(a, h, Omega_m, Omega_r, Omega_lambda) * cs

        rs, _ = quad(dsound_da, 1e-8, astar, epsabs=1e-8)

        # 3) Angular diameter distance to zstar
        DA = self.angular_diameter_distance(zstar, h, Omega_m, Omega_r, Omega_lambda)

        # 4) Theta = r_s / D_A
        return rs / DA

    def dtd_a(self, a, h, Om, Or, Ol):
        """
        d conformal time per unit scale factor: dt/da = 1/(a^2 H(a)),
        with H(a) = H0 * E(a).
        """
        H0 = h * 100.0  # km/s/Mpc
        E = np.sqrt(Om / a**3 + Or / a**4 + Ol)
        return (299792.458 / H0) / (a**2 * E)

    def angular_diameter_distance(self, z, h, Om, Or, Ol):
        """
        Compute D_A = (1/(1+z)) * \int_0^z c/H(z') dz'.
        """
        H0 = h * 100.0
        def inv_E(zp):
            return 1.0 / np.sqrt(Om * (1 + zp)**3 + Or * (1 + zp)**4 + Ol)

        D_c, _ = quad(lambda zp: inv_E(zp), 0, z, epsabs=1e-8)
        # Convert to Mpc: c/H0 * integral
        D_c *= (299792.458 / H0)
        return D_c / (1.0 + z)
