from ctypes import POINTER, byref, c_bool, c_double, c_int

from .baseconfig import AllocatableArrayDouble, CAMBError, F2003Class, f_pointer, fortran_class, np, numpy_1d

class DarkEnergyModel(F2003Class):
    """
    Abstract base class for dark energy model implementations.
    """

    _fields_ = [("__is_cosmological_constant", c_bool), ("__num_perturb_equations", c_int)]

    def validate_params(self) -> None:
        pass


class DarkEnergyEqnOfState(DarkEnergyModel):
    """
    Abstract base class for models using w and wa parameterization with use w(a) = w + (1-a)*wa parameterization,
    or call set_w_a_table to set another tabulated w(a). If tabulated w(a) is used, w and wa are set
    to approximate values at z=0.

    See :meth:`.model.CAMBparams.set_initial_power_function` for a convenience constructor function to
    set a general interpolated P(k) model from a python function.

    """

    _fortran_class_module_ = "DarkEnergyInterface"
    _fortran_class_name_ = "TDarkEnergyEqnOfState"

    _fields_ = [
        ("w", c_double, "w(0)"),
        ("wa", c_double, "-dw/da(0)"),
        ("cs2", c_double, "fluid rest-frame sound speed squared"),
        ("use_tabulated_w", c_bool, "using an interpolated tabulated w(a) rather than w, wa above"),
        ("__no_perturbations", c_bool, "turn off perturbations (unphysical, so hidden in Python)"),
    ]

    _methods_ = [("SetWTable", [numpy_1d, numpy_1d, POINTER(c_int)])]

    def set_params(self, w=-1.0, wa=0, cs2=1.0):
        """
         Set the parameters so that P(a)/rho(a) = w(a) = w + (1-a)*wa

        :param w: w(0)
        :param wa: -dw/da(0)
        :param cs2: fluid rest-frame sound speed squared
        """
        self.w = w
        self.wa = wa
        self.cs2 = cs2
        self.validate_params()

    def validate_params(self):
        if not self.use_tabulated_w and self.wa + self.w > 0:
            raise CAMBError("dark energy model has w + wa > 0, giving w>0 at high redshift")

    def set_w_a_table(self, a, w) -> "DarkEnergyEqnOfState":
        """
        Set w(a) from numerical values (used as cubic spline). Note this is quite slow.

        :param a: array of scale factors
        :param w: array of w(a)
        :return: self
        """
        if len(a) != len(w):
            raise ValueError("Dark energy w(a) table non-equal sized arrays")
        if not np.isclose(a[-1], 1):
            raise ValueError("Dark energy w(a) arrays must end at a=1")
        if np.any(a <= 0):
            raise ValueError("Dark energy w(a) table cannot be set for a<=0")

        a = np.ascontiguousarray(a, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)

        self.f_SetWTable(a, w, byref(c_int(len(a))))
        return self

    def __getstate__(self):
        if self.use_tabulated_w:
            raise TypeError("Cannot save class with splines")
        return super().__getstate__()


@fortran_class
class DarkEnergyFluid(DarkEnergyEqnOfState):
    """
    Class implementing the w, wa or splined w(a) parameterization using the constant sound-speed single fluid model
    (as for single-field quintessence).

    """

    _fortran_class_module_ = "DarkEnergyFluid"
    _fortran_class_name_ = "TDarkEnergyFluid"

    def validate_params(self) -> None:
        super().validate_params()
        if not self.use_tabulated_w:
            if self.wa and (self.w < -1 - 1e-6 or 1 + self.w + self.wa < -1e-6):
                raise CAMBError("fluid dark energy model does not support w crossing -1")

    def set_w_a_table(self, a, w) -> "DarkEnergyEqnOfState":
        # check w array has elements that do not cross -1
        if np.sign(1 + np.max(w)) - np.sign(1 + np.min(w)) == 2:
            raise CAMBError("fluid dark energy model does not support w crossing -1")
        return super().set_w_a_table(a, w)


@fortran_class
class DarkEnergyPPF(DarkEnergyEqnOfState):
    """
    Class implementing the w, wa or splined w(a) parameterization in the PPF perturbation approximation
    (`arXiv:0808.3125 <https://arxiv.org/abs/0808.3125>`_)
    Use inherited methods to set parameters or interpolation table.

    Note PPF is not a physical model and just designed to allow crossing -1 in an ad hoc smooth way. For models
    with w>-1 but far from cosmological constant, it can give quite different answers to the fluid model with c_s^2=1.

    """

    # cannot declare c_Gamma_ppf directly here as have not defined all fields in DarkEnergyEqnOfState (TCubicSpline)
    _fortran_class_module_ = "DarkEnergyPPF"
    _fortran_class_name_ = "TDarkEnergyPPF"


@fortran_class
class AxionEffectiveFluid(DarkEnergyModel):
    """
    Example implementation of a specific (early) dark energy fluid model
    (`arXiv:1806.10608 <https://arxiv.org/abs/1806.10608>`_).
    Not well tested, but should serve to demonstrate how to make your own custom classes.
    """

    _fields_ = [
        ("w_n", c_double, "effective equation of state parameter"),
        ("fde_zc", c_double, "energy density fraction at z=zc"),
        ("zc", c_double, "decay transition redshift (not same as peak of energy density fraction)"),
        ("theta_i", c_double, "initial condition field value"),
    ]

    _fortran_class_name_ = "TAxionEffectiveFluid"
    _fortran_class_module_ = "DarkEnergyFluid"

    def set_params(self, w_n, fde_zc, zc, theta_i=None):
        self.w_n = w_n
        self.fde_zc = fde_zc
        self.zc = zc
        if theta_i is not None:
            self.theta_i = theta_i


# base class for scalar field quintessence models
class Quintessence(DarkEnergyModel):
    r"""
    Abstract base class for single scalar field quintessence models.

    For each model the field value and derivative are stored and splined at sampled scale factor values.

    To implement a new model, need to define a new derived class in Fortran,
    defining Vofphi and setting up initial conditions and interpolation tables (see TEarlyQuintessence as example).

    """

    _fields_ = [
        ("DebugLevel", c_int),
        ("astart", c_double),
        ("integrate_tol", c_double),
        ("sampled_a", AllocatableArrayDouble),
        ("phi_a", AllocatableArrayDouble),
        ("phidot_a", AllocatableArrayDouble),
        ("__npoints_linear", c_int),
        ("__npoints_log", c_int),
        ("__dloga", c_double),
        ("__da", c_double),
        ("__log_astart", c_double),
        ("__max_a_log", c_double),
        ("__ddphi_a", AllocatableArrayDouble),
        ("__ddphidot_a", AllocatableArrayDouble),
        ("__state", f_pointer),
    ]
    _fortran_class_module_ = "Quintessence"

    def __getstate__(self):
        raise TypeError("Cannot save class with splines")


@fortran_class
class EarlyQuintessence(Quintessence):
    r"""
    Example early quintessence (axion-like, as `arXiv:1908.06995 <https://arxiv.org/abs/1908.06995>`_) with potential

     V(\phi) = m^2f^2 (1 - cos(\phi/f))^n + \Lambda_{cosmological constant}

    """

    _fields_ = [
        ("n", c_double, "power index for potential"),
        ("f", c_double, r"f/Mpl (sqrt(8\piG)f); only used for initial search value when use_zc is True"),
        (
            "m",
            c_double,
            "mass parameter in reduced Planck mass units; only used for initial search value when use_zc is True",
        ),
        ("theta_i", c_double, "phi/f initial field value"),
        ("frac_lambda0", c_double, "fraction of dark energy in cosmological constant today (approximated as 1)"),
        ("use_zc", c_bool, "solve for f, m to get specific critical redshift zc and fde_zc"),
        ("zc", c_double, "redshift of peak fractional early dark energy density"),
        ("fde_zc", c_double, "fraction of early dark energy density to total at peak"),
        ("npoints", c_int, "number of points for background integration spacing"),
        ("min_steps_per_osc", c_int, "minimum number of steps per background oscillation scale"),
        (
            "fde",
            AllocatableArrayDouble,
            "after initialized, the calculated background early dark energy fractions at sampled_a",
        ),
        ("__ddfde", AllocatableArrayDouble),
    ]
    _fortran_class_name_ = "TEarlyQuintessence"

    def set_params(self, n, f=0.05, m=5e-54, theta_i=0.0, use_zc=True, zc=None, fde_zc=None):
        self.n = n
        self.f = f
        self.m = m
        self.theta_i = theta_i
        self.use_zc = use_zc
        if use_zc:
            if zc is None or fde_zc is None:
                raise ValueError("must set zc and fde_zc if using 'use_zc'")
            self.zc = zc
            self.fde_zc = fde_zc

import sympy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def is_function_valid(func_string: str) -> bool:
    """
    More robustly analyzes a function string to determine if it's valid.
    """
    # 1. Check for obviously invalid substrings
    invalid_substrings = ['nan', 'oo', '<class'] # 'I' is sympy for imaginary unit
    if any(sub in func_string.lower() for sub in invalid_substrings):
        print(f"Validation failed: Function '{func_string}' contains invalid substring.")
        return False

    try:
        # 2. Parse and check for dependency on 'x'
        x = sympy.symbols('x')
        # We don't need the 'a' symbols for this check
        locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda v: 1/v,
                'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
        expr = sympy.sympify(func_string, locals=locs)

        # 3. If 'x' is not a free symbol, the function is constant and thus invalid
        if x not in expr.free_symbols:
            print(f"Validation failed: Function '{func_string}' is constant with respect to x.")
            return False

    except Exception as e:
        print(f"Validation failed: Could not parse function '{func_string}'. Error: {e}")
        return False

    return True

def identify_fixed_and_variable_parameters(expr_template, param_symbols):
    """
    Analyzes a sympy expression to separate its parameters into 'fixed'
    (purely additive constants) and 'variable' (all others).

    A parameter 'a_i' is classified as 'fixed' if and only if the partial
    derivative of the expression with respect to 'a_i' is exactly 1.

    If any part of the symbolic analysis fails, it safely defaults to
    classifying ALL parameters as variable.

    Args:
        expr_template (sympy.Expr): The symbolic expression for the potential.
        param_symbols (list): A list of the sympy symbols for the parameters (a0, a1...).

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
                                     1. The names of parameters to be fixed.
                                     2. The names of parameters to be treated as variable.
    """
    # Convert all symbols to a set of strings for easy processing
    all_param_names = {str(p) for p in param_symbols}
    x = sympy.symbols('x')

    # try:
    #     params_to_fix = set()
    #     # Loop to identify the fixed parameters
    #     for param in param_symbols:
    #         derivative = sympy.diff(expr_template, param)
    #         print(f"Symbols in derivative: {derivative.free_symbols}")
    #         print(f"Derivative with respect to {param}: {derivative}")

    #         # The parameter is a purely additive constant if its derivative
    #         # has no dependency on 'x'.
    #         if not derivative.has(x):
    #             params_to_fix.add(str(param))

    #         # The parameter is a purely additive constant ONLY if the derivative is 1
    #         # if derivative == 1:
    #         #     params_to_fix.add(str(param))

    #     # The variable parameters are all parameters MINUS the fixed ones
    #     params_to_sample = all_param_names.difference(params_to_fix)

    #     #convert to lists for return
    #     params_to_fix = list(params_to_fix)
    #     params_to_sample = list(params_to_sample)

    #     print(f"Identified fixed parameters: {params_to_fix}, variable parameters: {params_to_sample}")

    #     # Return sorted lists for a deterministic order
    #     return sorted(list(params_to_fix)), sorted(list(params_to_sample))

    # except Exception as e:
    #     # If ANY part of the symbolic analysis fails, default to sampling everything
    #     print(f"Warning: Symbolic analysis failed for expression '{expr_template}'. Error: {e}")
    #     print("Defaulting to sampling all parameters for this function.")

    #     # Return an empty list for fixed params, and all params as variable
    return [], sorted(list(all_param_names))

def load_esr_function_string(file_path, idx=0,verbose=False)-> dict:
    try:
        with open(file_path, "r") as f:
            all_functions = [line.strip() for line in f.readlines() if line.strip()]
            func_string = all_functions[idx]

            is_valid = is_function_valid(func_string)

            function_dict = {}

            if not is_valid:
                function_dict['valid'] = False
                return function_dict


            # 5.1. Convert the string to a sympy expression
            x = sympy.symbols('x', real=True)
            # Create parameter symbols a0, a1, a2, etc.
            a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)

            # Define locals for sympy to understand the function string
            locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x,
                    'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
            # Add parameter symbols to locs
            for j, a_sym in enumerate(a_symbols):
                locs[f'a{j}'] = a_sym
            # Parse the function string
            expr_template = sympy.sympify(func_string, locals=locs)

            # Check if the expression has any parameter symbols
            param_symbols = [sym for sym in expr_template.free_symbols if str(sym).startswith('a')]

            fixed_params, variable_params = identify_fixed_and_variable_parameters(expr_template, param_symbols)

            if verbose:
                print(f"Loaded ESR function: {func_string} with parameters: {[str(p) for p in param_symbols]}")

            function_dict['valid'] = True
            function_dict['func_string'] = func_string
            function_dict['expr_template'] = expr_template
            function_dict['param_symbols'] = param_symbols
            function_dict['fixed_params'] = fixed_params
            function_dict['variable_params'] = variable_params

            return function_dict

    except FileNotFoundError:
        raise CAMBError(f"Could not find file with generated equations: {file_path}")

def create_callable_function(expr_template, param_symbols, x_vals):
    """Create a callable function for parameter evaluation"""
    # print(f"Creating callable function for expression: {expr_template} with parameters: {[str(p) for p in param_symbols]}")
    def objective(params):
        # Substitute parameters into expression
        substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
        expr_with_params = expr_template.subs(substitutions)

        # Convert to callable and evaluate
        callable_func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
        y_pred = callable_func(x_vals)
        # print(f"Evaluated function with params {params}: {y_pred} at x={x_vals}")
        return y_pred
    return objective


# def exp_mapping(vals):

def create_potential_table(expr_template, param_symbols, param_vals, phi_vals, positive_mapping='exp'):
    """Create the potential table from the ESR /sympy expression and parameter values"""

    # print(f"Creating potential table for expression: {expr_template} with parameters: {[str(p) for p in param_symbols]}, param_vals: {param_vals}")

    phi_padding = 1e-2
    padded_phi_vals = np.concatenate((
        np.array([phi_vals[0] - phi_padding]),
        phi_vals,
        np.array([phi_vals[-1] + phi_padding])
    ))
    function = create_callable_function(expr_template, param_symbols, padded_phi_vals)
    if positive_mapping=='exp':
        vals = function(param_vals)
        V_vals = np.exp(vals)  # Ensure V(phi) > 0
        log_V_vals = vals
    elif positive_mapping=='square':
        vals = function(param_vals)
        V_vals = vals**2  # Ensure V(phi) > 0
        log_V_vals = np.log(V_vals + 1e-50)  # Avoid log(0)
    else:
        raise CAMBError(f"Unknown positive_mapping method: {positive_mapping}")

    # log_V_vals = function(param_vals)
    # # print(f"Evaluated log_V_vals: {log_V_vals}")
    # if positive_mapping == 'exp':
    # V_vals = np.exp(log_V_vals)  # Ensure V(phi) > 0
    # Check for invalid values

    invalid_mask = np.logical_or(np.isinf(V_vals), np.isnan(V_vals))
    if np.any(invalid_mask):
        success = False
        return {'success': success, 'phi_train': None, 'V_train': None, 'dV_train': None, 'ddV_train': None}
    else:
        success = True
        logV_interpolator = InterpolatedUnivariateSpline(padded_phi_vals, log_V_vals)
        dlogV_dphi = logV_interpolator.derivative(n=1)(phi_vals)
        dV_dphi = dlogV_dphi * V_vals[1:-1]
        ddV_dphi = V_vals[1:-1] * (logV_interpolator.derivative(n=2)(phi_vals) + dlogV_dphi**2)
        invalid_mask_dV = np.logical_or(np.isinf(dV_dphi), np.isnan(dV_dphi))
        invalid_mask_ddV = np.logical_or(np.isinf(ddV_dphi), np.isnan(ddV_dphi))
        if np.any(invalid_mask_dV) or np.any(invalid_mask_ddV):
            success = False
        return {'success': success, 'phi_train': phi_vals, 'V_train': V_vals[1:-1], 'dV_train': dV_dphi, 'ddV_train': ddV_dphi}


@fortran_class
class QuintessenceInterp(Quintessence):
    r"""
    Quintessence Models with an interpolated potential and its derivatives.
    """

    _fields_ = [
        ("phi_train", AllocatableArrayDouble, "nodes for spline interpolation of VofPhi"),
        ("V_train", AllocatableArrayDouble, "potential V(phi) at nodes"),
        ("dV_train", AllocatableArrayDouble, "dV/dphi at nodes"),
        ("ddV_train", AllocatableArrayDouble, "d^2V/dphi^2 at nodes"),
        ("V0", c_double, "Overall potential amplitude "
                        " used for tuning to get correct DE density today"),
        ("theta_i", c_double, "phi_init initial field value"),
        ("frac_lambda0", c_double, "fraction of dark energy in cosmological constant today"),
        ("npoints", c_int, "number of points for background integration spacing"),
        ("min_steps_per_osc", c_int, "minimumum number of steps per background oscillation scale"),
        ("fde", AllocatableArrayDouble, "after initialized, the calculated background early dark energy "
                                        "fractions at sampled_a"),
        ("__ddfde", AllocatableArrayDouble),
        ("omega_tol", c_double, "tolerance for omega_DE tuning"),
        ("atol", c_double, "scale factor ")
    ] # type: ignore
    _fortran_class_name_ = 'TQuintessenceInterp'

    def set_params(self, esr_param_a0 = None, esr_param_a1 = None, esr_param_a2 = None, esr_param_a3 = None,
                    esr_functions_file='',esr_potential_index=0, phi_min=-2, phi_max=2, n_phi=250,
                   V0=1e-8, theta_i=0.0, frac_lambda0=0.):

        function_dict = load_esr_function_string(esr_functions_file, esr_potential_index)
        # print(f"Loaded ESR function dictionary with potential index {esr_potential_index} from file {esr_functions_file}: {function_dict}")

        if not function_dict['valid']:
            raise CAMBError(f"ESR function at index {esr_potential_index} is invalid.")

        esr_params = []
        if esr_param_a0 is not None:
            esr_params.append(esr_param_a0)
        if esr_param_a1 is not None:
            esr_params.append(esr_param_a1)
        if esr_param_a2 is not None:
            esr_params.append(esr_param_a2)
        if esr_param_a3 is not None:
            esr_params.append(esr_param_a3)

        esr_param_symbols = function_dict['param_symbols']
        # esr_param_names = [str(p) for p in function_dict['param_symbols']]
        # esr_function_string = function_dict['func_string']
        esr_function_template = function_dict['expr_template']
        phi_vals = np.linspace(phi_min, phi_max, n_phi)
        potential_dict = create_potential_table(esr_function_template,
                                                    esr_param_symbols,
                                                    esr_params, phi_vals)
        success = potential_dict['success']
        if not success:
            raise CAMBError("Failed to create a valid potential table from ESR function for the given esr parameters")

        phi_train = potential_dict['phi_train']
        V_train = potential_dict['V_train']
        dV_train = potential_dict['dV_train']
        ddV_train = potential_dict['ddV_train']

        self.phi_train = np.ascontiguousarray(phi_train, dtype=np.float64)
        self.V_train = np.ascontiguousarray(V_train, dtype=np.float64)
        self.dV_train = np.ascontiguousarray(dV_train, dtype=np.float64)
        self.ddV_train = np.ascontiguousarray(ddV_train, dtype=np.float64)
        self.V0 = V0
        self.theta_i = theta_i
        self.frac_lambda0 = frac_lambda0

# short names for models that support w/wa
F2003Class._class_names.update({"fluid": DarkEnergyFluid, "ppf": DarkEnergyPPF})
