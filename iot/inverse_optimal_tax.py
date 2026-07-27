import numpy as np
import pandas as pd
import scipy.stats as st
import scipy
import scipy.integrate
import scipy.special
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq


class IOT:
    """
    Computes social welfare weights across the income distribution given
    data, tax policy parameters, and behavioral parameters.

    Args:
        data (Pandas DataFrame): micro data representing tax payers.
            Must include the following columns: income_measure,
            weight_var, mtr
        income_measure (str): name of income measure from data to use
        weight_var (str): name of weight measure from data to use
        eti (scalar or dict): compensated elasticity of taxable income
            w.r.t. the marginal tax rate. If a dict, must have keys
            ``knot_points`` and ``eti_values`` with equal-length lists.
        dist_type (None or str): type of distribution to use if
            parametric; if None, then non-parametric bin weights
        kde_bw (scalar or None): bandwidth for KDE estimation
        mtr_smoother (None or str): method used to smooth the mtr
            function; if None, then use bin average mtrs
        mtr_smooth_param (scalar): parameter for mtr_smoother
        kreg_bw (array_like): bandwidth for kernel regression
    """

    def __init__(
        self,
        data,
        income_measure="e00200",
        weight_var="s006",
        eti=0.25,
        dist_type="log_normal",
        kde_bw=None,
        mtr_smoother="kreg",
        mtr_smooth_param=1000,
        kreg_bw=[120_000],
    ):
        # keep the original data intact
        self.data_original = data.copy()
        # clean data based on upper and lower bounds
        # data = data[
        #     (data[income_measure] >= lower_bound)
        #     & (data[income_measure] <= upper_bound)
        # ]
        # Get income distribution
        self.z, self.F, self.f, self.f_prime = self.compute_income_dist(
            data, income_measure, weight_var, dist_type, kde_bw
        )
        # see if eti is a scalar
        if isinstance(eti, (int, float)):
            self.eti = eti
        else:  # if not, then it should be a dict with keys containing lists as values
            # check that same number of ETI values as knot points
            assert len(eti["knot_points"]) == len(eti["eti_values"])
            # want to interpolate across income distribution with knot points
            # NOTE: the spline extrapolates (linearly for k=1, cubically
            # for k=3) outside the range of knot_points; values are not
            # clipped, so verify the fitted eti is sensible at the
            # extremes of the income grid
            if len(eti["knot_points"]) > 3:
                spline_order = 3
            else:
                spline_order = 1
            eti_spl = UnivariateSpline(
                eti["knot_points"], eti["eti_values"], k=spline_order, s=0
            )
            self.eti = eti_spl(self.z)
        # compute marginal tax rate schedule
        self.mtr, self.mtr_prime = self.compute_mtr_dist(
            data,
            weight_var,
            income_measure,
            mtr_smoother,
            mtr_smooth_param,
            kreg_bw,
        )
        # compute theta_z, the elasticity of the tax base
        self.theta_z = 1 + ((self.z * self.f_prime) / self.f)
        # compute the social welfare weights
        self.g_z, self.g_z_numerical = self.sw_weights()

    def df(self):
        """
        Return all vector attributes in a DataFrame format.

        Returns:
            df (Pandas DataFrame): DataFrame with all inputs/outputs
                for each income bin
        """
        dict_out = {
            "z": self.z,
            "f": self.f,
            "F": self.F,
            "f_prime": self.f_prime,
            "mtr": self.mtr,
            "mtr_prime": self.mtr_prime,
            "theta_z": self.theta_z,
            "g_z": self.g_z,
            "g_z_numerical": self.g_z_numerical,
        }
        df = pd.DataFrame.from_dict(dict_out)
        return df

    def compute_mtr_dist(
        self,
        data,
        weight_var,
        income_measure,
        mtr_smoother,
        mtr_smooth_param,
        kreg_bw,
    ):
        """
        Compute marginal tax rates over the income distribution and
        their derivative.

        Args:
            data (Pandas DataFrame): micro data representing tax payers.
                Must include the following columns: income_measure,
                weight_var, mtr
            weight_var (str): name of weight measure from data to use
            income_measure (str): name of income measure from data to use
            mtr_smoother (None or str): method used to smooth the mtr
                function; if None, then use bin average mtrs
            mtr_smooth_param (scalar): parameter for mtr_smoother
            kreg_bw (array_like): bandwidth for kernel regression

        Returns:
            tuple:
                * mtr (array_like): mean marginal tax rate for each income bin
                * mtr_prime (array_like): rate of change in marginal tax rates
                    for each income bin
        """

        if mtr_smoother == "kreg":
            bins = mtr_smooth_param  # number of equal-width bins
            data.loc[:, ["z_bin"]] = pd.cut(
                data[income_measure], bins, include_lowest=True
            )
            binned_data = pd.DataFrame(
                data[["mtr", income_measure, "z_bin", weight_var]]
                .groupby(["z_bin"], observed=False)
                .apply(lambda x: wm(x[["mtr", income_measure]], x[weight_var]))
            )
            # make column 0 into two columns
            binned_data[["mtr", income_measure]] = pd.DataFrame(
                binned_data[0].tolist(), index=binned_data.index
            )
            binned_data.drop(columns=0, inplace=True)
            binned_data.reset_index(inplace=True)
            mtr_function = KernelReg(
                binned_data["mtr"].dropna(),
                binned_data[income_measure].dropna(),
                var_type="c",
                reg_type="ll",
                bw=kreg_bw,
            )
            mtr, _ = mtr_function.fit(self.z)
            mtr_prime = np.gradient(mtr, self.z, edge_order=2)
        elif mtr_smoother == "HSV":
            # estimate the HSV function on mtrs via weighted least squares
            # DATA CLEANING
            # drop rows with missing or inf mtr
            data = data[~data["mtr"].isna()]
            data = data[~data["mtr"].isin([np.inf, -np.inf])]
            # drop if MTR > 100%
            data = data[data["mtr"] < 1]
            # drop rows with missing, inf, or zero income
            data = data[data[income_measure] > 0]
            # drop rows with missing, inf or negative weights
            data = data[~data[weight_var].isna()]
            data = data[~data[weight_var].isin([np.inf, -np.inf])]
            data = data[data[weight_var] > 0]
            # ESTIMATION
            X = np.log(data[income_measure].values)
            X = np.column_stack((np.ones(len(X)), X))
            w = np.array(data[weight_var].values)
            w_sqrt = np.sqrt(w)
            y = np.log(1 - data["mtr"].values)
            X_weighted = X * w_sqrt[:, np.newaxis]
            y_weighted = y * w_sqrt
            coef, _, _, _ = lstsq(X_weighted, y_weighted)
            tau = -coef[1]
            lambda_param = np.exp(coef[0]) / (1 - tau)
            mtr = 1 - lambda_param * (1 - tau) * self.z ** (-tau)
            mtr_prime = lambda_param * tau * (1 - tau) * self.z ** (-tau - 1)
        else:
            print("Please enter a value mtr_smoother method")
            assert False

        return mtr, mtr_prime

    def compute_income_dist(
        self, data, income_measure, weight_var, dist_type, kde_bw=None
    ):
        """
        Compute the distribution of income (parametrically or not) from
        the raw data.

        This method computes the probability density function and its
        derivative.

        Args:
            data (Pandas DataFrame): micro data representing tax payers.
                Must include the following columns: income_measure,
                weight_var, mtr
            income_measure (str): name of income measure from data to
                use
            weight_var (str): name of weight measure from data to use
            dist_type (None or str): type of distribution to use if
                parametric, if None, then non-parametric bin weights
            kde_bw (array_like): bandwidth for kernel regression

        Returns:
            tuple:
                * z (array_like): income grid points
                * F (array_like): cumulative distribution function at each z
                * f (array_like): density at each z
                * f_prime (array_like): slope of the density function at each z
        """
        z_line = np.linspace(100, 1000000, 100000)
        # drop zero income observations
        data = data[data[income_measure] > 0]
        if dist_type == "log_normal":
            mu = (
                np.log(data[income_measure]) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (
                    ((np.log(data[income_measure]) - mu) ** 2)
                    * data[weight_var]
                ).values
                / data[weight_var].sum()
            ).sum()
            # F = st.lognorm.cdf(z_line, s=(sigmasq) ** 0.5, scale=np.exp(mu))
            # f = st.lognorm.pdf(z_line, s=(sigmasq) ** 0.5, scale=np.exp(mu))
            # f = f / np.sum(f)
            # f_prime = np.gradient(f, edge_order=2)

            # analytical derivative of lognormal
            sigma = np.sqrt(sigmasq)
            F = (1 / 2) * (
                1
                + scipy.special.erf(
                    (np.log(z_line) - mu) / (np.sqrt(2) * sigma)
                )
            )
            f = (
                (1 / (sigma * np.sqrt(2 * np.pi)))
                * np.exp(-((np.log(z_line) - mu) ** 2) / (2 * sigma**2))
                * (1 / z_line)
            )
            f_prime = (
                -1
                * np.exp(-((np.log(z_line) - mu) ** 2) / (2 * sigma**2))
                * (
                    (np.log(z_line) + sigma**2 - mu)
                    / (z_line**2 * sigma**3 * np.sqrt(2 * np.pi))
                )
            )
        elif dist_type == "kde":
            # uses the original full data for kde estimation
            f_function = st.gaussian_kde(
                data[income_measure],
                # bw_method=kde_bw,
                weights=data[weight_var],
            )
            f = f_function.pdf(z_line)
            # CDF via cumulative trapezoid integration of the density
            # (np.cumsum(f) alone ignores the grid spacing dz)
            F = scipy.integrate.cumulative_trapezoid(f, z_line, initial=0)
            f_prime = np.gradient(f, z_line, edge_order=2)
        elif dist_type == "Pln":

            def mills_ratio(t):
                # R(t) = (1 - Phi(t)) / phi(t), computed with the
                # scaled complementary error function for numerical
                # stability. The naive ratio underflows to 0/0 in the
                # tails, which can zero out the fitted density and
                # poison downstream calculations (theta_z, g_z) with
                # infs/NaNs.
                return np.sqrt(np.pi / 2) * scipy.special.erfcx(t / np.sqrt(2))

            def pln_pdf(y, mu, sigma, alpha):
                x1 = alpha * sigma - (np.log(y) - mu) / sigma
                phi = st.norm.pdf((np.log(y) - mu) / sigma)
                pdf = alpha / y * phi * mills_ratio(x1)
                return pdf

            def neg_weighted_log_likelihood(params, data, weights):
                mu, sigma, alpha = params
                likelihood = np.sum(
                    weights * np.log(pln_pdf(data, mu, sigma, alpha) + 1e-15)
                )
                # 1e-15 to avoid log(0)
                return -likelihood

            def fit_pln(data, weights, initial_guess):
                bounds = [(None, None), (0.01, None), (0.01, None)]
                result = scipy.optimize.minimize(
                    neg_weighted_log_likelihood,
                    initial_guess,
                    args=(data, weights),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                return result.x

            mu_initial = (
                np.log(data[income_measure]) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (
                    ((np.log(data[income_measure]) - mu_initial) ** 2)
                    * data[weight_var]
                ).values
                / data[weight_var].sum()
            ).sum()
            sigma_initial = np.sqrt(sigmasq)
            # Initial guess for m, sigma, alpha
            initial_guess = np.array([mu_initial, sigma_initial, 1.5])
            mu, sigma, alpha = fit_pln(
                data[income_measure], data[weight_var], initial_guess
            )

            def pln_cdf(y, mu, sigma, alpha):
                x1 = alpha * sigma - (np.log(y) - mu) / sigma
                CDF = st.norm.cdf((np.log(y) - mu) / sigma) - st.norm.pdf(
                    (np.log(y) - mu) / sigma
                ) * mills_ratio(x1)
                return CDF

            def pln_dpdf(y, mu, sigma, alpha):
                x = (np.log(y) - mu) / sigma
                R = mills_ratio(alpha * sigma - x)
                left = (1 + x / sigma) * pln_pdf(y, mu, sigma, alpha)
                right = (
                    alpha
                    * st.norm.pdf(x)
                    * ((alpha * sigma - x) * R - 1)
                    / (sigma * y)
                )
                return -(left + right) / y

            f = pln_pdf(z_line, mu, sigma, alpha)
            F = pln_cdf(z_line, mu, sigma, alpha)
            f_prime = pln_dpdf(z_line, mu, sigma, alpha)
        else:
            print("Please enter a valid value for dist_type")
            assert False

        return z_line, F, f, f_prime

    def sw_weights(self):
        r"""
        Return the social welfare weights for a given tax policy.

        See Jacobs, Jongen, and Zoutman (2017) and
        Lockwood and Weinzierl (2016) for details.

        .. math::
            g_{z} = 1 + \theta_z \varepsilon^{c}\frac{T'(z)}{1-T'(z)} +
            \varepsilon^{c}\frac{zT''(z)}{(1-T'(z))^{2}}

        Returns:
            tuple:
                * g_z (array_like): social welfare weights via analytical
                    formula
                * g_z_numerical (array_like): social welfare weights via
                    the Lockwood and Weinzierl numerical formula
        """
        g_z = (
            1
            + ((self.theta_z * self.eti * self.mtr) / (1 - self.mtr))
            + ((self.eti * self.z * self.mtr_prime) / (1 - self.mtr) ** 2)
        )
        integral = np.trapz(
            g_z * self.f, self.z
        )  # renormalize to integrate to 1
        g_z = g_z / integral

        # use Lockwood and Weinzierl formula, which should be equivalent but using numerical differentiation
        bracket_term = (
            1
            - self.F
            - (self.mtr / (1 - self.mtr)) * self.eti * self.z * self.f
        )
        # differentiate wrt z (must pass self.z; np.gradient otherwise
        # assumes unit spacing and scales the result by 1/dz)
        d_dz_bracket = np.gradient(bracket_term, self.z, edge_order=2)
        # d_dz_bracket = np.diff(bracket_term) / np.diff(self.z)
        # d_dz_bracket = np.append(d_dz_bracket, d_dz_bracket[-1])
        g_z_numerical = -(1 / self.f) * d_dz_bracket
        integral = np.trapz(g_z_numerical * self.f, self.z)
        g_z_numerical = g_z_numerical / integral

        return g_z, g_z_numerical


def find_eti(iot, g_z=None, eti_0=0.25, boundary="z0"):
    """
    This function solves for the ETI that would result in the
    policy represented via MTRs in IOT being consistent with the
    social welfare function supplied. It solves a first order
    ordinary differential equation.

    .. math::
            \varepsilon'(z)\left[\frac{zT'(z)}{1-T'(z)}\right] + \varepsilon(z)\left[\theta_z  \frac{T'(z)}{1-T'(z)} +\frac{zT''(z)}{(1-T'(z))^2}\right]+ (1-g(z))

    Args:
        iot (IOT): instance of the IOT class
        g_z (None or array_like): vector of social welfare weights
        eti_0 (scalar): guess for ETI at z=0 (used when boundary="z0")
        boundary (str): "z0" to use the initial condition at z=0
            (solves the ODE with eti_0), or "inf" to use the
            transversality condition that
            epsilon(z)*T'(z)/(1-T'(z))*z*f(z) -> 0 as z -> inf.

    Returns:
        eti_beliefs (array-like): vector of ETI beliefs over z
    """
    if g_z is None:
        g_z = iot.g_z

    if boundary == "z0":
        # Original ODE approach with boundary condition at the lowest
        # grid point z_min (so that eti(z_min) = eti_0 exactly).
        # Use cumulative trapezoid integration starting at 0 rather
        # than np.cumsum, which is both less accurate and makes the
        # integrating factor mu(z_min) != 1 (shifting the boundary
        # condition off of eti_0).
        P_z = (
            1 / iot.z
            + iot.f_prime / iot.f
            + iot.mtr_prime / (iot.mtr * (1 - iot.mtr))
        )
        mu_z = np.exp(
            scipy.integrate.cumulative_trapezoid(P_z, iot.z, initial=0)
        )
        Q_z = (g_z - 1) * (1 - iot.mtr) / (iot.mtr * iot.z)
        int_mu_Q = scipy.integrate.cumulative_trapezoid(
            mu_z * Q_z, iot.z, initial=0
        )
        eti_beliefs = (eti_0 + int_mu_Q) / mu_z

    elif boundary == "inf":
        # Transversality condition: eps(z)*T'/(1-T')*z*f -> 0 as z -> inf
        # eps(z) = [(1-T'(z))/T'(z)] * [1/(z*f(z))] * int_z^inf (1 - g(zt)) f(zt) dzt
        # CAUTION: the integral is truncated at the top of the income
        # grid (z_max), so the mass of int_{z_max}^inf (1-g) f dz is
        # dropped. This biases the implied eti toward zero as z
        # approaches z_max (verified via a self-consistency test in
        # which passing the model's own raw g_z should return the
        # constant eti used to generate it). Interpret results near
        # the top of the grid, and comparisons between the two
        # boundary conditions, with this truncation bias in mind.
        integrand = (1 - g_z) * iot.f
        # Reverse cumulative integral: int_z^inf = int_0^inf - int_0^z
        # Compute using reverse cumsum of trapezoid contributions
        # Use cumulative_trapezoid from the right
        dz = np.diff(iot.z)
        # Trapezoidal contributions for each interval
        trap_contributions = 0.5 * (integrand[:-1] + integrand[1:]) * dz
        # Reverse cumulative sum to get integral from z to z_max
        rev_cumsum = np.flip(np.cumsum(np.flip(trap_contributions)))
        # Append 0 for the last point (integral from z_max to inf ~ 0)
        tail_integral = np.append(rev_cumsum, 0.0)

        eti_beliefs = (
            ((1 - iot.mtr) / iot.mtr) * (1 / (iot.z * iot.f)) * tail_integral
        )
    else:
        raise ValueError(f"boundary must be 'z0' or 'inf', got '{boundary}'")

    return eti_beliefs


def wm(value, weight):
    """
    Weighted mean function that allows for zero division

    Args:
        value (array_like): values to be averaged
        weight (array_like): weights for each value

    Returns:
        scalar: weighted average
    """
    try:
        return np.average(value, weights=weight, axis=0)
    except ZeroDivisionError:
        return [np.nan, np.nan]
