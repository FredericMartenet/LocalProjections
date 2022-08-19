from typing import Any, Union
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings


class LocalProjections:
    """
    Local projections model (Jorda 2004).

    Estimates the impact of an exogenous variable on
    the endogenous variable using local projections
    as in Jorda (2004):
    .. math::
        y_{t+h} = \beta exog_t + \gamma controls_{t} + e_t
    where control variables can also include lagged values
    of the endogenous and exogenous variables. One regression
    is estimated for each horizon h in [0,1,..,H].

    Parameters
    ----------
    dta : pd.DataFrame
        DataFrame containg the data
    endog_name : str
        Name of the endogenous variable (must be stored in a column of dta)
    exog_name : str
        Name of the exogenous variable (must be stored in a column of dta)
    controls_names : list, default []
        Name of all the control variables (must be stored in columns of dta)

    Examples
    --------
    Estimating the response of real GDP following a rise in the
    nominal interest rate.

    Import dataset
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> dta = sm.datasets.macrodata.load_pandas().data

    Create nominal interest rate and logs
    >>> dta["i"] = dta["realint"] + dta["infl"]
    >>> for var in ["realgdp", "cpi"]:
    >>>     dta[f"{var}_log"] = np.log(dta[var])

    Set endogenous, exogenous and control variables
    >>> endog = "realgdp_log"
    >>> exog = "i"
    >>> controls = ["cpi_log", "unemp"]

    Estimate results
    >>> lp = LocalProjections(dta, endog, exog, controls)
    >>> irf = lp.fit(
    >>>     H=16,
    >>>     contemporaneous_control=1,
    >>>     scale=100,
    >>>     lags_exog=2,
    >>>     ylabel="Percent of real GDP",
    >>>     displaylag=True,
    >>> )

    References
    ----------
    Jorda (2005). Estimation and Inference of Impulse Responses
    by Local Projections. American Economic Review. 95(1). pp. 161-182
    """
    def __init__(
            self,
            dta: pd.DataFrame,
            endog_name: str,
            exog_name: str,
            controls_names: list = [],
    ) -> None:
        self.dta = self._is_valid_dta(dta)
        self.endog = self._is_valid_endog(endog_name)
        self.controls = self._is_valid_controls(controls_names)
        self.exog = self._is_valid_exog(exog_name)

    def _is_valid_dta(self, dta):
        if type(dta) is not pd.DataFrame:
            raise TypeError(f"'dta' is not a Pandas DataFrame")

        return dta

    def _is_valid_endog(self, endog):
        if endog not in self.dta:
            raise ValueError(f"'{endog}' is not a column of dta")

        return endog

    def _is_valid_controls(self, controls):
        for var in controls:
            if var not in self.dta:
                raise ValueError(f"'{var}' is not a column of dta")

        return controls

    def _is_valid_exog(self, exog):
        if exog not in self.dta:
            raise ValueError(f"'{exog}' is not a column of dta")
        # remove exog if it is also in list of controls to avoid duplication
        if exog in self.controls:
            self.controls.remove(exog)
            warnings.warn(f"'{exog}' in the list of controls was ignored")

        return exog

    def fit(
            self,
            contemporaneous_control: int = 0,
            lag_selection: Union[str, None] = "AIC",
            h_selectlag: int = 0,
            maxlagselect: int = 12,
            displaylag: bool = False,
            lags_control: int = 1,
            lags_endog: int = 1,
            lags_exog: int = 1,
            H: int = 12,
            alpha: list = [.01, .05, .1],
            cov_type: str = 'HAC',
            cov_kwds: Union[dict[str, Any], None] = {'maxlags': 6, 'kernel': 'bartlett', 'use_correction': True},
            scale: float = 1.0,
            show_plot: bool = True,
            xlabel: str = r"h",
            ylabel: str = "",
    ) -> pd.DataFrame:
        """
        Estimates local projections at all horizons h=0,1,..,H and store results.

        Parameters
        ----------
        contemporaneous_control: int
            Whether to include contemporaneous values of the control
            variables (set to 1) or not (set to 0).
        lag_selection: Union[str, None], default "AIC"
            Desired criterion for lag selection ("AIC", "BIC" or "R2_adj"), if
            None no lag selction is performed.
        h_selectlag: int, default 0
            Horizon of the regression for lag selection.
        maxlagselect: int, default 12
            Maximum number of lags to consider.
        displaylag: bool, default False
            Display selected number of lags or not.
        lags_control: int
            Number of lags of control variables to include.
        lags_endog: int
            Number of lags of endogenous variable to include.
        lags_exog: int
            Number of lags of exogenous variable to include.
        H: int
            Maximum horizon to which estimate projections.
        alpha: list, default [.01, .05, .1]
            Significance level for confidence intervals.
        cov_type: str, default "HAC"
            Covariance estimator to use. Default is set to Heteroskedasticity-
            autocorrelation robust covariance ("HAC") to account for serial
            auto-correlation in residuals. Can also use "nonrobust" or "HC0/1/2/3".
        cov_kwds: Union[dict[str, Any], None], default {'maxlags':6, 'kernel': 'bartlett', 'use_correction': True}
            Keyword arguments for the covariance estimator. "nonrobust" and
            "HC#" do not support cov_kwds.
        scale: float, default 1.0
            Scaling of the results.
        show_plot: bool, default True
            Draw plot of impulse response function or not.
        xlabel: str, default "h"
            x-axis label.
        ylabel: str, default ""
            y-axis label.

        Returns
        -------
        irf: pd.DataFrame (H x (1 + 2*len(alpha)))
            DataFrame containing the point estimates, upper, and lower confidence
        """
        # add necessary lags and leads
        self.regdta = self.make_leads_lags(
            max(lags_control, maxlagselect),
            max(lags_endog, maxlagselect),
            lags_exog,
            H
        )

        # perform lag selection
        if lag_selection:
            self.n_lags = self.select_lags(
                contemporaneous_control,
                lags_exog,
                h_selectlag=h_selectlag,
                criterion=lag_selection,
                maxlagselect=maxlagselect,
                displaylag=displaylag,
            )
            lags_endog, lags_control = self.n_lags, self.n_lags

        # initialize DataFrame to store results
        irf = pd.DataFrame(
            index=range(H + 1),
            columns=["IRF"] + [f"IRF_u{alf}" for alf in alpha] + [f"IRF_l{alf}" for alf in alpha]
        )
        # run regressions for each horizon and store results
        for h in range(H + 1):
            reg = self.fit_h(
                h,
                contemporaneous_control,
                lags_control,
                lags_endog,
                lags_exog,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
            )
            irf.loc[h]["IRF"] = reg.params[self.exog] * scale
            for alf in alpha:
                irf.loc[h][f"IRF_l{alf}"] = reg.conf_int(alpha=alf)[0][self.exog] * scale
                irf.loc[h][f"IRF_u{alf}"] = reg.conf_int(alpha=alf)[1][self.exog] * scale

        # draw plot of impulse response function
        if show_plot:
            self.plot_irf(
                irf,
                alpha,
                xlabel=xlabel,
                ylabel=ylabel,
            )

        return irf

    def make_leads_lags(
            self,
            lags_control: int,
            lags_endog: int,
            lags_exog: int,
            H: int,
    ) -> pd.DataFrame:
        """Constructs the lags and leads of necessary variables."""
        self.dta = self.make_lags(lags_control, lags_endog, lags_exog)
        self.dta = self.make_leads(H)

        return self.dta

    def make_lags(
            self,
            lags_control: int,
            lags_endog: int,
            lags_exog: int,
    ) -> pd.DataFrame:
        """
        Constructs the lags of the endngeous, exogenous and
        control variables and store in dta DataFrame.

        Parameters
        ----------
        lags_control: int
            Number of lags of the control variables to create
        lags_endog: int
            Number of lags of the endogenous variables to create
        lags_exog: int
            Number of lags of the exogenous variables to create

        Returns
        -------
        pd.DataFrame
            dta DataFrame including lagged variables
        """
        for lag in range(lags_control + 1):
            for var in self.controls:
                self.dta[f"{var}_L{lag}"] = self.dta[var].shift(lag)

        for lag in range(lags_endog + 1):
            self.dta[f"{self.endog}_L{lag}"] = self.dta[self.endog].shift(lag)

        for lag in range(lags_exog + 1):
            self.dta[f"{self.exog}_L{lag}"] = self.dta[self.exog].shift(lag)

        return self.dta

    def make_leads(self, H: int) -> pd.DataFrame:
        """
        Constructs the lead values of the endngeous and store in dta DataFrame.

        Parameters
        ----------
        H: int
            Maximum horizon to which estimate projections.

        Returns
        -------
        pd.DataFrame
            dta DataFrame including lead values of endogenous variable
        """
        for lead in range(H + 1):
            self.dta[f"{self.endog}_h{lead}"] = self.dta[self.endog].shift(-lead)

        return self.dta

    def select_lags(
            self,
            contemporaneous_control: int,
            lags_exog: int,
            h_selectlag: int = 0,
            criterion: str = "AIC",
            maxlagselect: int = 12,
            displaylag: bool = False,
    ) -> int:
        """
        Selects the number of lags of control variables and
        endogenous variables using criterion AIC, BIC or R2
        from regression for a given horizon `h`.

        Parameters
        ----------
        contemporaneous_control: int
            Whether to include contemporaneous values of the control
            variables (set to 1) or not (set to 0).
        lags_exog: int
            Number of lags of exogenous variable to include.
        h_selectlag: int, default 0
            Horizon of the regression
        criterion: str, default "AIC"
            Desired criterion of the selection ("AIC", "BIC" or "R2_adj")
        maxlagselect: int, default 12
            Maximum number of lags to consider.
        displaylag: bool, default False
            Display selected number of lags or not.

        Returns
        -------
        n_lags: int
            Best number of lags ot include according to desired criterion.

        Raises
        ------
        ValueError
            If criterion is not "AIC", "BIC" or "R2_adj"
        """
        crit = np.inf
        for lag in range(maxlagselect):
            reg = self.fit_h(
                h_selectlag,
                contemporaneous_control,
                lag,
                lag,
                lags_exog,
            )
            if criterion == "BIC":
                if reg.bic <= crit:
                    crit = reg.bic
                    n_lags = lag
            elif criterion == "AIC":
                if reg.aic <= crit:
                    crit = reg.aic
                    n_lags = lag
            elif criterion == "R2_adj":
                if reg.rsquared_adj <= crit:
                    crit = reg.rsquared_adj
                    n_lags = lag
            else:
                raise ValueError('criterion must be set to "AIC", "BIC" or "R2_adj"')

        if displaylag:
            print(f"{n_lags} lags selected using the {criterion} criterion")

        return n_lags

    def fit_h(
            self,
            h: int,
            contemporaneous_control: int,
            lags_control: int,
            lags_endog: int,
            lags_exog: int,
            cov_type: str = 'HAC',
            cov_kwds: Union[dict[str, Any], None] = {'maxlags': 6, 'kernel': 'bartlett', 'use_correction': True},
    ) -> linear_model.RegressionResults:
        """
        Estimates regression at horizon `h` and returns statsmodel regression results.

        Parameters
        ----------
        h: int
            Horizon of regression to estimate.
        contemporaneous_control: int
            Whether to include contemporaneous values of the control
            variables (set to 1) or not (set to 0).
        lags_control: int
            Number of lags of control variables to include.
        lags_endog: int
            Number of lags of endogenous variable to include.
        lags_exog: int
            Number of lags of exogenous variable to include.
        cov_type: str, default "HAC"
            Covariance estimator to use. Default is set to Heteroskedasticity-
            autocorrelation robust covariance ("HAC") to account for serial
            auto-correlation in residuals. Can also use "nonrobust" or "HC0/1/2/3".
        cov_kwds: Union[dict[str, Any], None], default {'maxlags':6, 'kernel': 'bartlett', 'use_correction': True}
            Keyword arguments for the covariance estimator. "nonrobust" and
            "HC#" do not support cov_kwds.

        Returns
        -------
        reg: sm.regression.linear_model.RegressionResults
            Regression results.
        """
        eqtn = self.make_equation(
            h,
            contemporaneous_control,
            lags_control,
            lags_endog,
            lags_exog,
        )
        reg = (
            smf.ols(eqtn, data=self.regdta)
                .fit(cov_type=cov_type, cov_kwds=cov_kwds)
        )

        return reg

    def make_equation(
            self,
            h: int,
            contemporaneous_control: int,
            lags_control: int,
            lags_endog: int,
            lags_exog: int,
    ) -> str:
        """
        Creates the string regression equation to be used using
        the statsmodels api for a given horizon `h`.

        Parameters
        ----------
        h: int
            Horizon of the regression
        contemporaneous_control: int
            Whether to include contemporaneous values of the control
            variables (set to 1) or not (set to 0).
        lags_control: int
            Number of lags of control variables to include.
        lags_endog: int
            Number of lags of endogenous variable to include.
        lags_exog: int
            Number of lags of exogenous variable to include.

        Returns
        -------
        eqtn, str
            Equation in string format to be read by statsmodels api.

        Raises
        ------
        ValueError
            If contemporaneous_control is not 0 or 1.
        """
        if not ((contemporaneous_control == 1) | (contemporaneous_control == 0)):
            raise ValueError("'contemporaneous_control' must be either 0 or 1")

        lagcontrols = "".join([
            f" + {var}_L{lag}"
            for var in self.controls
            for lag in range((1 - contemporaneous_control), lags_control + 1)
        ])

        lagendog = "".join([
            f" + {self.endog}_L{lag}" for lag in range(1, lags_endog + 1)
        ])

        lags_exog = f"{self.exog}" + "".join([
            f" + {self.exog}_L{lag}" for lag in range(1, lags_exog + 1)
        ])

        eqtn = f"{self.endog}_h{h} ~ {lags_exog} {lagcontrols} {lagendog}"

        return eqtn

    def plot_irf(
            self,
            irf: pd.DataFrame,
            alpha: list,
            xlabel: str = r"h",
            ylabel: str = "",
    ) -> None:
        """
        Draws impulse response function plot from results.

        Parameters
        ----------
        irf: pd.DataFrame
            DataFrame of results.
        alpha: list
            List of significance levels for confidence intervals in `irf`.
        xlabel: str, default "h"
            x-axis label.
        ylabel: str, default ""
            y-axis label.
        """
        for alf in alpha:
            plt.fill_between(
                irf.index,
                irf[f"IRF_l{alf}"].astype(float),
                irf[f"IRF_u{alf}"].astype(float),
                alpha=0.1,
                color="k",
            )
        plt.plot(
            irf["IRF"],
            c="k",
        )
        plt.axhline(0, c='k', linewidth=1)
        plt.ylabel(ylabel), plt.xlabel(xlabel)
        plt.tight_layout()
        plt.show()
