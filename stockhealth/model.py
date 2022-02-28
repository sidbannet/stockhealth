#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth models
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from scipy.stats import norm
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
_NUMBER_OF_TRADING_DAYS_PER_YEAR: float = 252.75
_NUMBER_OF_CALENDAR_DAYS_PER_YEAR: float = 365.2425


# noinspection PyPep8Naming,PyUnresolvedReferences
class BlackScholes:
    """Black-Scholes model to determine fair European options price."""

    @staticmethod
    def __f1(
        sigmas: np.float,
        Ss: np.float,
        Ks: np.float,
        Ts: np.float,
        rs: np.float,
        qs: np.float,
    ) -> np.float:
        return (
            (
                np.log(Ss / Ks) +
                (rs - qs + sigmas ** 2 / 2.0) * Ts
            ) / (sigmas * np.sqrt(Ts))
        )

    @staticmethod
    def __f2(
        sigmas: np.float,
        Ss: np.float,
        Ks: np.float,
        Ts: np.float,
        rs: np.float,
        qs: np.float
    ) -> np.float:
        return (
            (
                np.log(Ss / Ks) +
                (rs - qs + sigmas ** 2 / 2.0) * Ts
            ) / (sigmas * np.sqrt(Ts)) - (sigmas * np.sqrt(Ts))
        )

    def _call_price(
        self,
        S: np.float = np.nan,
        T: np.float = np.nan,
        K: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = 0.0,
        sigma: np.float = np.nan,
    ) -> np.float:
        """European Call option price given sigma, time and interest rate."""
        return (
            S * norm.cdf(
                self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            ) -
            K * norm.cdf(
                self.__f2(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            ) * np.exp(-r * T)
        )

    def _put_price(
        self,
        S: np.float = np.nan,
        T: np.float = np.nan,
        K: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """European Put option price given sigma, time and interest rate."""
        return (
            np.exp(-r * T) * K * norm.cdf(
                - self.__f2(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            ) - S * norm.cdf(
                - self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            )
        )

    def _call_delta(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get call delta."""
        return norm.cdf(
            self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
        ) * np.exp(-q * T)

    def _call_theta(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get call theta."""
        return (
            - np.exp(
                -q * T
            ) * (
                S * norm.pdf(
                    self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
                ) * sigma
            ) / (2 * np.sqrt(T)) -
            r * K * np.exp(-r * T) *
            norm.cdf(self.__f2(sigmas=sigma, Ss=S, Ts=T, Ks=K, rs=r, qs=q))
            + q * S * np.exp(-q * T) *
            norm.cdf(self.__f1(sigmas=sigma, Ss=S, Ts=T, Ks=K, rs=r, qs=q))
        )

    def _call_rho(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get call rho."""
        return (
            K * T * np.exp(
                -r * T
            ) * norm.cdf(self.__f2(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q))
        )

    def _put_delta(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get put delta."""
        return - norm.cdf(
            -self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
        ) * np.exp(-q * T)

    def _put_theta(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get put theta."""
        return (
            - np.exp(
                -q * T
            ) * (
                S * norm.pdf(
                    self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
                ) * sigma
            ) / (2 * np.sqrt(T)) +
            r * K * np.exp(-r * T) * norm.cdf(
                -self.__f2(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            ) -
            q * S * np.exp(
                -q * T
            ) * norm.cdf(
                -self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            )
        )

    def _put_rho(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get put rho."""
        return (
            -K * T * np.exp(
                -r * T
            ) * norm.cdf(
                -self.__f2(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            )
        )

    def _gamma(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get gamma."""
        return norm.pdf(
            self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
        ) / (S * sigma * np.sqrt(T)) * np.exp(-q * T)

    def _vega(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        sigma: np.float = np.nan,
    ) -> np.float:
        """Get vega."""
        return (
            S * norm.pdf(
                self.__f1(sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
            ) * np.sqrt(T)
        ) * np.exp(-q * T)

    def _sigma_call(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        price: np.float = np.nan,
    ) -> np.float:
        """Implied sigma given call option value."""
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (
                np.abs(
                    self._call_price(
                        sigma=sigma_, S=S, K=K, T=T, r=r, q=q
                    ) - price
                )
            ).argmin()
        ]

    def _sigma_put(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
        price: np.float = np.nan,
    ) -> np.float:
        """Implied sigma given put option value."""
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (
                np.abs(
                    self._put_price(
                        sigma=sigma_, S=S, K=K, T=T, r=r, q=q
                    ) - price
                )
            ).argmin()
        ]


# noinspection PyPep8Naming,PyUnresolvedReferences
class European(BlackScholes):
    """Fair European options price based on Black Scholes."""

    def __init__(
        self,
        S: np.float = np.nan,
        K: np.float = np.nan,
        T: np.float = np.nan,
        r: np.float = np.nan,
        q: np.float = np.float(0.0),
    ):
        """Instantiate the Black-Scholes model for options price.
        :type S: Spot price of the underlying asset
        :type K: Strike price
        :type T: Time to maturity (unit less fraction of one year)
        :type r: Risk free interest
        :type q: Annual dividend yield
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.__d1 = lambda sigma: self.__f1(
            sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
        self.__d2 = lambda sigma: self.__f2(
            sigmas=sigma, Ss=S, Ks=K, Ts=T, rs=r, qs=q)
        self._call_value = lambda sigma: self._call_price(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self._put_value = lambda sigma: self._put_price(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__call_delta = lambda sigma: self._call_delta(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__put_delta = lambda sigma: self._put_delta(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__call_theta = lambda sigma: self._call_theta(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__put_theta = lambda sigma: self._put_theta(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__call_rho = lambda sigma: self._call_rho(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__put_rho = lambda sigma: self._put_rho(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__gamma = lambda sigma: self._gamma(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__vega = lambda sigma: self._vega(
            sigma=sigma, S=S, K=K, T=T, r=r, q=q)
        self.__sigma_call = lambda price: self._sigma_call(
            S=S, K=K, T=T, r=r, q=q, price=price)
        self.__sigma_put = lambda price: self._sigma_put(
            S=S, K=K, T=T, r=r, q=q, price=price)
        self._call_values = np.vectorize(self._call_value)
        self._put_values = np.vectorize(self._put_value)
        self.values = np.vectorize(self._value)
        self.sigma_call__ = self.__sigma_call
        self.sigma_put__ = self.__sigma_put

    def _value(
        self,
        sigma: np.float = np.nan,
    ) -> dict:
        """Given sigma, the fair market value."""
        __call_value = self._call_value(sigma=sigma)
        __put_value = self._put_value(sigma=sigma)
        __call_delta = self.__call_delta(sigma=sigma)
        __put_delta = self.__put_delta(sigma=sigma)
        __call_theta = self.__call_theta(sigma=sigma)
        __put_theta = self.__put_theta(sigma=sigma)
        __call_rho = self.__call_rho(sigma=sigma)
        __put_rho = self.__put_rho(sigma=sigma)
        __gamma = self.__gamma(sigma=sigma)
        __vega = self.__vega(sigma=sigma)
        __call_intrinsic = max(self.S - self.K, 0)
        __call_extrinsic = __call_value - __call_intrinsic
        __put_intrinsic = max(self.K - self.S, 0)
        __put_extrinsic = __put_value - __put_intrinsic
        return {
            'call': {
                'value': __call_value,
                'delta': __call_delta,
                'gamma': __gamma,
                'vega': __vega,
                'theta': __call_theta,
                'rho': __call_rho,
                'intrinsic': __call_intrinsic,
                'extrinsic': __call_extrinsic,
            },
            'put': {
                'value': __put_value,
                'delta': __put_delta,
                'gamma': __gamma,
                'vega': __vega,
                'theta': __put_theta,
                'rho': __put_rho,
                'intrinsic': __put_intrinsic,
                'extrinsic': __put_extrinsic,
            },
        }

    def greeks(
        self,
        call_price: np.float = np.nan,
        put_price: np.float = np.nan,
    ) -> dict:
        """Get sigmas and calculate greeks for given options price."""
        return {
            'call': self._value(
                sigma=self.__sigma_call(price=call_price))['call'],
            'put': self._value(sigma=self.__sigma_put(price=put_price))['put'],
        }


# noinspection PyPep8Naming
class StochasticVolatility:
    """Log-normal Stochastic Volatility Model."""

    def __init__(
        self,
        mew: np.float = np.nan,
        S0: np.float = np.nan,
        sigma: np.float = np.nan,
        beta: np.float = np.nan,
        epsilon: np.float = np.nan,
        kappa: np.float = np.nan,
        number_of_instances: np.int = np.int(100000),
    ):
        """Instantiate the SV Model."""
        self.S = S0 * np.ones(shape=number_of_instances)
        self.mew = mew
        self.sigma = sigma
        self.beta = beta / sigma
        self.epsilon = epsilon / sigma
        self.kappa = kappa
        self.Y = np.float(0) + np.zeros(shape=number_of_instances)
        self.t = np.float(0)
        self.__N = number_of_instances
        self.__S0 = S0
        # //todo: assert if the model is setup correctly

    def update(
            self,
            dt: np.float = np.float(1 / _NUMBER_OF_TRADING_DAYS_PER_YEAR),
    ) -> None:
        """Update states and proceed forward in time with random walk."""
        dW = [
            np.random.normal(loc=0, scale=np.sqrt(dt), size=self.__N),
            np.random.normal(loc=0, scale=np.sqrt(dt), size=self.__N),
        ]
        self.S += self.mew * self.S * dt + \
            self.sigma * (1 + self.Y) * self.S * dW[0]
        self.Y += \
            - self.kappa * self.Y * dt \
            + self.beta * self.sigma * (1 + self.Y) * dW[0] \
            + self.epsilon * dW[1]
        self.t += dt

    def reset(self) -> None:
        """Reset model states to t=0."""
        self.S = self.__S0 * np.ones_like(self.S)
        self.Y = np.zeros_like(self.Y)
        self.t = np.float(0)

    @property
    def volatility(self) -> np.array:
        """Give stock price volatility state."""
        return (self.Y + np.float(1)) * self.sigma


# noinspection PyPep8Naming
class Heston:
    """Log-Normal Stochastic Volatility Model with mean reversion."""

    def __init__(
        self,
        S0: np.float = np.nan,
        mew0: np.float = np.nan,
        V0: np.float = np.nan,
        historical_roi: np.float = np.nan,
        historical_volatility: np.float = np.nan,
        mean_reversion_roi: np.float = np.nan,
        mean_reversion_log_volatility: np.float = np.nan,
        sigma_mew: np.float = np.nan,
        sigma_y: np.float = np.nan,
        correlation: np.float = np.float(0),
        number_of_instances: np.int = np.int(100000),
    ):
        """Instantiate the Heston Model."""
        self.S = S0 * np.ones(shape=number_of_instances)
        self.mew = mew0 * np.ones(shape=number_of_instances)
        self.volatility = V0 * np.ones(shape=number_of_instances)
        self.mew_hat = historical_roi
        self.sigma_hat = historical_volatility
        self.rho = correlation
        self.kappa_mew = mean_reversion_roi
        self.kappa_y = mean_reversion_log_volatility
        self.sigma_mew = sigma_mew
        self.sigma_y = sigma_y
        self.__N = number_of_instances
        self.__get_Y = np.vectorize(
            lambda volatility: np.log(volatility / historical_volatility)
        )
        self.__get_volatility = np.vectorize(
            lambda Y: historical_volatility * np.exp(Y)
        )
        self.Y = self.__get_Y(self.volatility)
        self.t = np.float(0)
        self.__S = S0
        self.__mew = mew0
        self.__volatility = V0

    def update(
        self,
        dt: np.float = np.float(1 / _NUMBER_OF_TRADING_DAYS_PER_YEAR),
    ) -> None:
        """Update states and proceed forward in time with random walks."""
        dW_mew, dW_Y = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[dt, dt * self.rho], [dt * self.rho, dt]],
            size=self.__N,
        ).T
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=self.__N)
        self.S += self.mew * self.S * dt + \
            self.volatility * self.S * dW
        self.mew += self.kappa_mew * (self.mew - self.mew_hat) * dt + \
            self.sigma_mew * dW_mew
        self.Y += self.kappa_y * self.Y * dt + self.sigma_y * dW_Y
        self.volatility = self.__get_volatility(self.Y)
        self.t += dt

    def reset(self) -> None:
        """Reset model states to t=0."""
        self.S = np.full_like(self.S, fill_value=self.__S)
        self.mew = np.full_like(self.mew, fill_value=self.__mew)
        self.volatility = np.full_like(
            self.volatility, fill_value=self.__volatility)
        self.Y = self.__get_Y(self.volatility)


# noinspection PyPep8Naming
class SimpleStochastic(StochasticVolatility):
    """
    Stochastic log-normal time dynamics model with constant volatility.
    """

    def __init__(
        self,
        mew: np.float = np.nan,
        S0: np.float = np.nan,
        sigma: np.float = np.nan,
        number_of_instances: np.int = np.int(10000),
    ):
        """Instantiate the Simple Stochastic model."""
        super().__init__(
            mew=mew,
            S0=S0,
            sigma=sigma,
            kappa=0,
            beta=0,
            epsilon=0,
            number_of_instances=number_of_instances,
        )


class VolatilitySmile:
    """Train a volatility smile model and obtain Huber regressor."""

    def __init__(
        self,
        chain_data: pd.DataFrame,
        current_stock_price: float,
        volatility_measure: str = 'impliedVolatility',
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: float = 1e-5
    ) -> None:
        """Instantiate the model object and get hubber regressor."""
        self.current_stock = current_stock_price
        self.__huber_above_current = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol,
        ).fit(
            X=np.reshape(
                chain_data[current_stock_price:].index - current_stock_price,
                (-1, 1)
            ),
            y=chain_data[volatility_measure][current_stock_price:].values,
        )
        self.__huber_below_current = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol,
        ).fit(
            X=np.reshape(
                chain_data[:current_stock_price].index - current_stock_price,
                (-1, 1)
            ),
            y=chain_data[volatility_measure][:current_stock_price].values,
        )

    @property
    def slopes(self) -> np.array:
        """Get the slopes of the two legs."""
        return np.array(
            [
                np.append(
                    self.__huber_below_current.coef_,
                    0.0,
                ).min(),
                np.append(
                    self.__huber_above_current.coef_,
                    0.0,
                ).max()
            ],
        )
        self.__huber_above_current.coef_
