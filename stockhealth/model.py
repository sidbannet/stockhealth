#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# StockHealth models
# [StockHealth GitRepo](https://github.com/sidbannet/stockhealth)
#
# Copyright 2021 [Siddhartha Banerjee](mailto:sidban@uwalumni.com)
#

from scipy.stats import norm
import numpy as np
_NUMBER_OF_TRADING_DAYS_PER_YEAR: float = 252.75


# noinspection PyPep8Naming
class BlackScholes:
    """Black-Scholes model to determine fair European options price."""

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
        :type q: Annual dividend yeild
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.__d1 = lambda sigma: (
            (
                np.log(self.S / self.K) +
                (self.r - self.q + sigma ** 2 / 2.0) * self.T
            ) / (sigma * np.sqrt(self.T))
        )
        self.__d2 = lambda sigma: (
            (
                np.log(self.S / self.K) +
                (self.r - self.q + sigma ** 2 / 2.0) * self.T
            ) / (sigma * np.sqrt(self.T)) - (sigma * np.sqrt(self.T))
        )
        self._call_values = np.vectorize(self._call_value)
        self._put_values = np.vectorize(self._put_value)
        self.values = np.vectorize(self._value)

    def _call_value(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """European Call option value given sigma."""
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        return (
            S * norm.cdf(
                self.__d1(sigma=sigma)
            ) -
            K * norm.cdf(
                self.__d2(sigma=sigma)
            ) * np.exp(-r * T)
        )

    def _put_value(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """European Put option valuation given sigma."""
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        return (
            np.exp(-r * T) * K * norm.cdf(
                - self.__d2(sigma=sigma)
            ) - S * norm.cdf(
                - self.__d1(sigma=sigma)
            )
        )

    def __call_delta(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get call delta."""
        return norm.cdf(self.__d1(sigma=sigma)) * np.exp(-self.q * self.T)

    def __call_theta(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get call theta."""
        return 0.01 * (
                - np.exp(
                    -self.q * self.T
                ) * (
                    self.S * norm.pdf(self.__d1(sigma=sigma)) * sigma
                ) / (2 * np.sqrt(self.T)) -
                self.r * self.K * np.exp(
                    -self.r * self.T) * norm.cdf(self.__d2(sigma=sigma)) +
                self.q * self.S * np.exp(
                    -self.q * self.T) * norm.cdf(self.__d1(sigma=sigma))
        )

    def __call_rho(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get call rho."""
        return 0.01 * (
            self.K * self.T * np.exp(
                -self.r * self.T
            ) * norm.cdf(self.__d2(sigma=sigma))
        )

    def __put_delta(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get put delta."""
        return - norm.cdf(-self.__d1(sigma=sigma)) * np.exp(-self.q * self.T)

    def __put_theta(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get put theta."""
        return 0.01 * (
            - np.exp(
                -self.q * self.T
            ) * (
                self.S * norm.pdf(self.__d1(sigma=sigma)) * sigma
            ) / (2 * np.sqrt(self.T)) +
            self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(
                -self.__d2(sigma=sigma)
            ) -
            self.q * self.S * np.exp(
                -self.q * self.T
            ) * norm.cdf(-self.__d1(sigma=sigma))
        )

    def __put_rho(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get put rho."""
        return 0.01 * (
                -self.K * self.T * np.exp(
                    -self.r * self.T
                ) * norm.cdf(-self.__d2(sigma=sigma))
        )

    def __gamma(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get gamma."""
        return norm.pdf(
            self.__d1(sigma=sigma)
        ) / (self.S * sigma * np.sqrt(self.T)) * np.exp(-self.q * self.T)

    def __vega(
            self,
            sigma: np.float = np.nan,
    ) -> np.float:
        """Get vega."""
        return 0.01 * (
                self.S * norm.pdf(self.__d1(sigma=sigma)) * np.sqrt(self.T)
        ) * np.exp(-self.q * self.T)

    def _sigma_call(
            self,
            price: np.float = np.nan,
    ) -> np.float:
        """Implied sigma given call option value."""
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (np.abs(self._call_values(sigma=sigma_) - price)).argmin()]

    def _sigma_put(
            self,
            price: np.float = np.nan,
    ) -> np.float:
        """Implied sigma given put option value."""
        sigma_ = np.linspace(start=0.001, stop=5.0, num=5000, endpoint=True)
        return sigma_[
            (np.abs(self._put_values(sigma=sigma_) - price)).argmin()]

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
            'call': self._value(sigma=self._sigma_call(price=call_price))[
                'call'],
            'put': self._value(sigma=self._sigma_put(price=put_price))['put'],
        }


# noinspection PyPep8Naming
class StochasticVolatility:
    """Log-normal Stochastic Volatility Model."""

    def __init__(
            self,
            mew: np.float = np.nan,
            S0: np.float = np.nan,
            volatility: np.float = np.nan,
            beta: np.float = np.nan,
            epsilon: np.float = np.nan,
            kappa: np.float = np.nan,
            number_of_instances: np.int = np.int(100000),
    ):
        """Instantiate the SV Model."""
        self.S = S0 * np.ones(shape=number_of_instances)
        self.mew = mew
        self._volatility = volatility
        self.beta = beta / volatility
        self.epsilon = epsilon / volatility
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
            self._volatility * (1 + self.Y) * self.S * dW[0]
        self.Y += \
            - self.kappa * self.Y * dt \
            + self.beta * self._volatility * (1 + self.Y) * dW[0] \
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
        return (self.Y + np.float(1)) * self._volatility


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
        self.Y += self.kappa_y * self.Y * self.sigma_y * dW_Y
        self.volatility = self.__get_volatility(self.Y)
        self.t += dt

    def reset(self) -> None:
        """Reset model states to t=0."""
        self.S = np.full_like(self.S, fill_value=self.__S)
        self.mew = np.full_like(self.mew, fill_value=self.__mew)
        self.volatility = np.full_like(self.volatility, fill_value=self.__volatility)
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
            volatility: np.float = np.nan,
            number_of_instances: np.int = np.int(10000),
    ):
        """Instantiate the Simple Stochastic model."""
        super().__init__(
            mew=mew,
            S0=S0,
            volatility=volatility,
            kappa=0,
            beta=0,
            epsilon=0,
            number_of_instances=number_of_instances,
        )
