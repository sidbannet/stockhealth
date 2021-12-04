# stockhealth
 `stockhealth` is a Python package built for investment management and analysis of security in US stock exchange.

# Table of content

# Models

## Log-Normal Stochastic Volatility Model for spot pricing

$$\frac{dS(t)}{S(t)} = \mu(t) dt + \sigma(t, S) [1 + Y(t)] dW^{(0)}, S(0) = S_0$$

$$dY(t) = - \kappa Y(t) dt + \beta \sigma(t, S)[1 + Y(t)]dW^{(0)} + \epsilon dW^{(1)}, Y(0) = 0$$