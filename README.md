# stockhealth
 `stockhealth` is a Python package built for investment management and analysis of security in US stock exchange.

# Table of content

- [Models](#models)
  - [Log-Normal Stochastic Volatility Model for spot pricing](#log-normal-stochastic-volatility-model-for-spot-pricing)
  - [Heston Stochastic Volatility Model for spot pricing](#heston-stochastic-volatility-model-for-spot-pricing)
  - [Heston Stochastic Volatility Model for option pricing](#heston-stochastic-volatility-model-for-option-pricing)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Install from PyPI](#install-from-pypi)
    - [Install from source](#install-from-source)
- [Usage](#usage)
    - [Import](#import)
    - [Example](#example)



# Models

## Log-Normal Stochastic Volatility Model for spot pricing

$$\frac{dS(t)}{S(t)} = \mu(t) dt + \sigma(t, S) [1 + Y(t)] dW^{(0)}, S(0) = S_0$$

$$dY(t) = - \kappa Y(t) dt + \beta \sigma(t, S)[1 + Y(t)]dW^{(0)} + \epsilon dW^{(1)}, Y(0) = 0$$

$$\sigma(t, S) = \sigma_0 \exp\left(\frac{1}{2} \rho \eta(t) + \frac{1}{2} \eta(t) \right)$$

$$\eta(t) = \int_0^t \kappa \beta^2 e^{-\kappa (t - \tau)} dW^{(0)}(\tau)$$

$$\mu(t) = \mu_0 + \frac{\alpha}{2} \int_0^t \sigma^2(t, S) e^{-\kappa (t - \tau)} d\tau$$

$$\alpha = \frac{\beta^2}{2 \kappa}$$

$$\rho = \frac{\alpha \kappa}{\beta^2}$$

$$\epsilon = \sqrt{1 - \rho^2}$$

$$dW^{(0)} = \sqrt{\Delta t} Z_1$$

$$dW^{(1)} = \sqrt{\Delta t} (\rho Z_1 + \epsilon Z_2)$$

$$Z_1, Z_2 \sim N(0, 1)$$

## Heston Stochastic Volatility Model for spot pricing

$$\frac{dS(t)}{S(t)} = \mu(t) dt + \sigma(t, S) [1 + Y(t)] dW^{(0)}, S(0) = S_0$$

$$dY(t) = - \kappa Y(t) dt + \beta \sigma(t, S)[1 + Y(t)]dW^{(0)} + \epsilon dW^{(1)}, Y(0) = 0$$

$$\sigma(t, S) = \sigma_0 \exp\left(\frac{1}{2} \rho \eta(t) + \frac{1}{2} \eta(t) \right)$$

$$\eta(t) = \int_0^t \kappa \beta^2 e^{-\kappa (t - \tau)} dW^{(0)}(\tau)$$

$$\mu(t) = \mu_0 + \frac{\alpha}{2} \int_0^t \sigma^2(t, S) e^{-\kappa (t - \tau)} d\tau$$

$$\alpha = \frac{\beta^2}{2 \kappa}$$

$$\rho = \frac{\alpha \kappa}{\beta^2}$$

$$\epsilon = \sqrt{1 - \rho^2}$$

$$dW^{(0)} = \sqrt{\Delta t} Z_1$$

$$dW^{(1)} = \sqrt{\Delta t} (\rho Z_1 + \epsilon Z_2)$$

$$Z_1, Z_2 \sim N(0, 1)$$

## Heston Stochastic Volatility Model for option pricing

$$\frac{dS(t)}{S(t)} = \mu(t) dt + \sigma(t, S) [1 + Y(t)] dW^{(0)}, S(0) = S_0$$

$$dY(t) = - \kappa Y(t) dt + \beta \sigma(t, S)[1 + Y(t)]dW^{(0)} + \epsilon dW^{(1)}, Y(0) = 0$$

$$\sigma(t, S) = \sigma_0 \exp\left(\frac{1}{2} \rho \eta(t) + \frac{1}{2} \eta(t) \right)$$

$$\eta(t) = \int_0^t \kappa \beta^2 e^{-\kappa (t - \tau)} dW^{(0)}(\tau)$$

$$\mu(t) = \mu_0 + \frac{\alpha}{2} \int_0^t \sigma^2(t, S) e^{-\kappa (t - \tau)} d\tau$$

$$\alpha = \frac{\beta^2}{2 \kappa}$$

$$\rho = \frac{\alpha \kappa}{\beta^2}$$

$$\epsilon = \sqrt{1 - \rho^2}$$

$$dW^{(0)} = \sqrt{\Delta t} Z_1$$

$$dW^{(1)} = \sqrt{\Delta t} (\rho Z_1 + \epsilon Z_2)$$

$$Z_1, Z_2 \sim N(0, 1)$$

# Installation

## Requirements

- Python 3.6 or higher
- Numpy
- Scipy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## Install from PyPI

```bash
pip install stockhealth
```

## Install from source

```bash
git clone
cd stockhealth
pip install -e .
```

# Usage

## Import

```python
from stockhealth import *
```

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from stockhealth import *

# Parameters
S0 = 100
r = 0.05
T = 1
K = 100
sigma = 0.2
N = 1000
M = 1000
dt = T / N

# Generate paths
paths = generate_paths(S0, r, sigma, T, N, M)

# Plot paths
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, T + dt, dt), paths.T)
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.title('Paths of stock price')
plt.show()
```

# License

GNU General Public License v3.0

# Author

Siddhartha Banerjee <sidban@uwalumni.com>