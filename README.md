# stockhealth
 `stockhealth` is a Python package built for investment management and analysis of security in US stock exchange.

# Table of content

# Models

## Log-Normal Stochastic Volatility Model for spot pricing

<!--suppress CheckDtdRefs -->
<img src="https://latex.codecogs.com/gif.latex?\frac{dS(t)}{S(t)}&space;=&space;\mu(t)&space;dt&space;&plus;&space;\sigma(t,&space;S)&space;[1&space;&plus;&space;Y(t)]&space;dW^{(0)},&space;S(0)&space;=&space;S_0" title="\frac{dS(t)}{S(t)} = \mu(t) dt + \sigma(t, S) [1 + Y(t)] dW^{(0)}, S(0) = S_0" />

<img src="https://latex.codecogs.com/png.latex?\inline&space;\dpi{150}&space;dY(t)&space;=&space;-&space;\kappa&space;Y(t)&space;dt&space;&plus;&space;\beta&space;\sigma(t,&space;S)[1&space;&plus;&space;Y(t)]dW^{(0)}&space;&plus;&space;\epsilon&space;dW^{(1)},&space;Y(0)&space;=&space;0" title="dY(t) = - \kappa Y(t) dt + \beta \sigma(t, S)[1 + Y(t)]dW^{(0)} + \epsilon dW^{(1)}, Y(0) = 0" />
