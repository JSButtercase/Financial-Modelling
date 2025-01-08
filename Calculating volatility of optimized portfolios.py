import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.integrate import quad

# 2a

Given_Returns = [0.05, 0.07, 0.15, 0.27]
Given_Vols = [0.07, 0.12, 0.30, 0.60]

Correlation = np.matrix([[1, 0.8, 0.5, 0.4], [0.8, 1, 0.7, 0.5], [0.5, 0.7, 1, 0.8], [0.4, 0.5, 0.8, 1]])


def calc_returns(weights):
    y = []
    for i in range(len(Given_Returns)):
        y.append(Given_Returns[i] * weights[i])
    return np.sum(y)


def calc_volatility(weights):
    weights = weights.reshape(4, 1)
    b = np.dot(weights.T, np.dot(Correlation, weights))
    return np.sqrt(b)


number_of_portfolios = 1000
portfolio_returns = []
portfolio_volatilities = []
np.random.seed(32)

for each_portfolio in range(number_of_portfolios):
    random_weight = np.random.random(len(Given_Returns))
    random_weight /= np.sum(random_weight)
    portfolio_returns.append(calc_returns(random_weight))
    portfolio_volatilities.append(calc_volatility(random_weight))

returns = np.array(portfolio_returns)
vols = np.array(portfolio_volatilities)

#2b
"""
plt.scatter(vols, returns)
plt.xlabel('volatility simulated')
plt.ylabel('returns simulated')
"""

# 2c
risk_free_rate = 0.02
"""Sharpe Ratio = (Portfolio Expected Return - risk free rate) / (Portfolio Volatility)"""
# set up negative sharpe ratio


def portfolio_sharpe(weights):
    return -calc_returns(weights) / calc_volatility(weights)


constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
boundaries = tuple((0, 1) for x in range(len(Given_Returns)))
equal_weights = np.array(len(Given_Returns) * [1 / len(Given_Returns)])

optimized_portfolio = sco.minimize(portfolio_sharpe, equal_weights, method='SLSQP',
                                   bounds=boundaries, constraints=constraints)

opti_rets = calc_returns(optimized_portfolio['x']).round(2)
opti_vol = calc_volatility(optimized_portfolio['x']).round(2)
print(opti_rets, opti_vol)

