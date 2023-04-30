# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import numpy as np
import pandas as pd

from scipy.stats import norm
import time

import matplotlib.pyplot as plt

# mu = 0.01, std = 0.10, 1000 bars, 10 assets
mu = 0.01
sigma = 0.10
bars = 1000
num_assets = 10

returns = np.random.normal(mu, sigma, (bars, num_assets))

# Fake asset names
names = ['Asset %s' % i for i in range(num_assets)]

# Put in a pandas dataframe
returns = pd.DataFrame(returns, columns=names)
print(returns.columns[0])
returns_new = returns.sort_values(returns.columns[0])
print(returns_new.head(10))
print(np.percentile(returns.iloc[:, 0], 1))  # 排序后，从小到大，取最小的分位数%

# Plot the last 50 bars
plt.plot(returns.head(50))
plt.xlabel('Time')
plt.ylabel('Return')
plt.show()

weights = np.ones((10, 1))
# Normalize
weights = weights / np.sum(weights)


# Non-Parametric Historical VaR
# 根据交易历史得到VaR，无需已知分布
def value_at_risk(p_value_invested, p_returns, p_weights, alpha=0.95, lookback_days=520):
    p_returns = p_returns.fillna(0.0)
    # Multiply asset returns by weights to get one weighted portfolio return
    portfolio_returns = p_returns.iloc[-lookback_days:].dot(p_weights)
    # Compute the correct percentile loss and multiply by value invested
    return np.percentile(portfolio_returns, 100 * (1 - alpha)) * p_value_invested


value_invested = 1000000
# We'll compute the VaR for alpha = 0.95
VaR = value_at_risk(value_invested, returns, weights, alpha=0.95)
print(VaR)

# Portfolio mean return is unchanged, but std has to be recomputed
# This is because independent variances sum, but std is sqrt of variance
portfolio_std = np.sqrt(np.power(sigma, 2) * num_assets) / num_assets


# Normal Distribution VaR manually
Normal_VaR = (mu - portfolio_std * norm.ppf(0.95)) * value_invested
print(Normal_VaR)


# Normal Distribution VaR func
def value_at_risk_N(p_mu=0, p_sigma=1.0, alpha=0.95):
    return p_mu - p_sigma * norm.ppf(alpha)


x = np.linspace(-3 * sigma, 3 * sigma, 1000)
y = norm.pdf(x, loc=mu, scale=portfolio_std)
plt.plot(x, y)
plt.axvline(value_at_risk_N(p_mu=0.01, p_sigma=portfolio_std, alpha=0.95), color='red', linestyle='solid')
plt.legend(['Return Distribution', 'VaR for Specified Alpha as a Return'])
plt.title('VaR in Closed Form for a Normal Distribution')
plt.show()


def cvar(p_value_invested, p_returns, p_weights, alpha=0.95, lookback_days=520):
    # Call out to our existing function
    var = value_at_risk(p_value_invested, p_returns, p_weights, alpha, lookback_days=lookback_days)
    p_returns = p_returns.fillna(0.0)
    portfolio_returns = p_returns.iloc[-lookback_days:].dot(p_weights)

    # Get back to a return rather than an absolute loss
    var_pct_loss = var / p_value_invested

    return p_value_invested * np.nanmean(portfolio_returns[portfolio_returns < var_pct_loss])
