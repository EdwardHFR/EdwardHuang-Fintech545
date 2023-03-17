import functionlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

### PART 1 ###
# define a function to simulate portfolio values with simulated prices
def sim_portfolio_value(portfolio, sim_p, ivol, rf, b, day_passed=0):
    sim_values = pd.DataFrame(index=portfolio.index, columns=list(range(sim_p.shape[0])))
    for i in portfolio.index:
        if portfolio["Type"][i] == "Stock":
            individual_value = sim_p
        else:
            underlying_p = sim_p
            strike = portfolio["Strike"][i]
            ttm = ((portfolio["ExpirationDate"][i] - datetime(2023,3,3)).days - day_passed) / 365
            individual_value = np.zeros(len(underlying_p))
            for z in range(len(underlying_p)):
                individual_value[z] = functionlib.gbsm(underlying_p[z], strike, ttm, rf, b, ivol[i], type=portfolio["OptionType"][i].lower())
        
        sim_values.iloc[i,:] = portfolio["Holding"][i] * individual_value
    
    sim_values['Portfolio'] = portfolio['Portfolio']
    return sim_values.groupby('Portfolio').sum()

# read data
portfolio = pd.read_csv("problem3.csv", parse_dates=["ExpirationDate"])
underlying = 151.03
rf = 0.0425
b = 0.0425 - 0.0053
ivol = np.zeros(len(portfolio.index))
for j in range(len(portfolio.index)):
    if type(portfolio["OptionType"][j]) != str:
        ivol[j] = 0
    else:
        ivol[j] = functionlib.implied_vol(underlying, portfolio["Strike"][j], (portfolio["ExpirationDate"][j] - datetime(2023,3,3)).days / 365, rf, b, portfolio["CurrentPrice"][j], type=portfolio["OptionType"][j].lower())

# apply the defined function
sim_p = np.linspace(100, 200, 50)
simulated_vals = sim_portfolio_value(portfolio, sim_p, ivol, rf, b)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
idx = 0
for portfolios, dataframes in simulated_vals.groupby('Portfolio'):
    i, j = idx // 3, idx % 3
    ax = axes[i][j]
    ax.plot(sim_p, dataframes.iloc[0, :].values)
    ax.set_title(portfolios)
    ax.set_xlabel('Underlying Price', fontsize=8)
    ax.set_ylabel('Portfolio Value', fontsize=8)
    idx += 1

### PART 2 ###
# read in prices data
prices = pd.read_csv("DailyPrices.csv")
lreturns = functionlib.return_cal(prices,method="log",datecol="Date")
aapl_lreturns = lreturns["AAPL"]
aapl_lreturns = aapl_lreturns - aapl_lreturns.mean()

# start fitting returns with AR(1) model
mod = sm.tsa.ARIMA(aapl_lreturns, order=(1, 0, 0))
results = mod.fit()
summary = results.summary()
m = float(summary.tables[1].data[1][1])
a1 = float(summary.tables[1].data[2][1])
s = math.sqrt(float(summary.tables[1].data[3][1]))
sim = pd.DataFrame(0, index=range(10000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(sim.columns)):
    for j in range(len(sim)):
        if i == 0:
            sim.iloc[j,i] =  a1 * (aapl_lreturns.iloc[-1]) + s * np.random.normal() + m
        else:
            sim.iloc[j,i] =  a1 * (sim.iloc[j,i-1]) + s * np.random.normal() + m

# calculate prices on the 10th day from current date
ar1_sim_p = pd.DataFrame(0, index=range(10000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(ar1_sim_p.columns)):
    if i == 0:
        ar1_sim_p.iloc[:,i] = np.exp(sim.iloc[:,i]) * underlying
    else:
        ar1_sim_p.iloc[:,i] = np.exp(sim.iloc[:,i]) * ar1_sim_p.iloc[:,i-1]
ar1_sim_10p = ar1_sim_p.iloc[:,-1]

# calculate portfolio values based on the 10th day's simulated prices from AR(1) model
ar1_sim_10port = sim_portfolio_value(portfolio, ar1_sim_10p, ivol, rf, b, day_passed=10)

# start calculating mean, VaR, ES for each portfolio
resulting_mat = pd.DataFrame(0, index=ar1_sim_10port.index.values, columns=["Mean of Portfolio Value($)", "Mean of Losses/Gains($)", "VaR($)", "ES($)", "VaR(%)", "ES(%)"])
for i in range(len(resulting_mat)):
    resulting_mat.iloc[i,0] = ar1_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,1] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - ar1_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,2], resulting_mat.iloc[i,3] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - functionlib.cal_ES(ar1_sim_10port.iloc[i,:],alpha=0.05)
    resulting_mat.iloc[i,4] = resulting_mat.iloc[i,2] * 100 / portfolio.groupby('Portfolio').sum().iloc[i,-1]
    resulting_mat.iloc[i,5] = resulting_mat.iloc[i,3] * 100 / portfolio.groupby('Portfolio').sum().iloc[i,-1]
resulting_mat["Current Value (on 2023/3/3)"] = portfolio.groupby('Portfolio').sum()["CurrentPrice"]