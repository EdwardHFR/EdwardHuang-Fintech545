import pandas as pd
import functionlib
from scipy.stats import norm
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# modify codes from week6 problem 3
# define a function to simulate portfolio values with simulated prices
def sim_portfolio_value(portfolio, sim_p, ivol, rf, div_date, current_date, divAmts, divTimes, N, day_passed=0):
    sim_values = pd.DataFrame(index=portfolio.index, columns=list(range(sim_p.shape[0])))
    for i in portfolio.index:
        if portfolio["Type"][i] == "Stock":
            individual_value = sim_p
        else:
            underlying_p = sim_p
            strike = portfolio["Strike"][i]
            ttm = ((portfolio["ExpirationDate"][i] - datetime(2023,3,3)).days - day_passed) / 365
            divTimes = [int(((div_date - current_date).days - day_passed) / ((portfolio["ExpirationDate"][i] - current_date).days - day_passed) * N)]
            individual_value = np.zeros(len(underlying_p))
            for z in range(len(underlying_p)):
                individual_value[z] = functionlib.bt_american_div(underlying_p[z], strike, ttm, rf, divAmts, divTimes, ivol[i], N, type=portfolio["OptionType"][i].lower())
        
        sim_values.iloc[i,:] = portfolio["Holding"][i] * individual_value
    
    sim_values['Portfolio'] = portfolio['Portfolio']
    return sim_values.groupby('Portfolio').sum()

# read data
portfolio = pd.read_csv("problem2.csv", parse_dates=["ExpirationDate"])
underlying = 151.03
rf = 0.0425
divAmts = [1]
N = 50
div_date = datetime(2023,3,15)
current_date = datetime(2023,3,3)
ivol = np.zeros(len(portfolio.index))
for j in range(len(portfolio.index)):
    if type(portfolio["OptionType"][j]) != str:
        ivol[j] = 0
    else:
        divTimes = [int((div_date - current_date).days / (portfolio["ExpirationDate"][j] - current_date).days * N)]
        ivol[j] = functionlib.implied_vol_americandiv(underlying, portfolio["Strike"][j], (portfolio["ExpirationDate"][j] - datetime(2023,3,3)).days / 365, rf, divAmts, divTimes, N, portfolio["CurrentPrice"][j], type=portfolio["OptionType"][j].lower())
        
# read in prices data
prices = pd.read_csv("DailyPrices.csv")
returns = functionlib.return_cal(prices,method="discrete",datecol="Date")
aapl_returns = returns["AAPL"]
aapl_returns = aapl_returns - aapl_returns.mean()

# fit aapl returns into a normal distribution, and randomly generate returns from the distribution for 1000 times each day
mu, std = norm.fit(aapl_returns)
sim = pd.DataFrame(0, index=range(1000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(sim.columns)):
    for j in range(len(sim)):
        sim.iloc[j,i] =  norm.rvs(mu, std)

# calculate simulated prices for each day
norm_sim_p = pd.DataFrame(0, index=range(1000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(norm_sim_p.columns)):
    if i == 0:
        norm_sim_p.iloc[:,i] = (1 + sim.iloc[:,i]) * underlying
    else:
        norm_sim_p.iloc[:,i] = (1 + sim.iloc[:,i]) * norm_sim_p.iloc[:,i-1]
norm_sim_10p = norm_sim_p.iloc[:,-1]
                
# calculate portfolio values based on the 10th day's simulated prices from AR(1) model
norm_sim_10port = sim_portfolio_value(portfolio, norm_sim_10p, ivol, rf, div_date, current_date, divAmts, divTimes, N, day_passed=10)

# start calculating mean, VaR, ES for each portfolio
resulting_mat = pd.DataFrame(0, index=norm_sim_10port.index.values, columns=["Mean of Portfolio Value($)", "Mean of Losses/Gains($)", "VaR($)", "ES($)"])
resulting_mat.index.name = 'Portfolio'
for i in range(len(resulting_mat)):
    resulting_mat.iloc[i,0] = norm_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,1] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - norm_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,2], resulting_mat.iloc[i,3] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - functionlib.cal_ES(norm_sim_10port.iloc[i,:],alpha=0.05)
resulting_mat.insert(0,"Current Value (on 2023/3/3)", portfolio.groupby('Portfolio').sum()["CurrentPrice"])
print(resulting_mat)
print("\n\n\n")



# using delta normal calculating VaR and ES
underlying = 151.03
rf = 0.0425
divAmts = [1]
N = 50
div_date = datetime(2023,3,15)
current_date = datetime(2023,3,3)

# first calculate deltas of each American option in the portfolio
differential_equation_delta_a_call = functionlib.partial_derivative(functionlib.bt_american_div, "underlying", delta=1e-3, order=1)
deltas = np.zeros(len(portfolio.index))
for j in range(len(portfolio.index)):
    if type(portfolio["OptionType"][j]) != str:
        deltas[j] = 1
    else:
        divTimes = [int(((div_date - current_date).days - 10) / ((portfolio["ExpirationDate"][j] - current_date).days - 10) * N)]
        deltas[j] = differential_equation_delta_a_call(underlying, portfolio["Strike"][j], ((portfolio["ExpirationDate"][j] - datetime(2023,3,3)).days - 10) / 365, rf, divAmts, divTimes, ivol[j], N, type=portfolio["OptionType"][j].lower())
portfolio["delta"] = deltas

# start calculating VaR and ES of each portfolio
std = aapl_returns.std()
t = 10
resulting_mat_del = pd.DataFrame(0, index=sorted(portfolio['Portfolio'].unique()), columns=["Mean($)", "VaR($)", "ES($)"])
resulting_mat_del.index.name = 'Portfolio'
for pfl, df in portfolio.groupby('Portfolio'):
    gradient = underlying / df['CurrentPrice'].sum() * (df['Holding'] * df['delta']).sum()
    pfl_10d_std = abs(gradient) * std * np.sqrt(t)
    N = norm(0, 1)
    present_value = df['CurrentPrice'].sum() 
    resulting_mat_del.loc[pfl,'Mean($)'] = 0
    resulting_mat_del.loc[pfl,'VaR($)'] = -present_value * N.ppf(0.05) * pfl_10d_std
    resulting_mat_del.loc[pfl,'ES($)'] = present_value * pfl_10d_std * N.pdf(N.ppf(0.05)) / 0.05
print(resulting_mat_del)



# copy last week results here
resulting_mat_lastweek = pd.DataFrame(0, index=sorted(portfolio['Portfolio'].unique()), columns=["VaR($)", "ES($)"])
resulting_mat_lastweek.index.name = 'Portfolio'
VaRs = np.array([6.016988, 8.292276, 20.135761, 18.353101, 4.795894, -4.022981, 15.843101, 2.966320, 25.220350])
ESs = np.array([6.360480, 8.599723, 23.879030, 22.718202, 4.829382, -3.904941, 19.708202, 2.977172, 29.341559])
resulting_mat_lastweek.iloc[:,0] = VaRs
resulting_mat_lastweek.iloc[:,1] = ESs

# Combine results from different categories into a single DataFrame
results = []
for category, result in zip(['Normal Monte Carlo', 'Delta Normal', 'Week 6'], [resulting_mat.loc[:,"VaR($)":"ES($)"], resulting_mat_del.loc[:,"VaR($)":"ES($)"], resulting_mat_lastweek]):
    new_result = result.reset_index()
    new_result['Type'] = category
    results.append(new_result)
results = pd.concat(results, axis=0)

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Create a barplot for VaR in the first subplot with green color palette
sns.barplot(x='Portfolio', y='VaR($)', hue='Type', palette='Greens', data=results, ax=axes[0])
axes[0].set_title('VaR Comparison')

# Create a barplot for ES in the second subplot with green color palette
sns.barplot(x='Portfolio', y='ES($)', hue='Type', palette='Greens', data=results, ax=axes[1])
axes[1].set_title('ES Comparison')

# Show the plot
plt.show()