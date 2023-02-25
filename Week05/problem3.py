import pandas as pd
from scipy.stats import t, norm, spearmanr
import functionlib
import numpy as np
import math

# read input data, extract data by different portfolios
prices = pd.read_csv("DailyPrices.csv")
current_prices = prices.iloc[-1,:]
portfolio = pd.read_csv("portfolio.csv")
portfolio_a = portfolio.loc[portfolio["Portfolio"] == "A"]
portfolio_b = portfolio.loc[portfolio["Portfolio"] == "B"]
portfolio_c = portfolio.loc[portfolio["Portfolio"] == "C"]
current_prices_tot = current_prices[portfolio.Stock]
current_prices_a = current_prices[portfolio_a.Stock]
current_prices_b = current_prices[portfolio_b.Stock]
current_prices_c = current_prices[portfolio_c.Stock]

# calculate total value of each portfolio
# total portfolio
PV_tot = 0
for i in range(len(portfolio)):
    value = portfolio.iloc[i,-1] * current_prices_tot[i]
    PV_tot += value
    
# portfolio a
PV_a = 0
for i in range(len(portfolio_a)):
    value = portfolio_a.iloc[i,-1] * current_prices_a[i]
    PV_a += value
    
# portfolio b
PV_b = 0
for i in range(len(portfolio_b)):
    value = portfolio_b.iloc[i,-1] * current_prices_b[i]
    PV_b += value
    
# portfolio c
PV_c = 0
for i in range(len(portfolio_c)):
    value = portfolio_c.iloc[i,-1] * current_prices_c[i]
    PV_c += value

# calculate arithmetic return, and extract returns by different portfolios
returns = functionlib.return_cal(prices,method="discrete",datecol="Date")
del returns["Date"]
# demean all returns
returns -= returns.mean()
port_tot_re = returns[portfolio.Stock]
port_a_re = returns[portfolio_a.Stock]
port_b_re = returns[portfolio_b.Stock]
port_c_re = returns[portfolio_c.Stock]

# define a function to fit returns to t distribution
def returns_fit(data):
    out = {"example": ["df", "loc", "scale", "dist"]}
    for i in data.columns:
        temp_results = t.fit(data.loc[:,i])
        temp_dist = t(temp_results[0], temp_results[1], temp_results[2])
        temp = [temp_results[0], temp_results[1], temp_results[2], temp_dist]
        out[i] = temp
    return out

# start calling defined function to fit returns to t distribution for each portfolio
port_tot_fit = returns_fit(port_tot_re)
port_a_fit = returns_fit(port_a_re)
port_b_fit = returns_fit(port_b_re)
port_c_fit = returns_fit(port_c_re)

# define a function to calculate U matrix
def generate_U(data, fits):
    assert len(data.columns) == len(fits) - 1
    temp = []
    df = pd.DataFrame()
    for i in data.columns:
        temp_cdf = fits[i][-1].cdf(data.loc[:,i])
        df[i] = temp_cdf
    return df

# call defined function to derive U matrices for each portfolio
port_tot_U = generate_U(port_tot_re, port_tot_fit)
port_a_U = generate_U(port_a_re, port_a_fit)
port_b_U = generate_U(port_b_re, port_b_fit)
port_c_U = generate_U(port_c_re, port_c_fit)

# calculate spearman correlation matrix for each portfolio
port_tot_spear = pd.DataFrame(spearmanr(port_tot_U)[0], columns=port_tot_re.columns, index=port_tot_re.columns)
port_a_spear = pd.DataFrame(spearmanr(port_a_U)[0], columns=port_a_re.columns, index=port_a_re.columns)
port_b_spear = pd.DataFrame(spearmanr(port_b_U)[0], columns=port_b_re.columns, index=port_b_re.columns)
port_c_spear = pd.DataFrame(spearmanr(port_c_U)[0], columns=port_c_re.columns, index=port_c_re.columns)

# simulate using the copula
port_tot_sim = np.random.multivariate_normal(np.zeros(len(port_tot_re.columns)), port_tot_spear, (1,len(port_tot_re.columns),500))[0][0]
port_a_sim = np.random.multivariate_normal(np.zeros(len(port_a_re.columns)), port_a_spear, (1,len(port_a_re.columns),500))[0][0]
port_b_sim = np.random.multivariate_normal(np.zeros(len(port_b_re.columns)), port_b_spear, (1,len(port_b_re.columns),500))[0][0]
port_c_sim = np.random.multivariate_normal(np.zeros(len(port_c_re.columns)), port_c_spear, (1,len(port_c_re.columns),500))[0][0]

# convert to U_sim
port_tot_sim_U = pd.DataFrame(norm.cdf(port_tot_sim), columns=port_tot_re.columns)
port_a_sim_U = pd.DataFrame(norm.cdf(port_a_sim), columns=port_a_re.columns)
port_b_sim_U = pd.DataFrame(norm.cdf(port_b_sim), columns=port_b_re.columns)
port_c_sim_U = pd.DataFrame(norm.cdf(port_c_sim), columns=port_c_re.columns)

# define a function to convert to sim values
def convert_sim_values(fit, sim_U):
    out = pd.DataFrame()
    for i in sim_U.columns:
        out[i] = fit[i][-1].ppf(sim_U.loc[:,i])
    return out

# call defined function to convert U_sim to sim values for each portfolio
port_tot_simout = convert_sim_values(port_tot_fit, port_tot_sim_U)
port_a_simout = convert_sim_values(port_a_fit, port_a_sim_U)
port_b_simout = convert_sim_values(port_b_fit, port_b_sim_U)
port_c_simout = convert_sim_values(port_c_fit, port_c_sim_U)

# update current prices
current_prices_tot = current_prices_tot * (1 + port_tot_simout)
current_prices_a = current_prices_a * (1 + port_a_simout)
current_prices_b = current_prices_b * (1 + port_b_simout)
current_prices_c = current_prices_c * (1 + port_c_simout)

# calculate each stock's value in each portfolio
# total portfolio
port_tot_newval = np.multiply(current_prices_tot, portfolio.loc[:,"Holding"])
port_tot_newval = np.transpose(port_tot_newval)
port_tot_val = np.zeros(len(port_tot_newval.columns))
for i in range(len(port_tot_newval.columns)):
    port_tot_val[i] = port_tot_newval.iloc[:,i].sum()

# portfolio a
port_a_newval = np.multiply(current_prices_a, portfolio_a.loc[:,"Holding"])
port_a_newval = np.transpose(port_a_newval)
port_a_val = np.zeros(len(port_a_newval.columns))
for i in range(len(port_a_newval.columns)):
    port_a_val[i] = port_a_newval.iloc[:,i].sum()

# portfolio b
port_b_newval = np.multiply(current_prices_b, portfolio_b.loc[:,"Holding"])
port_b_newval = np.transpose(port_b_newval)
port_b_val = np.zeros(len(port_b_newval.columns))
for i in range(len(port_b_newval.columns)):
    port_b_val[i] = port_b_newval.iloc[:,i].sum()

# portfolio c
port_c_newval = np.multiply(current_prices_c, portfolio_c.loc[:,"Holding"])
port_c_newval = np.transpose(port_c_newval)
port_c_val = np.zeros(len(port_c_newval.columns))
for i in range(len(port_c_newval.columns)):
    port_c_val[i] = port_c_newval.iloc[:,i].sum()

# copy codes from functionlib and modify for use
def cal_ES(x, PV, alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    ES = xs[0:idn].mean()
    return PV - VaR, PV - ES    

# call defined fucntion to calculate ES
VaR_05_tot, ES_05_tot = cal_ES(port_tot_val, PV_tot)
VaR_05_a, ES_05_a = cal_ES(port_a_val, PV_a)
VaR_05_b, ES_05_b = cal_ES(port_b_val, PV_b)
VaR_05_c, ES_05_c = cal_ES(port_c_val, PV_c)
print("VaR: Total Portfolio: ${:.4f}, Portfolio A: ${:.4f}, Portfolio B: ${:.4f}, Portfolio C: ${:.4f}".format(VaR_05_tot, VaR_05_a, VaR_05_b, VaR_05_c))
print("ES: Total Portfolio: ${:.4f}, Portfolio A: ${:.4f}, Portfolio B: ${:.4f}, Portfolio C: ${:.4f}".format(ES_05_tot, ES_05_a, ES_05_b, ES_05_c))
