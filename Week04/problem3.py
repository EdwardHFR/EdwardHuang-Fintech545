import pandas as pd
import math
import numpy as np
from scipy import stats

# copy codes for calculation of returns from problem 2
def return_cal(data,method="discrete",datecol="Date"):
    # check if there is a date column in the data file
    vars = data.columns
    assert datecol in vars
    # number of variables(in this case stocks/assets)
    nvars = len(vars) - 1
    vars = list(vars)
    vars.pop(vars.index(datecol))
    
    # construct a new dataframe for prices only
    prices = data.iloc[:,data.columns.get_loc(datecol)+1:]
    
    # construct a new pandas dataframe used to record returns
    returns = np.zeros(shape=(len(prices)-1,nvars))
    returns = pd.DataFrame(returns)
    
    for i in range(len(prices)-1):
        for j in range(nvars):
            returns.iloc[i,j] = prices.iloc[i+1,j] / prices.iloc[i,j]
            if method=="discrete":
                returns.iloc[i,j] -= 1
            elif method=="log":
                returns.iloc[i,j] = math.log(returns.iloc[i,j])
            else:
                return sys.exit(1)
    
    dates = data.iloc[:, data.columns.get_loc(datecol)]
    dates = dates.drop(index=0)
    # update the pandas dataframe's index from starting from 1 to starting from 0
    dates.index -= 1
    out = pd.DataFrame({datecol: dates})
    # combine dates and returns together
    for i in range(nvars):
        out[vars[i]] = returns.iloc[:, i]
    return out

# copy codes from problem 2 for calculation of weights and variance-covariance matrix
def populateWeights(x,w,cw, λ):
    n = len(x)
    tw = 0
    # start a for loop to calculate the weight for each stock, and recording total weights and cumulative weights for each stock
    for i in range(n):
        individual_w = (1-λ)*pow(λ,i)
        w.append(individual_w)
        tw += individual_w
        cw.append(tw)
    
    # start another for loop to calculate normalized weights and normalized cumulative weights for each stock
    for i in range(n):
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw

def exwCovMat(data, weights_vector):
    # get the stock names listed in the file, and delete the first item, since it is the column of dates
    stock_names = list(data.columns)
    
    # set up an empty matrix, and transform it into a pandas Dataframe
    mat = np.empty((len(stock_names),len(stock_names)))
    w_cov_mat = pd.DataFrame(mat, columns = stock_names, index = stock_names)
    
    # calculate variances and covariances
    for i in stock_names:
        for j in stock_names:
            # get data of stock i and data of stock j respectively
            i_data = data.loc[:,i]
            j_data = data.loc[:,j]
            
            # calculate means of data of stock i and data of stock j
            i_mean = i_data.mean()
            j_mean = j_data.mean()
            
            # make sure i_data, j_data, and weights_vector all have the same number of items
            assert len(i_data) == len(j_data) == len(weights_vector)
            
            # set up sum for calculation of variance and covariance, and a for loop for that
            s = 0
            
            for z in range(len(data)):
                
                part = weights_vector[z] * (i_data[z] - i_mean) * (j_data[z] - j_mean)
                s += part
            
            # store the derived variance into the matrix
            w_cov_mat.loc[i,j] = s
    
    return w_cov_mat



### Arithmetic Returns with Delta Normal VaR :: START
# read prices from the csv
prices = pd.read_csv("DailyPrices.csv")
current_prices = prices.iloc[-1,:]
portfolios = pd.read_csv("portfolio.csv")
portfolio_a = portfolios.loc[portfolios["Portfolio"] == "A"]
portfolio_b = portfolios.loc[portfolios["Portfolio"] == "B"]
portfolio_c = portfolios.loc[portfolios["Portfolio"] == "C"]
current_prices_tot = current_prices[portfolios.Stock]
current_prices_a = current_prices[portfolio_a.Stock]
current_prices_b = current_prices[portfolio_b.Stock]
current_prices_c = current_prices[portfolio_c.Stock]
# call defined function above to calculate returns with arithmetic system first
ari_returns = return_cal(prices)

# calculate total holdings of each stock for the combination of three portfolios
total_holdings = portfolios.groupby('Stock').sum('Holding')

# derive returns for stocks included in portfolio a, b, c, and also for the combination of three portfolios
port_tot_re = ari_returns[portfolios.Stock]
port_a_re = ari_returns[portfolio_a.Stock]
port_b_re = ari_returns[portfolio_b.Stock]
port_c_re = ari_returns[portfolio_c.Stock]

# determine whether returns of each portfolio are normally distributed or not - to determine what methods to use for VaR later
# I use shapiro-wilk test to test normality - derive a p-value for each stock's returns, and derive an average of all p-values in a portfolio, use that average p-value to determine whether the whole portfolio follows a normal distribution or not
stats_tot = port_tot_re.apply(stats.shapiro)
p_tot = stats_tot.iloc[1,:]
avg_p_tot = p_tot.sum() / len(p_tot)
print("The average shapiro wilk test p-value of the combination of three portfolios is {:.4f}".format(avg_p_tot))

stats_a = port_a_re.apply(stats.shapiro)
p_a = stats_a.iloc[1,:]
avg_p_a = p_a.sum() / len(p_a)
print("The average shapiro wilk test p-value of portfolio a is {:.4f}".format(avg_p_a))

stats_b = port_b_re.apply(stats.shapiro)
p_b = stats_b.iloc[1,:]
avg_p_b = p_b.sum() / len(p_b)
print("The average shapiro wilk test p-value of portfolio b is {:.4f}".format(avg_p_b))

stats_c = port_c_re.apply(stats.shapiro)
p_c = stats_c.iloc[1,:]
avg_p_c = p_c.sum() / len(p_c)
print("The average shapiro wilk test p-value of portfolio c is {:.4f}".format(avg_p_c))

# now, after deriving those p-values, since they are all greater than 0.05, we assume that these returns in these portfolios are following a normal distribution. Since returns are derived using arithmetic system, then we could use Delta Normal VaR
# compute portfolio values
PV_tot = 0
PV_a = 0
PV_b = 0
PV_c = 0
delta_tot = np.zeros(len(portfolios))
delta_a = np.zeros(len(portfolio_a))
delta_b = np.zeros(len(portfolio_b))
delta_c = np.zeros(len(portfolio_c))

# compute value of the combination of three portfolios
for i in range(len(portfolios)):
    value = portfolios.iloc[i,-1] * current_prices_tot[i]
    PV_tot += value
    delta_tot[i] = value
delta_tot = delta_tot / PV_tot
print("Portfolio value of the combination of three portfolios is ${:.4f}".format(PV_tot))
print("Delta vector of this combined portfolio is")
print(delta_tot)

# compute value of portfolio a
for i in range(len(portfolio_a)):
    value = portfolio_a.iloc[i,-1] * current_prices_a[i]
    PV_a += value
    delta_a[i] = value
delta_a = delta_a / PV_a
print("Portfolio value of portfolio A is ${:.4f}".format(PV_a))
print("Delta vector of portfolio A is")
print(delta_a)

# compute value of portfolio b
for i in range(len(portfolio_b)):
    value = portfolio_b.iloc[i,-1] * current_prices_b[i]
    PV_b += value
    delta_b[i] = value
delta_b = delta_b / PV_b
print("Portfolio value of portfolio B is ${:.4f}".format(PV_b))
print("Delta vector of portfolio B is")
print(delta_b)

# compute value of portfolio c
for i in range(len(portfolio_c)):
    value = portfolio_c.iloc[i,-1] * current_prices_c[i]
    PV_c += value
    delta_c[i] = value
delta_c = delta_c / PV_c
print("Portfolio value of portfolio C is ${:.4f}".format(PV_c))
print("Delta vector of portfolio C is")
print(delta_c)

# calculation of Delta Normal VaR of the combination of three portfolios
w = []
cw = []
populateWeights(port_tot_re,w,cw, 0.94)
w = w[::-1]
tot_cov = exwCovMat(port_tot_re, w)
tot_fac = math.sqrt(np.transpose(delta_tot) @ tot_cov @ delta_tot)
VaR_05_tot = -PV_tot * stats.norm.ppf(0.05, loc=0, scale=1) * tot_fac
print("5% VaR of the combination of three portfolios is ${:.4f}".format(VaR_05_tot))
print("5% VaR of the combination of three portfolios could also be expressed as {:.4f}%".format(VaR_05_tot * 100 / PV_tot))

# calculation of Delta Normal VaR of portfolio a
w = []
cw = []
populateWeights(port_a_re,w,cw, 0.94)
w = w[::-1]
a_cov = exwCovMat(port_a_re, w)
a_fac = math.sqrt(np.transpose(delta_a) @ a_cov @ delta_a)
VaR_05_a = -PV_a * stats.norm.ppf(0.05, loc=0, scale=1) * a_fac
print("5% VaR of portfolio a is ${:.4f}".format(VaR_05_a))
print("5% VaR of portfolio a could also be expressed as {:.4f}%".format(VaR_05_a * 100 / PV_a))

# calculation of Delta Normal VaR of portfolio b
w = []
cw = []
populateWeights(port_b_re,w,cw, 0.94)
w = w[::-1]
b_cov = exwCovMat(port_b_re, w)
b_fac = math.sqrt(np.transpose(delta_b) @ b_cov @ delta_b)
VaR_05_b = -PV_b * stats.norm.ppf(0.05, loc=0, scale=1) * b_fac
print("5% VaR of portfolio b is ${:.4f}".format(VaR_05_b))
print("5% VaR of portfolio b could also be expressed as {:.4f}%".format(VaR_05_b * 100 / PV_b))


# calculation of Delta Normal VaR of portfolio c
w = []
cw = []
populateWeights(port_c_re,w,cw, 0.94)
w = w[::-1]
c_cov = exwCovMat(port_c_re, w)
c_fac = math.sqrt(np.transpose(delta_c) @ c_cov @ delta_c)
VaR_05_c = -PV_c * stats.norm.ppf(0.05, loc=0, scale=1) * c_fac
print("5% VaR of portfolio c is ${:.4f}".format(VaR_05_c))
print("5% VaR of portfolio c could also be expressed as {:.4f}%".format(VaR_05_c * 100 / PV_c))
### Arithmetic Returns with Delta Normal VaR :: END



### Log Returns with Historic VaR :: START
# calculate returns with log method
log_returns = return_cal(prices,method="log")
port_tot_logre = log_returns[portfolios.Stock]
port_a_logre = log_returns[portfolio_a.Stock]
port_b_logre = log_returns[portfolio_b.Stock]
port_c_logre = log_returns[portfolio_c.Stock]

# calculate VaR with Historic VaR of combined portfolio
sim_prices_tot = (np.exp(port_tot_logre)) * np.transpose(current_prices_tot)
port_values_tot = np.dot(sim_prices_tot, portfolios.Holding)
port_values_tot_sorted = np.sort(port_values_tot)
index_tot = np.floor(0.05*len(port_tot_logre))
VaR_05_tot_logH = PV_tot - port_values_tot_sorted[int(index_tot-1)]
print("5% VaR of combined portfolio using Historic VaR is ${:.4f}".format(VaR_05_tot_logH))

# calculate VaR with Historic VaR of portfolio A
sim_prices_a = (np.exp(port_a_logre)) * np.transpose(current_prices_a)
port_values_a = np.dot(sim_prices_a, portfolio_a.Holding)
port_values_a_sorted = np.sort(port_values_a)
index_a = np.floor(0.05*len(port_a_logre))
VaR_05_a_logH = PV_a - port_values_a_sorted[int(index_a-1)]
print("5% VaR of portfolio A using Historic VaR is ${:.4f}".format(VaR_05_a_logH))

# calculate VaR with Historic VaR of portfolio B
sim_prices_b = (np.exp(port_b_logre)) * np.transpose(current_prices_b)
port_values_b = np.dot(sim_prices_b, portfolio_b.Holding)
port_values_b_sorted = np.sort(port_values_b)
index_b = np.floor(0.05*len(port_b_logre))
VaR_05_b_logH = PV_b - port_values_b_sorted[int(index_b-1)]
print("5% VaR of portfolio B using Historic VaR is ${:.4f}".format(VaR_05_b_logH))

# calculate VaR with Historic VaR of portfolio C
sim_prices_c = (np.exp(port_c_logre)) * np.transpose(current_prices_c)
port_values_c = np.dot(sim_prices_c, portfolio_c.Holding)
port_values_c_sorted = np.sort(port_values_c)
index_c = np.floor(0.05*len(port_c_logre))
VaR_05_c_logH = PV_c - port_values_c_sorted[int(index_c-1)]
print("5% VaR of portfolio C using Historic VaR is ${:.4f}".format(VaR_05_c_logH))
### Log Returns with Historic VaR :: END