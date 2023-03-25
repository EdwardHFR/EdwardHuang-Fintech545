import pandas as pd
import functionlib
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# read those three csv files
prices = pd.read_csv("DailyPrices.csv")
date_index_prices = pd.read_csv("DailyPrices.csv", parse_dates=['Date']).set_index('Date')
all_other_facs = pd.read_csv("F-F_Research_Data_Factors_daily.CSV", parse_dates=['Date']).set_index('Date')
mom_fac = pd.read_csv("F-F_Momentum_Factor_daily.CSV", parse_dates=['Date']).set_index('Date')

# join factors data and add a column of constants to the data
all_facs = all_other_facs.join(mom_fac, how='right')

# define a list of stocks
stocks = ['AAPL', 'META', 'UNH', 'MA', 'MSFT' ,'NVDA', 'HD', 'PFE', 'AMZN' ,'BRK-B', 'PG', 'XOM', 'TSLA' ,'JPM' ,'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO']
factors = ['Mkt-RF', 'SMB', 'HML', 'Mom   ']

# calculate the stock returns
returns = functionlib.return_cal(prices,method="discrete",datecol="Date")
returns = returns * 100

# filter all_facs and needed_returns to have the same time length as returns
all_facs_partial = all_facs.loc[date_index_prices.index[1]:]
last_date = date_index_prices.index.get_loc(all_facs_partial.index[-1])
returns_partial = returns.iloc[:last_date]
needed_returns = returns_partial[stocks]
Xs = all_facs_partial[factors]
Rfs = all_facs_partial["RF"]

# subtract RFs from returns of stocks first, used for regression later
Ys = pd.DataFrame().reindex_like(needed_returns)
for i in range(len(needed_returns.columns)):
    for j in range(len(Ys)):
        Ys.iloc[j,i] = needed_returns.iloc[j,i] - Rfs[j]

# start running the fitting process and calculate expected return of each stock
avg_fac_returns = all_facs.loc['2013-1-31':'2023-1-31'].mean()
avg_stock_returns_dic = dict.fromkeys(stocks)
for stock in stocks:
    reg = LinearRegression().fit(Xs, Ys[stock])
    avg_stock_returns_dic[stock] = np.dot(avg_fac_returns[factors], reg.coef_) + reg.intercept_ + avg_fac_returns["RF"]

# annualize expected daily returns of each stock
avg_stock_returns = np.fromiter(avg_stock_returns_dic.values(), dtype=float) * len(returns)

# compute the annual covariance matrix for those 20 stocks using equal weights on dates, and annualize those covariances
day_cov_mat_20_stocks = functionlib.exwCovMat(returns[stocks], np.ones(len(returns))/len(returns))
year_cov_mat_20_stocks = day_cov_mat_20_stocks * len(returns)

# define a function to find optimized portfolio (minimize risk with a target return)
def optimize_Sharpe(covar, expected_r, rf):
    # Define objective function
    def negative_Sharpe(w):
        returns = np.dot(expected_r, w)
        std = np.sqrt(w @ covar @ w.T)
        return -(returns - rf) / std

    # Define constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    # Define bounds
    bounds = [(0, None)] * len(expected_r)

    # Define initial guess
    x0 = np.full(len(expected_r), 1/len(expected_r))

    # Use minimize function to solve optimization problem
    result = minimize(negative_Sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    # Return the objective value (risk) and the portfolio weights
    return {"max_Sharpe_Ratio": -result.fun, "weights": result.x}

# apply the defined function above to find some portfolios
optimized_results = optimize_Sharpe(year_cov_mat_20_stocks, avg_stock_returns, 0.0425 * 100)
max_Sharpe = optimized_results["max_Sharpe_Ratio"]
weights = np.array(optimized_results["weights"])
percentage_weights = weights * 100
weights_mat = pd.DataFrame(percentage_weights, index=stocks, columns=["weights(%)"])
print(max_Sharpe)
print(weights_mat)