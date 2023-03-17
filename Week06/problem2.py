from math import log, sqrt, exp
from scipy.stats import norm
import pandas as pd
from datetime import date
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import functionlib
import seaborn as sns

# copy the defined function from problem 1 to calculate option values with BLack Scholes Formula
def gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)

    if type == "call":
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)*norm.cdf(d2)
    elif type == "put":
        return strike*exp(-rf*ttm)*norm.cdf(-d2) - underlying*exp((b-rf)*ttm)*norm.cdf(-d1)
    else:
        print("Invalid type of option")

# set up the problem
current_day = date(2023,3,3)
rf = 0.0425
coupon = 0.0053
b = rf - coupon
underlying = 151.03

# read in data
data = pd.read_csv("AAPL_Options.csv")
expiration = data["Expiration"]
option_type = data["Type"]
strike = data["Strike"]
price = data["Last Price"]
implied_vol = np.zeros(len(data))

# start calculating implied volatility
for i in range(len(data)):
    # define a function so that we could find roots later
    f = lambda ivol: gbsm(underlying, int(strike[i]), (date(int(expiration[i].split("/")[2]), int(expiration[i].split("/")[0]), int(expiration[i].split("/")[1])) - current_day).days/365, rf, b, ivol, type=option_type[i].lower()) - float(price[i])
    implied_vol[i] = fsolve(f,0.5)
    
data["ivol"] = implied_vol

# extract call and put data from the combine dataframe
call = data.loc[data["Type"]=="Call"]
put = data.loc[data["Type"]=="Put"]

# plot graphs of implied volatility with respect to strike prices
plt.figure()
plt.plot(strike, implied_vol, label="All Options")
plt.plot(call.Strike, call.ivol, label="Call")
plt.plot(put.Strike, put.ivol, label="Put")
plt.xlabel("Strike Prices")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

# calculate log returns of prices of Apple Stock with defined function in my function library
prices = pd.read_csv("DailyPrices.csv")
lreturns = functionlib.return_cal(prices,method="log",datecol="Date")
aapl_lreturns = lreturns["AAPL"]

# randomly draw from the log returns of Apple stock following a normal distribution
mean = np.mean(aapl_lreturns)
std = np.std(aapl_lreturns)
normal_lreturns = norm(mean, std).rvs(10000)

# plot both distributions
plt.figure()
sns.kdeplot(aapl_lreturns, color="b", label='Actual Returns')
sns.kdeplot(normal_lreturns, color="r", label='Simulated')
plt.show()