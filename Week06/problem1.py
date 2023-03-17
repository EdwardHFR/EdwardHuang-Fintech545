from datetime import date
from math import log, sqrt, exp, isclose
from scipy.stats import lognorm, norm
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# calculate the time to maturity
current_day = date(2023,3,3)
expire_day = date(2023,3,17)
days_to_mat = (expire_day - current_day).days
ttm = days_to_mat / 365
print("Time to Maturity is {:.4f}".format(ttm))

# define a function to calculate option values with BLack Scholes Formula
def gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)

    if type == "call":
        return underlying * exp((b-rf)*ttm) * norm.cdf(d1) - strike*exp(-rf*ttm)*norm.cdf(d2)
    elif type == "put":
        return strike*exp(-rf*ttm)*norm.cdf(-d2) - underlying*exp((b-rf)*ttm)*norm.cdf(-d1)
    else:
        print("Invalid type of option")

# initialize the problem, assuming that both strikes for put option and call option would be 165
underlying = 165
strike = 165
rf = 0.0425
coupon = 0.0053
b = rf - coupon
ivol = np.linspace(0.1, 0.8, 200)

gbsm_call_values = np.zeros(len(ivol))
gbsm_put_values = np.zeros(len(ivol))

# start the calculation of call values, and checking if call values derived from two methods are close
for i in range(len(ivol)):
    gbsm_call_values[i] = gbsm(underlying, strike, ttm, rf, b, ivol[i], type="call")
    gbsm_put_values[i] = gbsm(underlying, strike, ttm, rf, b, ivol[i], type="put")

# checking put call parity
result = True
for i in range(len(gbsm_call_values)):
    if isclose(gbsm_call_values[i] + strike * exp(-rf*ttm), gbsm_put_values[i] + underlying, abs_tol = 0.1) == False:
        result = False
print(result)

# ploting values of put and call options with impied vols ranging from 0.1 to 0.8
plt.figure()
plt.plot(ivol, gbsm_call_values, label="Call")
plt.plot(ivol, gbsm_put_values, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Value of Option")
plt.legend()
plt.title("Same Strike")
plt.show()

# check what will happen if strike prices become difference for puts and calls
gbsm_call_values_diff = np.zeros(len(ivol))
gbsm_put_values_diff = np.zeros(len(ivol))

for i in range(len(ivol)):
    gbsm_call_values_diff[i] = gbsm(underlying, strike+20, ttm, rf, b, ivol[i], type="call")
    gbsm_put_values_diff[i] = gbsm(underlying, strike-20, ttm, rf, b, ivol[i], type="put")

# ploting values of put and call options with impied vols ranging from 0.1 to 0.8
plt.figure()
plt.plot(ivol, gbsm_call_values_diff, label="Call")
plt.plot(ivol, gbsm_put_values_diff, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Value of Option")
plt.legend()
plt.title("Different Strike")
plt.show()
