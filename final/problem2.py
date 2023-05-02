import pandas as pd
import numpy as np
import functionlib

# first read in data from problem2.csv
data = pd.read_csv("problem2.csv")

# store data into variables separately
underlying = data["Underlying"]
strike = data["Strike"]
ivol = data["IV"]
ttm = data["TTM"]
ttm = ttm / 255
rf = data["RF"]
b = data["DivRate"]

### part a
# call function defined in functionlib to calculate price of the option
price = functionlib.gbsm(underlying, strike, ttm, rf, b, ivol, type="call")

### part b
# call function defined in functionlib to calculate delta of the option
delta = functionlib.delta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")

### part c
# call function defined in functionlib to calculate gamma of the option
gamma = functionlib.gamma_gbsm(underlying, strike, ttm, rf, b, ivol)

### part d
# call function defined in functionlib to calculate vega of the option
vega = functionlib.vega_gbsm(underlying, strike, ttm, rf, b, ivol)

### part e
# call function defined in functionlib to calculate rho of the option
rho = functionlib.rho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")

### part f
# set up the monte carlo simulations
np.random.seed(0)
sim_returns = np.random.normal(0, ivol, 5000)
sim_returns = pd.DataFrame(sim_returns)

# simulate prices with simulated returns
sim_prices = (1 + sim_returns) * underlying

# calculate the option price at each simulated price
new_ttm = ttm - 1 / 255
sim_opt_prices = []
for i in range(len(sim_prices)):
    temp_opt_price = functionlib.gbsm(sim_prices[0][i], strike, new_ttm, rf, b, ivol, type="call")
    sim_opt_prices.append(temp_opt_price[0])
sim_prices["opt"] = sim_opt_prices

# calculate profit and loss of each simulation
pnl = []
for i in range(len(sim_prices)):
    temp_pnl = sim_prices.iloc[i,0] - underlying + price - sim_prices.iloc[i,1]
    pnl.append(temp_pnl[0])
    
# call function defined in functionlib to calculate 5% VaR
pnl = np.array(pnl)
VaR = functionlib.cal_VaR(pnl,alpha=0.05)

### part g
# call function defined in functionlib to calculate ES
VaR, ES = functionlib.cal_ES(pnl,alpha=0.05)