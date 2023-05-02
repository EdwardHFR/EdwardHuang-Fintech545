import pandas as pd
import numpy as np
import functionlib

# first read in data
prices = pd.read_csv("problem5.csv", parse_dates =["Date"])

# manipulate data into the format I would like to have (move column "Date" to the first column in the dataframe
date = prices.pop("Date")
prices.insert(0,"Date",date)

# call function to calculate arithmetic returns
returns = functionlib.return_cal(prices,method="discrete",datecol="Date")
returns.pop("Date")

# apply Gaussian Copula on returns
sim_returns = functionlib.Gaussian_Copula_same(returns,dist="t",N=1000)

### part a
# record last prices of each asset
price1 = prices["Price1"].tolist()
price1 = price1[-1]

price2 = prices["Price2"].tolist()
price2 = price2[-1]

price3 = prices["Price3"].tolist()
price3 = price3[-1]

price4 = prices["Price4"].tolist()
price4 = price4[-1]

# multiply last prices with sim returns
last_prices = np.array([price1,price2,price3,price4])
sim_prices = (1 + sim_returns) * last_prices

# calculate VaR for each asset
VaR = []
for i in sim_prices.columns:
    value = -functionlib.cal_VaR(sim_prices[i],alpha=0.05)
    VaR.append(value)

VaR = -(np.array(VaR) - np.array(last_prices))

### part b

### part c
last_total_value = np.sum(last_prices)
total_value = []
for i in range(len(sim_prices)):
    sum = np.sum(sim_prices.iloc[i,:])
    total_value.append(sum)
sim_prices["total"] = total_value

# calculate VaR for all four assets
diff = sim_prices["total"] - last_total_value
VaR_total = functionlib.cal_VaR(diff,alpha=0.05)