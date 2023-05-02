import pandas as pd
import numpy as np
import functionlib

# first read in data
returns = pd.read_csv("problem4_returns.csv", parse_dates=["Date"])
s_w = pd.read_csv("problem4_startWeight.csv")


### part a
# copy paste a section of codes defined in a function called "rr_attribute" in my library
del returns["Date"]
len_data = len(returns)
    
pReturn = np.empty(len_data)
weights = np.empty((len_data, len(s_w.iloc[0,:])))
lastw = s_w.iloc[0,:]
    
### start return attribution process
for i in range(len_data):
    # Save Current Weights in Matrix
    weights[i,:] = lastw
    # Update Weights by return
    lastw = lastw * (1 + returns.iloc[i,:]).to_numpy()
    # Portfolio return is the sum of the updated weights
    pR = np.sum(lastw)
    # Normalize the wieghts back so sum = 1
    lastw = lastw / pR
    # Store the return
    pReturn[i] = pR - 1
weights = pd.DataFrame(weights,columns=["Asset1","Asset2","Asset3"])

### part b
# call defined function to conduct the ex-post return attribution
# Set the portfolio return in the Update Return DataFrame
returns["Portfolio"] = pReturn
    
# Calculate the total return
totalRet = np.exp(np.sum(np.log(pReturn + 1)))-1
# Calculate the Carino K
k = np.log(totalRet + 1 ) / totalRet
    
# Carino k_t is the ratio scaled by 1/K 
carinoK = np.log(1.0 + pReturn) / pReturn / k

# Calculate the return attribution
attrib = pd.DataFrame(data=returns * weights * carinoK.reshape(-1, 1), columns=returns.columns)
    
Attribution_return = pd.DataFrame({"Stock": ["TotalReturn", "Return Attribution"]})
for s in returns.columns:
    # Total Stock return over the period
    tr = np.exp(np.sum(np.log(returns[s] + 1))) - 1
    # Attribution Return (total portfolio return if we are updating the portfolio column)
    if s == 'Portfolio':
        atr = tr
    else:
        atr = attrib[s].sum()
    # Set the values
    Attribution_return[s] = [tr, atr]

### part c
### start risk attribution process
Y = returns * weights
X = np.hstack((np.ones((20,1)), pReturn.reshape(-1,1))))
# Calculate the Beta and discard the intercept
B = np.linalg.inv(X.T @ X) @ X.T @ Y
B = B[1:]

# Component SD is Beta times the standard Deviation of the portfolio
cSD = B * np.std(pReturn)
Attribution_risk = pd.DataFrame({"Stock": ["Vol Attribution"]})
for s in returns.columns:
    # Attribution Risk (total portfolio return if we are updating the portfolio column)
    if s == 'Portfolio':
        vol = np.std(pReturn)
    else:
        vol = cSD[s]
        vol = vol[1]
    # Set the values
    Attribution_risk[s] = [vol]
    