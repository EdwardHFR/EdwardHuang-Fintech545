import pandas as pd
import math
import sys
import numpy as np
from scipy.stats import norm, t
import statsmodels.api as sm


# define a function similar to the one called "reture_calculate" in class
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


# copy codes from week3 for calculation of weights
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
        

# copy codes from week3 for calculation of covariance matrix
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
        
    
    
    

# read input file
data = pd.read_csv('DailyPrices.csv')

# call the function defined above to calculate arithmetic returns
returns = return_cal(data)
                
# extract returns of META from the pandas dataframe derived above            
meta_returns = returns.loc[:,"META"].copy()

# de-mean returns of META by subtracting the mean of returns of META from each return
meta_mean = meta_returns.mean()
meta_returns -= meta_mean
meta_returns = pd.DataFrame(meta_returns)
sd = meta_returns.std()

####### calculating VaRs
### using normal distribution
VaR_05_nor = -norm.ppf(0.05, loc=0, scale=sd)
VaR_01_nor = -norm.ppf(0.01, loc=0, scale=sd)
print("VaR_05 using normal distribution is {:.4f}%".format(VaR_05_nor[0]*100))
print("VaR_01 using normal distribution is {:.4f}%".format(VaR_01_nor[0]*100))

### using normal distribution with an exponentially weighted variance (lamda = 0.94)
# initializing weights vector
w = []
cw = []
populateWeights(meta_returns,w,cw, 0.94)
# reversing the weights vector to match the order of dates
w = w[::-1]
meta_var_mat = exwCovMat(meta_returns, w)
meta_sigma = math.sqrt(meta_var_mat.iloc[0,0])
VaR_05_exnor = -norm.ppf(0.05, loc=0, scale=meta_sigma)
VaR_01_exnor = -norm.ppf(0.01, loc=0, scale=meta_sigma)
print("VaR_05 using normal distribution with exponential weights applied is {:.4f}%".format(VaR_05_exnor*100))
print("VaR_01 using normal distribution with exponential weights applied is {:.4f}%".format(VaR_01_exnor*100))

### using a MLE fitted T distribution
# here, for convenience, use the scipy package to fit t distribution with mle
fit_results = t.fit(meta_returns)
df = fit_results[0]
loc = fit_results[1]
scale = fit_results[2]
VaR_05_mlet = -t.ppf(0.05, df=df, loc=loc, scale=scale)
VaR_01_mlet = -t.ppf(0.01, df=df, loc=loc, scale=scale)
print("VaR_05 using MLE fitted t distribution is {:.4f}%".format(VaR_05_mlet*100))
print("VaR_01 using MLE fitted t distribution is {:.4f}%".format(VaR_01_mlet*100))

### using a fitted AR(1) model
# here, for convenience, use the AR(1) model simulated in statsmodels package
mod = sm.tsa.ARIMA(meta_returns, order=(1, 0, 0))
results = mod.fit()
sigma = np.std(results.resid)
sim = np.empty(10000)
for i in range(10000):
    sim[i] =  results.params[1] * (meta_returns.iloc[0]) + sigma * np.random.normal()
VaR_05_ar1 = -np.percentile(sim, 0.05*100)
VaR_01_ar1 = -np.percentile(sim, 0.01*100)
print("VaR_05 using fitted AR(1) model is {:.4f}%".format(VaR_05_ar1*100))
print("VaR_01 using fitted AR(1) model is {:.4f}%".format(VaR_01_ar1*100))

### using a historic simulation
# in historic simulation, there is no distributions assumed
VaR_05_hist = meta_returns.mean() - np.quantile(meta_returns,0.05)
VaR_01_hist = meta_returns.mean() - np.quantile(meta_returns,0.01)
print("VaR_05 using historic simulation is {:.4f}%".format(VaR_05_hist[0]*100))
print("VaR_01 using historic simulation is {:.4f}%".format(VaR_01_hist[0]*100))