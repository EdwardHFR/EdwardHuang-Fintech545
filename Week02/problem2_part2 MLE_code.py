import pandas as pd
import numpy as np
import statsmodels.tools.tools as stmtools
import statsmodels.regression.linear_model as stmLRmodel
import math
import scipy


# from line 8 to line 20, these are lines of codes copied from problem2_part1 OLS_code.py
# read data from provided file
data = pd.read_csv("problem2.csv")

# extract X and Y from the data
x = data.loc[:,"x"]
y = data.loc[:,"y"]

# define a likelihood function used for optimization later, assuming the normality of errors/residuals
def LLfunc_nor(parameters, x, y): 
    # setting parameters
    beta1 = parameters[0] 
    beta0 = parameters[1] 
    sigma = parameters[2] 
    
    # derive estimated values of y
    y_est = beta1 * x + beta0 
    
    # compute log likelihood, but return the negative LL for convenience of later optimization
    LL = np.sum(scipy.stats.norm.logpdf(y-y_est, loc=0, scale=sigma))
    return -LL

# define another function to explicitly show constraints of our optimization problem
def constraints(parameters):
    sigma = parameters[2]
    return sigma

cons = {
    'type': 'ineq',
    'fun': constraints
}

# the step where optimization really takes place with the initial guessing of three parameters set in the LLfunc_t function to be [2,2,2]
lik_normal = scipy.optimize.minimize(LLfunc_nor, np.array([2, 2, 2]), args=(x,y))

# define a likelihood function used for optimization later, assuming a t-distribution of errors/residuals
def LLfunc_t(parameters, x, y): 
    # setting parameters
    beta1 = parameters[0] 
    beta0 = parameters[1] 
    sigma = parameters[2]
    df = parameters[3]
    
    # derive estimated values of y
    y_est = beta1 * x + beta0 
    
    # compute log likelihood, but return the negative LL for convenience of later optimization
    LL = np.sum(scipy.stats.t.logpdf(y-y_est, sigma, df))
    return -LL

# define another function to explicitly show constraints of our optimization problem
def constraints(parameters):
    sigma = parameters[2]
    return sigma

cons = {
    'type': 'ineq',
    'fun': constraints
}

# the step where optimization really takes place, with the initial guessing of four parameters set in the LLfunc_t function to be [1,1,1,1]
lik_t = scipy.optimize.minimize(LLfunc_t, np.array([1, 1, 1, 1]), args=(x,y))