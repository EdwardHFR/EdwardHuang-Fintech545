import numpy as np
import pandas as pd
import functionlib

# read in data first
cov = pd.read_csv("problem3_cov.csv")
cov = cov.to_numpy()
er = pd.read_csv("problem3_ER.csv")
rf = er["RF"][0]
expr = er.iloc[:,1:].to_numpy()[0]

### part a
# call defined function to derive the portfolio that maximizes Sharpe Ratio
result = functionlib.optimize_Sharpe(cov, expr, rf)
weights = result["weights"]

### part b
# call defined function to derive the risk parity portfolio
cov = pd.read_csv("problem3_cov.csv")
risk_par_weights = functionlib.risk_budget_parity(cov,B=None)

### part c
# call defined function to derive the correlation matrix of those three assets
corr = functionlib.cov2cor(cov)