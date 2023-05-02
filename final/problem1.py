import pandas as pd
import functionlib
import numpy as np
from numpy.linalg import eigh

# read in data from problem1.py
data = pd.read_csv("problem1.csv", parse_dates=["Date"])

# manipulate data into the format I would like to have (move column "Date" to the first column in the dataframe
date = data.pop("Date")
data.insert(0,"Date",date)

### part a
# call function from functionlib to calculate log returns
log_return = functionlib.return_cal(data,method="log",datecol="Date")

### part b
# call function from functionlib to derive a pairwise covariance matrix
prices = ["Price1","Price2","Price3"]
log_return_part = log_return[prices]
log_return_part_np = log_return_part.to_numpy()
pairwise_cov = functionlib.missing_cov(log_return_part_np, skipMiss=False, fun=np.cov)
pairwise_cov = pd.DataFrame(pairwise_cov,index=prices,columns=prices)

### part c
# first check eigenvalues of the resulting matrix
e_val, e_vec = eigh(pairwise_cov)

# call function from functionlib to use near_psd method to correct the matrix
corrected_cov = functionlib.near_PSD(pairwise_cov, epsilon=0.0)

