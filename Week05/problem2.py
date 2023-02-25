import functionlib
import pandas as pd
import numpy as np
from numpy.linalg import eigh

#### tests for written library

# read data
data = pd.read_csv("DailyPrices.csv")
dates = data.iloc[:,0]

# test for covariance matrix
# set up lists of weights, cumulative weights, and set up lamda equal to 0.5
weights = []
cum_weights = []
lamda = 0.5

# call both defined functions to get the result
functionlib.populateWeights(data.iloc[:,0],weights, cum_weights, lamda)
print("Length of exponential weights for lamda being 0.5:", len(weights))

# reverse w so that it corresponds to the ascending order of dates used in the DailyReturn.csv
rev_weights = weights[::-1]
covariance_matrix = functionlib.exwCovMat(data, dates, rev_weights)
print("Shape of exponentially weighted covariance matrix is:", covariance_matrix.shape)



# test for fixations of matrices
# set up positive semi-definite matrix, positive definite matrix, and root matrix
# generate a N * N non-psd correlation matrix
N = 3
psd = np.full((N, N), 0.9)
for i in range(N):
    psd[i][i] = 1.0
psd[0][1] = 0.7357
psd[1][0] = 0.7357
pd = np.array([[2, 1], [1, 2]])
root = np.array([[0, 0], [0, 0]])

# call defined functions and print results
print(functionlib.chol_psd_forpsd(root,psd))
print(functionlib.chol_psd_forpd(root,pd))
print(functionlib.near_PSD(psd, epsilon=0.0))
print(functionlib.Higham(psd, tolerance=1e-8))

# check eigenvalues of resulting matrices from near_PSD and Higham
mat1 = functionlib.near_PSD(psd, epsilon=0.0)
val1, vec1 = eigh(mat1)
print(val1)
mat2 = functionlib.Higham(psd, tolerance=1e-8)
val2, vec2 = eigh(mat2)
print(val2)



# test for simulation methods
# use the covariance matrix derived previously to test PCA simulation and direct simulation
result_PCA = functionlib.simulate_PCA(covariance_matrix, 10000, percent_explained=1)
result_direct = functionlib.direct_simulate(covariance_matrix, 10000)
print(result_PCA.shape)
print(result_direct.shape)



# test for VaR calculations
# calculate return based on given prices first
log_returns = functionlib.return_cal(data,method="log",datecol="Date")
del log_returns["Date"]
# demean log_returns
log_returns_aapl = log_returns.loc[:,"AAPL"]
log_returns_aapl = log_returns_aapl - log_returns_aapl.mean()
VaR_05_norm = functionlib.VaR_bas_dist(log_returns_aapl, alpha=0.05, dist="normal", n=10000)
VaR_05_t = functionlib.VaR_bas_dist(log_returns_aapl, alpha=0.05, dist="t", n=10000)
VaR_05_ar1 = functionlib.VaR_bas_dist(log_returns_aapl, alpha=0.05, dist="ar1", n=10000)
print("VaR_05_norm: {:.4f}; VaR_05_t: {:.4f}; VaR_05_ar1: {:.4f}".format(VaR_05_norm, VaR_05_t, VaR_05_ar1))

# set up portfolio holdings
hold_val = np.empty(len(data.columns[1:]))
hold_val.fill(1)
portfolio = {"Stock": data.columns[1:], "Holdings": hold_val}
portfolio = pd.DataFrame(portfolio)
current_p = data.iloc[-1,1:]
# call defined functions to test
del_norm = functionlib.del_norm_VaR(current_p, portfolio, log_returns, lamda=0.94, alpha=0.05)
his_VaR = functionlib.hist_VaR(current_p, portfolio, log_returns, alpha=0.05)
MC_VaR = functionlib.MC_VaR(current_p, portfolio, log_returns, n=1000, alpha=0.05)
print("Delta Normal VaR: {:.4f}; historic VaR: {:.4f}; MC VaR: {:.4f}".format(del_norm, his_VaR, MC_VaR))



# test for ES calculation
problem1 = pd.read_csv("problem1.csv")
ES = functionlib.cal_ES(problem1,alpha=0.05)
print("ES: {:.4f}".format(ES))



# test for F_norm
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[5,6],[7,8]])
diff = mat2 - mat1
functionlib.F_Norm(diff)



# test for cal_return has been done in previous section when testing VaR