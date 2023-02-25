import math
import numpy as np
import sys
import pandas as pd
from numpy.linalg import eigh
import itertools
from scipy.stats import norm, t
import statsmodels.api as sm

#####calculation of covariance matrix#####START
# define a function to calculate exponential weights
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

# define a function to calculate the exponentially weighted covariance matrix
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
#####calculation of covariance matrix#####END



#####fix non-psd matrices#####START
# cholesky factorization for psd matrices
def chol_psd_forpsd(root,a):
    n = a.shape
    # initialize root matrix with zeros
    root = np.zeros(n)
    
    for j in range(n[0]):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            s = np.dot(root[j,:(j-1)], root[j,:(j-1)])
            
        # working on diagonal elements
        temp = a[j,j] - s
        if temp <= 0 and temp >= -1e-8:
            temp = 0
        root[j,j] = math.sqrt(temp)
        
        # check for zero eigenvalues; set columns to zero if we have one(s)
        if root[j,j] == 0:
            root[j,(j+1):] = 0
        else:
            ir = 1/root[j,j]
            for i in range((j+1),n[0]):
                s = np.dot(root[i,:(j-1)], root[j,:(j-1)])
                root[i,j] = (a[i,j] - s) * ir
    return root

# cholesky factorization for pd matrices                
def chol_psd_forpd(root,a):
    n = a.shape
    # initialize root matrix with zeros
    root = np.zeros(n)
    
    for j in range(n[0]):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            s = np.dot(root[j,:(j-1)], root[j,:(j-1)])
            
        # working on diagonal elements
        temp = a[j,j] - s
        root[j,j] = math.sqrt(temp)
        
        ir = 1/root[j,j]
        # update off diagonal rows of the column
        for i in range((j+1),n[0]):
            s = np.dot(root[i,:(j-1)], root[j,:(j-1)])
            root[i,j] = (a[i,j] - s) * ir
    return root            

# define a function calculating PSD via near_PSD
def near_PSD(matrix, epsilon=0.0):
    # check if the matrix is a correlation matrix - if all numbers on the diagonal are one
    matrix_diag = np.diag(matrix)
    for i in matrix_diag:
        assert i == 1
    
    # calculate the eigenvalues and eigenvectors
    e_val, e_vec = eigh(matrix)
    
    # sort eigenvalues and corresponding eigenvectors in a descending order
    index = np.argsort(-1 * e_val)
    d_e_val = e_val[index]
    d_e_vec = e_vec[:,index]
    
    # set eigenvalues that are smaller than epsilon to epsilon
    d_e_val[d_e_val < epsilon] = epsilon
    
    # construct the scaling diagonal matrix, calculating t(s) and store them into the list called t_vec
    t_vec = []
    for i in range(len(d_e_val)):
        sum_t = 0
        for j in range(len(d_e_val)):
            t = pow(d_e_vec[i][j],2) * d_e_val[j]
            sum_t += t
        t_i = 1 / sum_t
        t_vec.append(t_i)
    
    # construct the resulting near_PSD matrix
    B_matrix = np.diag(np.sqrt(t_vec)) @ d_e_vec @ np.diag(np.sqrt(d_e_val))
    B_matrix_transpose = B_matrix.transpose()
    C_prime_matrix = B_matrix @ B_matrix_transpose
    
    # checking if eigenvalues are all non-negative now (assuming all significantly small eigenvalues are zero, the tolerance level here is set to be -1e-8)
    result_vals, result_vecs = eigh(C_prime_matrix)
    neg_result_vals = result_vals[result_vals < 0]
    if neg_result_vals.any() < -1e-8:
        print("There are still significantly negative eigenvalues, recommend to run the function again over the result until a PSD is generated")
    
    return C_prime_matrix


# define a function calculating PSD via Higham's method
def Higham(A, tolerance=1e-8):
    # set up delta S, Y, and gamma
    delta_s = np.full(A.shape,0)
    Y = A.copy()
    gamma_last = sys.float_info.max
    gamma_now = 0
    
    # start the actual iteration
    for i in itertools.count(start=1):        
        R = Y - delta_s
        
        # conduct the second projection of Higham's method over R
        Rval, Rvec = eigh(R)
        Rval[Rval < 0] = 0
        Rvec_transpose = Rvec.transpose()
        
        X = Rvec @ np.diag(Rval) @ Rvec_transpose
        
        delta_s = X - R
        
        # conduct the first projection of Higham's method over X
        size_X = X.shape
        for i in range(size_X[0]):
            for j in range(size_X[1]):
                if i == j:
                    Y[i][j] = 1
                else:
                    Y[i][j] = X[i][j]
        
        difference_mat = Y - A
        gamma_now = F_Norm(difference_mat)
        
        # get eigenvalues and eigenvectors of updated Y
        Yval, Yvec = eigh(Y)
        
        #set breaking conditions
        if np.amin(Yval) > -1*tolerance:
            break
        else:
            gamma_last = gamma_now
    
    return Y
#####fix non-psd matrices#####END



#####simulations method (e.g PCA)#####START
# PCA to simulate the system through defining a new function
def simulate_PCA(a, nsim, percent_explained=1):
    # calculate the eigenvalues and eigenvectors of derived matrix, and sort eigenvalues from largest to smallest
    e_val, e_vec = eigh(a)
    sort_index = np.argsort(-1 * e_val)
    d_sorted_e_val = e_val[sort_index]
    d_sorted_e_vec = e_vec[:,sort_index]

    # we assume all negative eigenvalues derived are zero, since they are effectively zero (larger than -1e-8)
    assert np.amin(d_sorted_e_val) > -1e-8
    d_sorted_e_val[d_sorted_e_val<0] = 0
    
    # calculate the sum of all eigenvalues
    e_sum = sum(d_sorted_e_val)

    # choose a certain number of eigenvalues from the descending list of all eigenvalues so that the system explains the same percent of variance as the level inputed as parameter "percent_explained"
    total_percent = []
    sum_percent = 0
    for i in range(len(d_sorted_e_val)):
        each_percent = d_sorted_e_val[i] / e_sum
        sum_percent += each_percent
        total_percent.append(sum_percent)
    total_percent_np = np.array(total_percent)
    diff = total_percent_np - percent_explained
    abs_diff = abs(diff)
    index = np.where(abs_diff==abs_diff.min())
    
    # update eigenvalues and eigenvectors with the list of indices we generate above
    upd_e_val = d_sorted_e_val[:(index[0][0]+1)]
    upd_e_vec = d_sorted_e_vec[:,:(index[0][0]+1)]
    
    # construct the matrix for the simulating process
    B = upd_e_vec @ np.diag(np.sqrt(upd_e_val))
    r = np.random.randn(len(upd_e_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t

# direct simulation
def direct_simulate(a, nsim):
    # get eigenvalues and eigenvectors of the input matrix
    val, vec = eigh(a)
    sort_index = np.argsort(-1 * val)
    d_sorted_val = val[sort_index]
    d_sorted_vec = vec[:,sort_index]
    
    # to check if all eigenvalues are non-negative or negative but effectively zero, and set all effectively-zero eigenvalues to zero
    assert np.amin(d_sorted_val) > -1e-8
    d_sorted_val[d_sorted_val<0] = 0
    
    # construct the matrix for the simulating process
    B = d_sorted_vec @ np.diag(np.sqrt(d_sorted_val))
    r = np.random.randn(len(d_sorted_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t
#####simulations method (e.g PCA)#####END



#####VaR calculation#####START
# assume no distribution
def cal_VaR(x,alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    return -VaR

# another way without assuming any distribution
def comp_VaR(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)

# assume basic distributions (only normal, t, and AR(1) are available in this function)
def VaR_bas_dist(data, alpha=0.05, dist="normal", n=10000):
    # demean data
    data = data - data.mean()
    if dist=="normal":
        fit_result = norm.fit(data)
        return -norm.ppf(alpha, loc=fit_result[0], scale=fit_result[1])
    elif dist=="t":
        fit_result = t.fit(data)
        return -t.ppf(alpha, df=fit_result[0], loc=fit_result[1], scale=fit_result[2])
    elif dist=="ar1":
        mod = sm.tsa.ARIMA(data, order=(1, 0, 0))
        fit_result = mod.fit()
        summary = fit_result.summary()
        m = float(summary.tables[1].data[1][1])
        a1 = float(summary.tables[1].data[2][1])
        s = math.sqrt(float(summary.tables[1].data[3][1]))
        out = np.zeros(n)
        sim = np.random.normal(size=n)
        data_last = data.iloc[-1] - m
        for i in range(n):
            out[i] = a1 * data_last + sim[i] * s + m
        return comp_VaR(out, mean=out.mean())
    else:
        return "Invalid distribution in this method."

# delta normal VaR for portfolios (check the order of data, if it is from farthest to nearest, this is correct; if not, plz modify the code or reverse the order to "farthest and nearest"; make sure that there should not be a date column in returns)
def del_norm_VaR(current_prices, holdings, returns, lamda=0.94, alpha=0.05):
    # demean returns
    returns -= returns.mean()
    w = []
    cw = []
    PV = 0
    delta = np.zeros(len(holdings))
    populateWeights(returns, w, cw, lamda)
    w = w[::-1]
    cov = exwCovMat(returns, w)
    for i in range(len(holdings)):
        temp_holding = holdings.iloc[i,-1] 
        value = temp_holding * current_prices[i]
        PV += value
        delta[i] = value
    delta = delta / PV
    fac = math.sqrt(np.transpose(delta) @ cov @ delta)
    VaR = -PV * norm.ppf(alpha, loc=0, scale=1) * fac
    return VaR

# historic VaR (note that when used, check how returns are derived; if they are log returns, you are fine; if they are arithmetic returns, change the way you calculate simulated prices; also, there should not be a date column in returns)
def hist_VaR(current_prices, holdings, returns, alpha=0.05):
    # demean returns
    returns -= returns.mean()
    PV = 0
    for i in range(len(holdings)):
        value = holdings.iloc[i,-1] * current_prices[i]
        PV += value
    sim_prices = (np.exp(returns)) * np.transpose(current_prices)
    port_values = np.dot(sim_prices, holdings.iloc[:,-1])
    port_values_sorted = np.sort(port_values)
    index = np.floor(alpha*len(returns))
    VaR = PV - port_values_sorted[int(index-1)]
    return VaR

# Monte Carlo normal VaR (note that when used, check how returns are derived; if they are log returns, you are fine; if they are arithmetic returns, change the way you calculate simulated prices)
def MC_VaR(current_prices, holdings, returns, n=10000, alpha=0.05):
    # demean returns
    returns -= returns.mean()
    PV = 0
    for i in range(len(holdings)):
        value = holdings.iloc[i,-1] * current_prices[i]
        PV += value
    sim_returns = np.random.multivariate_normal(returns.mean(), returns.cov(), (1,len(holdings),n))
    sim_returns = np.transpose(sim_returns)
    sim_prices = (np.exp(returns)) * np.transpose(current_prices)
    port_values = np.dot(sim_prices, holdings.iloc[:,-1])
    port_values_sorted = np.sort(port_values)
    index = np.floor(alpha*n)
    VaR = PV - port_values_sorted[int(index-1)]
    return VaR
    
#####VaR calculation#####END



#####ES calculation#####START
# ES calculation of individual data
def cal_ES(x,alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    ES = xs[0:idn].mean()
    return -ES
#####ES calculation#####END



#####norm calculation#####START
def F_Norm(M):
    # get the number of rows and columns of the input matrix M
    size = M.shape
    rows = size[0]
    columns = size[1]
    
    # compute the norm
    sum = 0
    for i in range(rows):
        for j in range(columns):
            square = pow(M[i][j],2)
            sum += square
    
    return math.sqrt(sum)
#####norm calculation#####END



#####return calculation#####START
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
#####return calculation#####END