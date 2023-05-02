import math
import numpy as np
import sys
import pandas as pd
from numpy.linalg import eigh
import itertools
from scipy.stats import norm, t, spearmanr
import statsmodels.api as sm
from scipy.optimize import fsolve, minimize
import inspect
from sklearn.linear_model import LinearRegression
import statsmodels.graphics.tsaplots as stmplot
import matplotlib.pyplot as plt
import seaborn as sns

#####fitting data into distributions#####START
# define a function to fit data into several distributions using MLE (either normal distribution or t distribution) (note that log likelihood returned from this function is negated)
def fit_MLE(x,y,dist="norm"):
    # define another function to use MLE to fit data into normal distribution
    def LLfunc_nor(parameters, x, y): 
        # setting parameters
        beta1 = parameters[0] 
        beta0 = parameters[1] 
        sigma = parameters[2] 
    
        # derive estimated values of y
        y_est = beta1 * x + beta0 
    
        # compute log likelihood, but return the negative LL for convenience of later optimization
        LL = np.sum(norm.logpdf(y-y_est, loc=0, scale=sigma))
        return -LL
    # define another function to use MLE to fit data into t distribution
    def LLfunc_t(parameters, x, y): 
        # setting parameters
        beta1 = parameters[0] 
        beta0 = parameters[1] 
        sigma = parameters[2]
        df = parameters[3]
    
        # derive estimated values of y
        y_est = beta1 * x + beta0 
    
        # compute log likelihood, but return the negative LL for convenience of later optimization
        LL = np.sum(t.logpdf(y-y_est, sigma, df))
        return -LL
    # define another function to explicitly show constraints of our optimization problem
    def constraints(parameters):
        sigma = parameters[2]
        return sigma

    cons = {
        'type': 'ineq',
        'fun': constraints
    }
    
    if dist == "norm":
        lik_normal = minimize(LLfunc_nor, np.array([2, 2, 2]), args=(x,y))
        return {"Log_likelihood" : lik_normal.fun, "Parameters" : lik_normal.x}
    elif dist == "t":
        lik_t = minimize(LLfunc_t, np.array([1, 1, 1, 1]), args=(x,y))
        return {"Log_likelihood" : lik_t.fun, "Parameters" : lik_t.x}
    
# define a function to use linear regression to fit data
def linear_reg(x,y):
    reg = LinearRegression().fit(x,y)
    return {"Coefficients" : reg.coef_, "Intercept" : reg.intercept_}

# define a function to fit data into a AR(x) model, simulate data with the funtion - y1 = a1*y0 + s*np.random.normal + m (注意y0在一开始就要先减m）
def fit_AR1(data):
    mod = sm.tsa.ARIMA(data, order=(1, 0, 0))
    results = mod.fit()
    summary = results.summary()
    m = float(summary.tables[1].data[1][1])
    a1 = float(summary.tables[1].data[2][1])
    s = np.sqrt(float(summary.tables[1].data[3][1]))
    return {"Mean" : m, "Coefficient" : a1, "sqrt_Sig2" : s}
#####fitting data into distributions#####END



#####calculation of covariance matrix#####START
# basic transformation from correlation matrix to covariance matrix
def cor2cov(corel,vols):
    covar = np.diag(vols).dot(corel).dot(np.diag(vols))
    return covar

# basic transformation from covariance matrix to correlation matrix
def cov2cor(cov):
    std = np.sqrt(np.diag(cov))
    for i in range(len(cov.columns)):
        for j in range(len(cov.index)):
            cov.iloc[j,i] = cov.iloc[j,i] / (std[i] * std[j])
    return cov

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

# calculate covariance matrix for data with missing values(注意：x这里只能是numpy.array的形式，其他形式都要转换成numpy.array)
def missing_cov(x, skipMiss=True, fun=np.cov):
    n, m = x.shape
    nMiss = np.sum([np.isnan(x[:, i]) for i in range(m)], axis=1)
    
    if np.sum(nMiss) == 0:
        return fun(x)
    
    idxMissing = [set(np.where(np.isnan(x[:, i]))[0]) for i in range(m)]
    
    if skipMiss:
        rows = set(range(n))
        for c in range(m):
            for rm in idxMissing[c]:
                rows.discard(rm)
        rows = sorted(rows)
        #如果fun不是np.cov的话，下面这行.T注意删除
        return fun(x[rows, :].T)
    
    else:
        out = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                rows = set(range(n))
                for c in (i, j):
                    for rm in idxMissing[c]:
                        rows.discard(rm)
                rows = sorted(rows)
                #print(x[rows][:,[i,j]].shape)
                #如果fun不是np.cov的话，下面这行.T注意删除
                temp_out = fun(x[rows][:,[i,j]].T)
                out[i,j] = temp_out[0, 1]
        return out
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
        root[j,j] = np.sqrt(temp)
        
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
        root[j,j] = np.sqrt(temp)
        
        ir = 1/root[j,j]
        # update off diagonal rows of the column
        for i in range((j+1),n[0]):
            s = np.dot(root[i,:(j-1)], root[j,:(j-1)])
            root[i,j] = (a[i,j] - s) * ir
    return root            

# define a function calculating PSD via near_PSD
def near_PSD(matrix, epsilon=0.0):
    # check if the matrix is a correlation matrix - if all numbers on the diagonal are one
    # matrix_diag = np.diag(matrix)
    # for i in matrix_diag:
    #    assert i == 1
    
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

# AR(1) simulation
def AR1_simulate(data,n=10000,ahead=1):
    params = fit_AR1(data)
    l = len(data)
    out = pd.DataFrame(0,index=range(ahead), columns=range(n))
    data_last = data[l-1] - params["Mean"]
    
    for i in range(n):
        datal = data_last
        next = 0
        for j in range(ahead):
            next = params["Coefficient"] * datal + params["sqrt_Sig2"] * np.random.normal()
            datal = next
            out[j,i] = next
    
    out = out + params["Mean"]
    return out
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
        s = np.sqrt(float(summary.tables[1].data[3][1]))
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
    fac = np.sqrt(np.transpose(delta) @ cov @ delta)
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
    return -VaR,-ES
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
    
    return np.sqrt(sum)
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
                returns.iloc[i,j] = np.log(returns.iloc[i,j])
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



#####Option Pricing#####START
# calculate implied volatility for GBSM
def implied_vol_gbsm(underlying, strike, ttm, rf, b, price, type="call"):
    f = lambda ivol: gbsm(underlying, strike, ttm, rf, b, ivol, type="call") - price
    result = fsolve(f,0.5)
    return result

# calculate implied volatility for American options with dividends
def implied_vol_americandiv(underlying, strike, ttm, rf, divAmts, divTimes, N, price, type="call"):
    f = lambda ivol: bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call") - price
    result = fsolve(f,0.5)
    return result
    
# Black Scholes Model for European option
def gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        return underlying * np.exp((b-rf)*ttm) * norm.cdf(d1) - strike*np.exp(-rf*ttm)*norm.cdf(d2)
    elif type == "put":
        return strike*np.exp(-rf*ttm)*norm.cdf(-d2) - underlying*np.exp((b-rf)*ttm)*norm.cdf(-d1)
    else:
        print("Invalid type of option")
        
# binomial trees used to price American option with no dividends
def bt_american(underlying, strike, ttm, rf, b, ivol, N, otype="call"):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    if otype == "call":
        z = 1
    elif otype == "put":
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = [0.0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u**i * d**(j-i)
            optionValues[idx] = max(0, z * (price - strike))
            
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1, j+1)] + pd*optionValues[idxFunc(i, j+1)]))
    
    return optionValues[0]

# binomial trees used to price American option with dividends
def bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call"):
    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_american(underlying, strike, ttm, rf, rf, ivol, N, otype=type)
    
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    if type == "call":
        z = 1
    elif type == "put":
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nDiv = len(divTimes)
    nNodes = nNodeFunc(divTimes[0])

    optionValues = np.zeros(len(range(nNodes)))

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u ** i * d ** (j - i)

            if j < divTimes[0]:
                # times before the dividend working backward induction
                optionValues[idx] = max(0, z * (price - strike))
                optionValues[idx] = max(optionValues[idx], df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]))
            else:
                # time of the dividend
                valNoExercise = bt_american_div(price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, divAmts[1:], [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0], type=type)
                valExercise = max(0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]
#####Option Pricing#####END



#####Greeks with Options#####START
# calculate delta of options with closed-form formulas
def delta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))

    if type == "call":
        return np.exp((b - rf) * ttm) * norm.cdf(d1)
    elif type == "put":
        return np.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)
    else:
        print("Invalid type of option")
        
# calculate Gamma of options with closed-form formulas
def gamma_gbsm(underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    result = norm.pdf(d1) * np.exp((b - rf) * ttm) / (underlying * ivol * np.sqrt(ttm))
    return result

# calculate Vega of options with closed-form formulas
def vega_gbsm(underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    result = underlying * norm.pdf(d1) * np.exp((b - rf) * ttm) * np.sqrt(ttm)
    return result

# calculate Theta of options with closed-form formulas
def theta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        result_call = - underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm)) - (b - rf) * underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - rf * strike * np.exp(-rf * ttm) * norm.cdf(d2)
        return result_call
    elif type == "put":
        result_put = - underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm)) + (b - rf) * underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1) + rf * strike * np.exp(-rf * ttm) * norm.cdf(-d2)
        return result_put
    else:
        print("Invalid type of option")

# calculate Rho of options with closed-form formulas
def rho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        return ttm * strike * np.exp(-rf * ttm) * norm.cdf(d2)
    elif type == "put":
        return -ttm * strike * np.exp(-rf * ttm) * norm.cdf(-d2)
    else:
        print("Invalid type of option")

# calculate Carry Rho of options with closed-form formulas
def crho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    
    if type == "call":
        return ttm * underlying * np.exp((b - rf) * ttm) * norm.cdf(d1)
    elif type == "put":
        return -ttm * underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1)
    else:
        print("Invalid type of option")
#####Greeks with Options#####END



#####Partial Derivative#####START
# define a function to calculate first order derivative with central difference
def first_order_derivative(func, x, delta=1e-3):
    return (func(x + delta) - func(x - delta)) / (2 * delta)

# define a function to calculate second order derivative with central difference
def second_order_derivative(func, x, delta=1e-3):
    return (func(x + delta) - 2 * func(x) + func(x - delta)) / delta ** 2

# incorporate above functions to calculate partial derivatives of indicated functions and return corresponding partial derivative functions
def partial_derivative(func, arg_name, delta=1e-3, order=1):
    arg_names = inspect.signature(func).parameters.keys()
    derivative_functions = {1: first_order_derivative, 2: second_order_derivative}

    def partial_func(*args, **kwargs):
        arg_values = dict(zip(arg_names, args))
        arg_values.update(kwargs)
        x = arg_values.pop(arg_name)

        def f(xi):
            arg_values[arg_name] = xi
            return func(**arg_values)

        return derivative_functions[order](f, x, delta)

    return partial_func
#####Partial Derivative#####END



#####Portfolio Optimization#####START
# calculate minimal risk with target return
def optimize_risk(covar, expected_r, R):
    # Define objective function
    def objective(w):
        return w @ covar @ w.T

    # Define constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: expected_r @ w - R},
    ]

    # Define bounds
    bounds = [(0, None)] * len(expected_r)

    # Define initial guess
    x0 = np.full(len(expected_r), 1/len(expected_r))

    # Use minimize function to solve optimization problem
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    # Return the objective value (risk) and the portfolio weights
    return {"risk": result.fun, "weights": result.x, "R": R}

# calculate maximized Sharpe Ratio with target return
def optimize_Sharpe(covar, expected_r, R, rf):
    # Define objective function
    def negative_Sharpe(w):
        returns = np.dot(expected_r, w)
        std = np.sqrt(w @ covar @ w.T)
        return -(returns - rf) / std

    # Define constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: expected_r @ w - R},
    ]

    # Define bounds
    bounds = [(0, None)] * len(expected_r)

    # Define initial guess
    x0 = np.full(len(expected_r), 1/len(expected_r))

    # Use minimize function to solve optimization problem
    result = minimize(negative_Sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    # Return the objective value (risk) and the portfolio weights
    return {"max_Sharpe_Ratio": -result.fun, "weights": result.x}

# calculate maximized Sharpe Ratio without target return
def optimize_Sharpe(covar, expected_r, rf):
    # Define objective function
    def negative_Sharpe(w):
        returns = np.dot(expected_r, w)
        std = np.sqrt(w @ covar @ w.T)
        return -(returns - rf) / std

    # Define constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    # Define bounds
    bounds = [(0, None)] * len(expected_r)

    # Define initial guess
    x0 = np.full(len(expected_r), 1/len(expected_r))

    # Use minimize function to solve optimization problem
    result = minimize(negative_Sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    # Return the objective value (risk) and the portfolio weights
    return {"max_Sharpe_Ratio": -result.fun, "weights": result.x}



#####Copula#####START
# Gaussian Copula for same distributions
def Gaussian_Copula_same(data,dist="norm",N=500):
    # define a function to fit returns to indicated distribution
    def returns_fit(data,dist_fit="norm"):
        if dist == "norm":
            out = {"example": ["loc", "scale", "dist"]}
            for i in data.columns:
                temp_results = norm.fit(data.loc[:,i])
                temp_dist = norm(temp_results[0], temp_results[1])
                temp = [temp_results[0], temp_results[1], temp_dist]
                out[i] = temp
            return out
        elif dist == "t":
            out = {"example": ["df", "loc", "scale", "dist"]}
            for i in data.columns:
                temp_results = t.fit(data.loc[:,i])
                temp_dist = t(temp_results[0], temp_results[1], temp_results[2])
                temp = [temp_results[0], temp_results[1], temp_results[2], temp_dist]
                out[i] = temp
            return out
        
    # define a function to calculate U matrix
    def generate_U(data, fits):
        assert len(data.columns) == len(fits) - 1
        temp = []
        df = pd.DataFrame()
        for i in data.columns:
            temp_cdf = fits[i][-1].cdf(data.loc[:,i])
            df[i] = temp_cdf
        return df
    
    # define a function to convert to sim values
    def convert_sim_values(fit, sim_U):
        out = pd.DataFrame()
        for i in sim_U.columns:
            out[i] = fit[i][-1].ppf(sim_U.loc[:,i])
        return out
    
    # fit data into indicated distribution
    data_fit = returns_fit(data,dist_fit=dist)
    # generate U matrix from data and fitted model
    data_U = generate_U(data,data_fit)
    # calculate spearman correlation matrix for input data
    data_spear = pd.DataFrame(spearmanr(data_U)[0], columns=data.columns, index=data.columns)
    # simulate values
    data_sim = np.random.multivariate_normal(np.zeros(len(data.columns)), data_spear, (1,len(data.columns),N))[0][0]
    # convert to U_sim
    data_sim_U = pd.DataFrame(norm.cdf(data_sim), columns=data.columns)
    # convert U_sim to sim values for each portfolio
    data_simout = convert_sim_values(data_fit, data_sim_U)
    return data_simout

# Gaussian Copula for different distributions （注意，这里的data需要是dataframe的形式，需要在第一行输入该列的数据是什么分布：either “norm” or “t”）
def Gaussian_Copula_diff(data,N=500):
    # define a function to fit returns to indicated distribution
    def returns_fit(data):
        out = {"example": ["df", "loc", "scale", "dist"]}
        for i in data.columns:
            if data.loc[data.index[0],i] == "norm":
                temp_results = norm.fit(data.loc[data.index[1]:,i])
                temp_dist = norm(temp_results[0], temp_results[1])
                temp = [temp_results[0], temp_results[1], temp_dist]
                out[i] = [0] + temp
            elif data.loc[data.index[0],i] == "t":
                temp_results = t.fit(data.loc[data.index[1]:,i])
                temp_dist = t(temp_results[0], temp_results[1], temp_results[2])
                temp = [temp_results[0], temp_results[1], temp_results[2], temp_dist]
                out[i] = temp
        return out
        
    # define a function to calculate U matrix
    def generate_U(data, fits):
        assert len(data.columns) == len(fits) - 1
        temp = []
        df = pd.DataFrame()
        for i in data.columns:
            temp_cdf = fits[i][-1].cdf(data.loc[data.index[1]:,i])
            df[i] = temp_cdf
        return df
    
    # define a function to convert to sim values
    def convert_sim_values(fit, sim_U):
        out = pd.DataFrame()
        for i in sim_U.columns:
            out[i] = fit[i][-1].ppf(sim_U.loc[:,i])
        return out
    
    # fit data into indicated distribution
    data_fit = returns_fit(data)
    # generate U matrix from data and fitted model
    data_U = generate_U(data,data_fit)
    # calculate spearman correlation matrix for input data
    data_spear = pd.DataFrame(spearmanr(data_U)[0], columns=data.columns, index=data.columns)
    # simulate values
    data_sim = np.random.multivariate_normal(np.zeros(len(data.columns)), data_spear, (1,len(data.columns),N))[0][0]
    # convert to U_sim
    data_sim_U = pd.DataFrame(norm.cdf(data_sim), columns=data.columns)
    # convert U_sim to sim values for each portfolio
    data_simout = convert_sim_values(data_fit, data_sim_U)
    return data_simout
#####Copula#####END



#####Risk & Return Attribution#####START
# 最简单的rr attribution，不考虑residual和factor component
# data是dataframe形式的return；w是1*n的格式储存weights，有多少个stock的return就有多少个weights
def rr_attribute(data,w):
    len_data = len(data)
    
    pReturn = np.empty(len_data)
    weights = np.empty((len_data, len(w)))
    lastw = w
    
    ### start return attribution process
    for i in range(len_data):
        # Save Current Weights in Matrix
        weights[i,:] = lastw
        # Update Weights by return
        lastw = lastw * (1 + data.iloc[i,:])
        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastw)
        # Normalize the wieghts back so sum = 1
        lastw = lastw / pR
        # Store the return
        pReturn[i] = pR - 1
    
    # Set the portfolio return in the Update Return DataFrame
    data["Portfolio"] = pReturn
    
    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1)))-1
    # Calculate the Carino K
    k = np.log(totalRet + 1 ) / totalRet
    
    # Carino k_t is the ratio scaled by 1/K 
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(data=data * weights * carinoK.reshape(-1, 1), columns=data.columns+["Portfolio"])
    
    Attribution_return = pd.DataFrame({"Stock": ["TotalReturn", "Return Attribution"]})
    for s in data.columns:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(data[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        if s == 'Portfolio':
            atr = tr
        else:
            atr = attrib[s].sum()
        # Set the values
        Attribution_return[s] = [tr, atr]
    
    
    ### start risk attribution process
    Y = data * weights
    X = np.hstack((np.ones((len(pReturn,1)), pReturn.reshape(-1,1))))
    # Calculate the Beta and discard the intercept
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    B = B[1:]

    # Component SD is Beta times the standard Deviation of the portfolio
    cSD = B * np.std(pReturn)
    Attribution_risk = pd.DataFrame({"Stock": ["Vol Attribution"]})
    for s in data.columns:
        # Attribution Risk (total portfolio return if we are updating the portfolio column)
        if s == 'Portfolio':
            vol = np.std(pReturn)
        else:
            vol = cSD[data.columns.to_list().index(s)]
        # Set the values
        Attribution_risk[s] = [vol]
    
    # combine both Attribution dataframes
    Attribution = pd.concat([Attribution_return,Attribution_risk], ignore_index=True)
    return Attribution

# 考虑factor component也考虑residual value(注意betas是m*n格式的矩阵，其中n是变量的个数，m是asset的个数）
# 具体看https://github.com/EdwardHFR/FinTech-545-Spring2023/blob/main/Week08/week08_problem2.jl

# Betas求法：
# stocks = [:AAPL, :MSFT, Symbol("BRK-B"), :CSCO, :JNJ]
# to_reg = innerjoin(returns[!,vcat(:Date, :SPY, stocks)], ffData, on=:Date)
# xnames = [:Mkt_RF, :SMB, :HML, :Mom]
# #OLS Regression for all Stocks
# X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))    **这一行怎么改可以参考922行
# Y = Matrix(to_reg[!,stocks])
# Betas = (inv(X'*X)*X'*Y)'[:,2:size(xnames,1)+1]
def expost_factor(w, upReturns, upFfData, Betas):
    stocks = upReturns.columns
    factors = upFfData.columns

    n = len(upReturns)
    m = len(stocks)

    pReturn = np.empty(n)
    residReturn = np.empty(n)
    weights = np.empty((n, len(w)))
    factorWeights = np.empty((n, len(factors)))
    lastW = w
    matReturns = upReturns[stocks].to_numpy()
    ffReturns = upFfData[factors].to_numpy()

    for i in range(n):
        # Save Current Weights in Matrix
        weights[i, :] = lastW

        # Factor Weight
        factorWeights[i, :] = np.sum(Betas * lastW, axis=0)

        # Update Weights by return
        lastW = lastW * (1.0 + matReturns[i, :])

        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastW)
        # Normalize the weights back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1

        # Residual
        residReturn[i] = (pR - 1) - factorWeights[i, :].dot(ffReturns[i, :])

    # Set the portfolio return in the Update Return DataFrame
    upFfData['Alpha'] = residReturn
    upFfData['Portfolio'] = pReturn

    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    # Calculate the Carino K
    k = np.log(totalRet + 1) / totalRet

    # Carino k_t is the ratio scaled by 1/K
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(ffReturns * factorWeights * carinoK.reshape(-1, 1), columns=factors)
    attrib['Alpha'] = residReturn * carinoK

    # Set up a DataFrame for output
    Attribution = pd.DataFrame({"Stock": ["TotalReturn", "Return Attribution"]})

    newFactors = factors.to_list() + ['Alpha']
    # Loop over the factors
    for s in newFactors + ['Portfolio']:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(upFfData[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        if s == 'Portfolio':
            atr = tr 
        else:
            atr = attrib[s].sum()
        # Set the values
        Attribution[s] = [tr, atr]

    # Realized Volatility Attribution
    # Y is our stock returns scaled by their weight at each time
    Y = np.hstack((ffReturns * factorWeights, residReturn.reshape(-1, 1)))
    # Set up X with the Portfolio Return
    X = np.column_stack((np.ones_like(pReturn), pReturn))
    # Calculate the Beta and discard the intercept
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    B = B[1, :]
    # Component SD is Beta times the standard deviation of the portfolio
    cSD = B * np.std(pReturn)

    # Check that the sum of component SD is equal to the portfolio SD
    assert np.isclose(np.sum(cSD), np.std(pReturn), rtol=1e-05, atol=1e-08)

    # Add the Vol attribution to the output
    vol_attrib = pd.DataFrame({"Stock": "Vol Attribution"})
    for i, factor in enumerate(newFactors):
        vol_attrib[factor] = [cSD[i]]
    vol_attrib['Portfolio'] = [np.std(pReturn)]
    Attribution = pd.concat([Attribution, vol_attrib], ignore_index=True)

    return Attribution

# Risk Budgeting
def risk_budget(w,covar):
    pSig = np.sqrt(w.T @ covar @ w)
    CSD = (w * (covar @ w)) / pSig
    return pd.DataFrame((CSD).T)

# Risk Budgeting with Risk Parity（注意covar的格式是columns的index是stock的名称）
def risk_budget_parity(covar,B=None):
    # Function for Portfolio Volatility
    def pvol(w, covar):
        return np.sqrt(np.dot(w.T, np.dot(covar, w)))

    # Function for Component Standard Deviation
    def pCSD(w, covar):
        pVol = pvol(w, covar)
        csd = w * (covar @ w) / pVol
        return csd

    # Sum Square Error of cSD
    def sseCSD(w, covar,B=None):
        if B == None:
            csd = pCSD(w, covar)
        else:
            csd = pCSD(w, covar) / B
        mCSD = sum(csd) / n
        dCsd = csd - mCSD
        se = dCsd ** 2
        return 1.0e5 * sum(se)  # Add a large multiplier for better convergence

    n = len(covar.columns)

    # Weights with boundary at 0
    w0 = np.ones(n) / n
    bounds = [(0, None)] * n

    res = minimize(sseCSD, w0, args=covar, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
    riskBudget = pd.DataFrame({'Stock': covar.columns, 'w': res.x,'RiskBudget': [risk_budget(res.x,covar)[0][i] for i in range(len(covar.columns))],'σ': np.sqrt(np.diag(covar))})
    return riskBudget

# Nonnormal Risk Parity (for returns that are not normally distributed)
def nonnormal_risk_parity(simReturn):
    def _ES(w, simReturn):
        r = simReturn @ w
        VaR, ES = cal_ES(r,alpha=0.05)
        return ES

    # Function for the component ES
    def CES(w, simReturn):
        n = len(w)
        ces = np.zeros(n)
        es = _ES(w, simReturn)
        e = 1e-6
        for i in range(n):
            old = w[i]
            w[i] = w[i]+e
            ces[i] = old*(_ES(w, simReturn) - es)/e
            w[i] = old
        return ces

    # SSE of the Component ES
    def SSE_CES(w, simReturn):
        ces = CES(w, simReturn)
        ces = ces - np.mean(ces)
        return 1e3 * (np.transpose(ces) @ ces)
    
    n = len(simReturn[0])
    w0 = np.ones(n) / n
    bnds = [(0, None)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    res = minimize(SSE_CES, w0, args=simReturn, method='SLSQP', bounds=bnds, constraints=cons)
    w = res.x
    ES_RPWeights = pd.DataFrame({'Stock': simReturn.columns, 'Weight': w, 'CES': CES(w,simReturn)})
    return w
#####Risk & Return Attribution#####END
    


#####Graphing#####START
###before using these functions, remember to first "plt.figure()" and eventually "plt.show()"
# define a function to graph ACF
def acf(data):
    stmplot.plot_acf(data)
    

# define a function to graph PACF
def acf(data):
    stmplot.plot_pacf(data)

# define a function to plot given data's distribution in the form of curve
def plot_dist_curve(data):
    sns.kdeplot(data, color="b", label=None)

# define a function to plot given data's distribution in the form of histgram
def plot_dist_hist(data):
    # plot original data
    sns.displot(data, stat='density', palette=('Greys'), label=None)
    
# define a function to plot a vertical line intersecting with x axis
def add_vertical_line(value):
    plt.axvline(x=value, color='b', label=None)
#####Graphing#####END