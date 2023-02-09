import pandas as pd
import math
import numpy as np
from numpy.linalg import eigh
from matplotlib import pyplot as plt
import time

# read data from provided file
stock_data = pd.read_csv("DailyReturn.csv")

# get all dates in the file
dates = list(stock_data.iloc[:,0])

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
    stock_names.pop(0)
    
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
            sum = 0
            
            for z in dates:
                z_index = dates.index(z)
                
                part = weights_vector[z_index] * (i_data[z_index] - i_mean) * (j_data[z_index] - j_mean)
                sum += part
            
            # store the derived variance into the matrix
            w_cov_mat.loc[i,j] = sum
    
    return w_cov_mat

# modify the function defined above to calculate the exponentially weighted variance matrix
def exwVarMat(w_cov_mat):
    # the diagonal of exponentially weighted matrix is the expontially weighted variances
    w_var_mat = np.diag(w_cov_mat)
    
    return w_var_mat

# we are now using PCA to simulate the system through defining a new function
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
    
    return sum




# set up lists of weights, cumulative weights, and set up lamda
weights = []
cum_weights = []
lamda = 0.97

# call both defined functions to get the result
populateWeights(dates,weights, cum_weights, lamda)

# reverse w so that it corresponds to the ascending order of dates used in the DailyReturn.csv
rev_weights = weights[::-1]

### 
# derive the first covariance matrix - exponentially weighted covariance matrix
w_cov = exwCovMat(stock_data, rev_weights)

# run simulations
# direct simulation
st = time.time()
d_sim = direct_simulate(w_cov, 25000)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
d_sim_pd = pd.DataFrame(d_sim)
d_sim_cov = d_sim_pd.cov()
diff = d_sim_cov.to_numpy() - w_cov.to_numpy()
F_Norm(diff)

# PCA simulation with 100% explained
st = time.time()
P_100_sim = simulate_PCA(w_cov, 25000, percent_explained=1)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_100_sim_pd = pd.DataFrame(P_100_sim)
P_100_sim_cov = P_100_sim_pd.cov()
diff = P_100_sim_cov.to_numpy() - w_cov.to_numpy()
F_Norm(diff)

# PCA simulation with 75% explained
st = time.time()
P_75_sim = simulate_PCA(w_cov, 25000, percent_explained=0.75)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_75_sim_pd = pd.DataFrame(P_75_sim)
P_75_sim_cov = P_75_sim_pd.cov()
diff = P_75_sim_cov.to_numpy() - w_cov.to_numpy()
F_Norm(diff)

# PCA simulation with 50% explained
st = time.time()
P_50_sim = simulate_PCA(w_cov, 25000, percent_explained=0.5)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_50_sim_pd = pd.DataFrame(P_50_sim)
P_50_sim_cov = P_50_sim_pd.cov()
diff = P_50_sim_cov.to_numpy() - w_cov.to_numpy()
F_Norm(diff)
###




# together compute the exponentially variance matrix(vector) for later use
w_var = exwVarMat(w_cov)
w_std = np.sqrt(w_var)
# compute the exponentially weighted correlation matrix for later use
w_cor = np.zeros((len(w_cov),len(w_cov)))
for i in range(len(w_cov)):
    for j in range(len(w_cov)):
        w_cor[i][j] = w_cov.iloc[i,j] / (w_std[i] * w_std[j])
# compute the list of (normally computed) standard deviations for later use
std = stock_data.iloc[:,1:].std()
# compute (normally computed) correlation matrix for later use
cor = stock_data.corr()




###
# derive the second covariance matrix - exponentially weighted correlation matrix and (normally computed) standard deviations
second_cov = np.zeros((len(w_cor),len(w_cor)))
for i in range(len(w_cor)):
    for j in range(len(w_cor)):
        second_cov[i][j] = w_cor[i][j] * std[i] * std[j]

# run simulations
# direct simulation
st = time.time()
d_sim = direct_simulate(second_cov, 25000)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
d_sim_pd = pd.DataFrame(d_sim)
d_sim_cov = d_sim_pd.cov()
diff = d_sim_cov.to_numpy() - second_cov
F_Norm(diff)

# PCA simulation with 100% explained
st = time.time()
P_100_sim = simulate_PCA(second_cov, 25000, percent_explained=1)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_100_sim_pd = pd.DataFrame(P_100_sim)
P_100_sim_cov = P_100_sim_pd.cov()
diff = P_100_sim_cov.to_numpy() - second_cov
F_Norm(diff)

# PCA simulation with 75% explained
st = time.time()
P_75_sim = simulate_PCA(second_cov, 25000, percent_explained=0.75)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_75_sim_pd = pd.DataFrame(P_75_sim)
P_75_sim_cov = P_75_sim_pd.cov()
diff = P_75_sim_cov.to_numpy() - second_cov
F_Norm(diff)

# PCA simulation with 50% explained
st = time.time()
P_50_sim = simulate_PCA(second_cov, 25000, percent_explained=0.5)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_50_sim_pd = pd.DataFrame(P_50_sim)
P_50_sim_cov = P_50_sim_pd.cov()
diff = P_50_sim_cov.to_numpy() - second_cov
F_Norm(diff)
###

###
# derive the third covariance matrix - (normally computed) correlation matrix and exponentially weighted standard deviations
third_cov = np.zeros((len(cor),len(cor)))
for i in range(len(cor)):
    for j in range(len(cor)):
        third_cov[i][j] = cor.iloc[i,j] * w_std[i] * w_std[j]

# run simulations
# direct simulation
st = time.time()
d_sim = direct_simulate(third_cov, 25000)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
d_sim_pd = pd.DataFrame(d_sim)
d_sim_cov = d_sim_pd.cov()
diff = d_sim_cov.to_numpy() - third_cov
F_Norm(diff)

# PCA simulation with 100% explained
st = time.time()
P_100_sim = simulate_PCA(third_cov, 25000, percent_explained=1)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_100_sim_pd = pd.DataFrame(P_100_sim)
P_100_sim_cov = P_100_sim_pd.cov()
diff = P_100_sim_cov.to_numpy() - third_cov
F_Norm(diff)

# PCA simulation with 75% explained
st = time.time()
P_75_sim = simulate_PCA(third_cov, 25000, percent_explained=0.75)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_75_sim_pd = pd.DataFrame(P_75_sim)
P_75_sim_cov = P_75_sim_pd.cov()
diff = P_75_sim_cov.to_numpy() - third_cov
F_Norm(diff)

# PCA simulation with 50% explained
st = time.time()
P_50_sim = simulate_PCA(third_cov, 25000, percent_explained=0.5)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_50_sim_pd = pd.DataFrame(P_50_sim)
P_50_sim_cov = P_50_sim_pd.cov()
diff = P_50_sim_cov.to_numpy() - third_cov
F_Norm(diff)
###

###
# derive the fourth covariance matrix - (normally computed) correlation matrix and (normally computed) standard deviations
fourth_cov = np.zeros((len(cor),len(cor)))
for i in range(len(cor)):
    for j in range(len(cor)):
        fourth_cov[i][j] = cor.iloc[i,j] * std[i] * std[j]

# run simulations
# direct simulation
st = time.time()
d_sim = direct_simulate(fourth_cov, 25000)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
d_sim_pd = pd.DataFrame(d_sim)
d_sim_cov = d_sim_pd.cov()
diff = d_sim_cov.to_numpy() - fourth_cov
F_Norm(diff)

# PCA simulation with 100% explained
st = time.time()
P_100_sim = simulate_PCA(fourth_cov, 25000, percent_explained=1)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_100_sim_pd = pd.DataFrame(P_100_sim)
P_100_sim_cov = P_100_sim_pd.cov()
diff = P_100_sim_cov.to_numpy() - fourth_cov
F_Norm(diff)

# PCA simulation with 75% explained
st = time.time()
P_75_sim = simulate_PCA(fourth_cov, 25000, percent_explained=0.75)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_75_sim_pd = pd.DataFrame(P_75_sim)
P_75_sim_cov = P_75_sim_pd.cov()
diff = P_75_sim_cov.to_numpy() - fourth_cov
F_Norm(diff)

# PCA simulation with 50% explained
st = time.time()
P_50_sim = simulate_PCA(fourth_cov, 25000, percent_explained=0.5)
et = time.time()
sim_time = et - st
print("Execution time:", sim_time, "seconds")
P_50_sim_pd = pd.DataFrame(P_50_sim)
P_50_sim_cov = P_50_sim_pd.cov()
diff = P_50_sim_cov.to_numpy() - fourth_cov
F_Norm(diff)
###
