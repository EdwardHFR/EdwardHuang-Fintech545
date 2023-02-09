import pandas as pd
import math
import numpy as np
from numpy.linalg import eigh
from matplotlib import pyplot as plt

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

# set up lists of weights, cumulative weights, and set up lamda
weights = []
cum_weights = []
lamda = 0.94

# call both defined functions to get the result
populateWeights(dates,weights, cum_weights, lamda)

# reverse w so that it corresponds to the ascending order of dates used in the DailyReturn.csv
rev_weights = weights[::-1]
    
covariance_matrix = exwCovMat(stock_data, rev_weights)

# we are now using PCA - simply calculate eigenvalues here - to show how many percent of variance could be explained by first k eigenvalues
# calculate the eigenvalues and eigenvectors of derived matrix, and sort eigenvalues from largest to smallest
e_val, e_vec = eigh(covariance_matrix)
a_sorted_e_val = np.sort(e_val)
d_sorted_e_val = a_sorted_e_val[::-1]

# we assume all negative eigenvalues derived are zero, since they are effectively zero (larger than -1e-8)
assert np.amin(d_sorted_e_val) > -1e-8
d_sorted_e_val[d_sorted_e_val<0] = 0

# calculate the sum of all eigenvalues
e_sum = sum(d_sorted_e_val)

# remove all eigenvalues that too small - in this case, we set the threshold to be 1e-8
sub_d_sorted_e_val = d_sorted_e_val[d_sorted_e_val> 1e-8]

# calculate the percent of variance explained by the first k eigenvalues, and store them into a preset list
individual_percent = []
total_percent = []
sum_percent = 0
for i in range(len(sub_d_sorted_e_val)):
    each_percent = sub_d_sorted_e_val[i] / e_sum
    individual_percent.append(each_percent)
    sum_percent += each_percent
    total_percent.append(sum_percent)

# plot the cumulative variance explained by each eigenvalue; note: when there are a lot of eigenvalues chosen, do not implement "plt.xticks(range(len(sub_d_sorted_e_val)))" so that the x-axis of the graph would not be crowded
plt.bar(range(len(sub_d_sorted_e_val)), individual_percent)
plt.xlabel("Index of Each Eigenvalue")
plt.ylabel("Percent of Variance Explained")
plt.xticks(range(len(sub_d_sorted_e_val)))

# plot the cumulative variance explained by first k eigenvalues; note: when there are a lot of eigenvalues chosen, do not implement "plt.xticks(range(len(sub_d_sorted_e_val)))" so that the x-axis of the graph would not be crowded
plt.plot(range(1,(len(sub_d_sorted_e_val)+1)),total_percent)
plt.xlabel("First K Eigenvalues Chosen")
plt.ylabel("Percent of Variance Explained")
plt.xticks(range(1,(len(sub_d_sorted_e_val)+1)))