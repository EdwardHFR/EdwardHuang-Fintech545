import numpy as np
from numpy.linalg import eigh
import sys
import itertools
import time
import math

# generate a N * N non-psd correlation matrix
N = 1000
n_psd = np.full((N, N), 0.9)
for i in range(N):
    n_psd[i][i] = 1.0
n_psd[0][1] = 0.7357
n_psd[1][0] = 0.7357

# define a function following Cholesky algorithm
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

# define a function to calculate Frobenius Norm
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


# implement near_PSD()
# execute the above function, and print execution time used
st = time.time()
n = near_PSD(n_psd)
et = time.time()
exe_time = et - st
print("Execution time:", exe_time, "seconds")

# Check eigenvalues of the resulting matrix
nval, nvec = eigh(n)
nval

# compute the Fabenious norm, as defined below, of the difference between resulting matrix n and original matrix n_psd
F_Norm(n - n_psd)



# implement Higham()
# execute the above function, and print execution time used
st = time.time()
h = Higham(n_psd)
et = time.time()
exe_time = et - st
print("Execution time:", exe_time, "seconds")

# Check eigenvalues of the resulting matrix
hval, hvec = eigh(h)
hval

# compute the Fabenious norm, as defined below, of the difference between resulting matrix n and original matrix n_psd
F_Norm(h - n_psd)