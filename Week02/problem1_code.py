from numpy.random import normal
import pandas as pd
import statsmodels.stats.descriptivestats as stm
import numpy as np
import scipy.stats
import math

# init a list for recording skewness and kurtosis later
skewness = []
kurtosis = []

# init a for loop for sampling
for i in range(100):
    # generate a normal distribution
    data_array = normal(loc=0, scale=1, size=100000)

    # transform the numpy array into a dataframe
    df = pd.DataFrame(data=data_array, columns=["random sample data"])

    # compute skewness and kurtosis
    descriptive_df = stm.describe(df)
    skewness.append(descriptive_df.loc["skew"])
    kurtosis.append(descriptive_df.loc["kurtosis"])

# get mean and variance of sampled skewness and kurtosis
skewness_array = np.array(skewness)
kurtosis_array = np.array(kurtosis)

skewness_df = pd.DataFrame(data=skewness_array, columns=["skewness data"])
kurtosis_df = pd.DataFrame(data=kurtosis_array, columns=["kurtosis data"])

descriptive_skewness_df = stm.describe(skewness_df)
s_mean = descriptive_skewness_df.loc["mean"]
s_variance = pow(descriptive_skewness_df.loc["std"], 2)
s_num_obs = descriptive_skewness_df.loc["nobs"]

descriptive_kurtosis_df = stm.describe(kurtosis_df)
k_mean = descriptive_kurtosis_df.loc["mean"]
k_variance = pow(descriptive_kurtosis_df.loc["std"], 2)
k_num_obs = descriptive_kurtosis_df.loc["nobs"]

# compute t-stats and corresponding two-way p-value with scipy, and print out the p-value
s_t_stats = s_mean / math.sqrt(s_variance/s_num_obs)
s_p_value = scipy.stats.t.sf(abs(s_t_stats), df=(s_num_obs - 1))*2
print("skewness p-value:",s_p_value)

k_t_stats = k_mean / math.sqrt(k_variance/k_num_obs)
k_p_value = scipy.stats.t.sf(abs(k_t_stats), df=(k_num_obs - 1))*2
print("kurtosis p-value:",k_p_value)
math.isclose(k_p_value, 0, rel_tol=1e-06)
