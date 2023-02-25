import pandas as pd
from scipy.stats import norm, t
# to execute the following step, please execute "pip install riskfolio-lib" first
import riskfolio as rf
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# define the function for expected shortfall
def cal_ES(x,alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    ES = xs[0:idn].mean()
    return -ES

data = pd.read_csv("problem1.csv")
# fit the data into a normal distribution
norm_result = norm.fit(data)
mu = norm_result[0]
sigma = norm_result[1]

# fit the data into a student t distribution
t_result = t.fit(data)
df = t_result[0]
loc = t_result[1]
scale = t_result[2]

# set up alpha
alpha = 0.05
n = 10000

# calculate VaR for each distribution separately
VaR_05_norm = -norm.ppf(alpha, loc=mu, scale=sigma)
VaR_05_t = -t.ppf(alpha, df=df, loc=loc, scale=scale)

# calculate ES for each distribution separately
sim_norm = np.random.normal(loc=mu, scale=sigma, size=n)
ES_05_norm = cal_ES(sim_norm)
sim_t = t.rvs(df, loc=loc, scale=scale, size=n, random_state=None)
ES_05_t = cal_ES(sim_t)

# call packaged functions to compute VaR and ES to check answers
VaR_05_pack = rf.RiskFunctions.VaR_Hist(data, alpha=0.05)
ES_05_pack = rf.RiskFunctions.CVaR_Hist(data, alpha=0.05)

# print out results
print("VaR_05_norm {:.4f} vs. VaR_05_t {:.4f} vs. VaR_05_pack {:.4f}".format(VaR_05_norm, VaR_05_t, VaR_05_pack))
print("ES_05_norm  {:.4f} vs. ES_05_t {:.4f} vs. ES_05_pack {:.4f}".format(ES_05_norm, ES_05_t, ES_05_pack))

# plot distributions and ES and VaR
plt.figure()
# plot original data
sns.displot(data, stat='density', palette=('Greys'), label='Original Data')
# plot simulation
sns.kdeplot(sim_norm, color="b", label='Normal')
sns.kdeplot(sim_t, color="r", label='T')
# plot ES and VaR onto the graph
plt.axvline(x=-VaR_05_norm, color='b', label='VaR_05_Normal')
plt.axvline(x=-VaR_05_t, color='r', label='VaR_05_T')
plt.axvline(x=-ES_05_norm, color='b', label='ES_05_Normal', linestyle="dashed")
plt.axvline(x=-ES_05_t, color='r', label='ES_05_T', linestyle="dashed")
plt.legend()
plt.show()