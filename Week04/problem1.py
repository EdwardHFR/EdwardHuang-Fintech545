import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# set up price of time 0 P(t-1) to be 100, and standard deviation sigma to be 0.75
p_previous = 100
sigma = 0.75

# simulate prices with returns for Classic Brownian Motion - P(t) = P(t-1) + r(t)
# simulate returns
sim_returns = np.random.normal(0,0.75,10000)

# simulate prices with 10000 times
sim_prices_cla = p_previous + sim_returns

# calculate mean and standard deviation of sim_prices
sim_mean = sim_prices_cla.mean()
sim_std = sim_prices_cla.std()

# theoretical mean and standard deviation of P(t) in Classic Brownian Motion
theory_mean = p_previous
theory_std = sigma



# simulate prices with returns for Arithmetic Return System - P(t) = P(t-1) * (1 + r(t))
# simulate returns
sim_returns = np.random.normal(0,0.75,10000)

# simulate prices with 10000 times
sim_prices_ari = p_previous * (1 + sim_returns)

# calculate mean and standard deviation of sim_prices
sim_mean = sim_prices_ari.mean()
sim_std = sim_prices_ari.std()

# theoretical mean and standard deviation of P(t) in Classic Brownian Motion
theory_mean = p_previous
theory_std = p_previous * sigma



# simulate prices with returns for Geometric Brownian Motion - P(t) = P(t-1) * exp(r(t))
# simulate returns
sim_returns = np.random.normal(0,0.75,10000)

# simulate prices with 10000 times
sim_prices_geo = p_previous * np.exp(sim_returns)

# calculate mean and standard deviation of sim_prices
sim_mean = sim_prices_geo.mean()
sim_std = sim_prices_geo.std()

# theoretical mean and standard deviation of P(t) in Classic Brownian Motion
theory_mean = p_previous * math.exp(0.5 * pow(sigma,2))
theory_std = p_previous * math.sqrt(math.exp(2 * pow(sigma,2)) - math.exp(pow(sigma,2)))




# graph all three simulated prices
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(3)
sns.distplot(sim_prices_cla, ax=axis[0])
sns.distplot(sim_prices_ari, ax=axis[1])
sns.distplot(sim_prices_geo, ax=axis[2])
  
# For Classic Brownian Motion
axis[0].set_title("Classic Brownian Motion")
  
# For Arithmetic Return System
axis[1].set_title("Arithmetic Return System")
  
# For Geometric Brownian Motion
axis[2].set_title("Geometric Brownian Motion")
  
# Combine all the operations and display
plt.show()