import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_process as stmarma
import statsmodels.graphics.tsaplots as stmplot

# stimulate AR(1) process, and draw corresponding ACF and PACF graphs
ar1 = np.array([1, -0.9])
ma0 = np.array([1])
AR1_stimulated_pro = stmarma.ArmaProcess(ar1, ma0)
simulated_data_AR1 = AR1_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_AR1)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_AR1)
plt.show()

# stimulate AR(2) process, and draw corresponding ACF and PACF graphs
ar2 = np.array([1, -0.9, 0.5])
ma0 = np.array([1])
AR2_stimulated_pro = stmarma.ArmaProcess(ar2, ma0)
simulated_data_AR2 = AR2_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_AR2)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_AR2)
plt.show()

# stimulate AR(3) process, and draw corresponding ACF and PACF graphs
ar3 = np.array([1, -0.9, 0.5, 0.1])
ma0 = np.array([1])
AR3_stimulated_pro = stmarma.ArmaProcess(ar3, ma0)
simulated_data_AR3 = AR3_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_AR3)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_AR3)
plt.show()

# stimulate MA(1) process, and draw corresponding ACF and PACF graphs
ar0 = np.array([1])
ma1 = np.array([1, -0.5])
MA1_stimulated_pro = stmarma.ArmaProcess(ar0, ma1)
simulated_data_MA1 = MA1_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_MA1)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_MA1)
plt.show()

# stimulate MA(2) process, and draw corresponding ACF and PACF graphs
ar0 = np.array([1])
ma2 = np.array([1, -0.5, 0.5])
MA2_stimulated_pro = stmarma.ArmaProcess(ar0, ma2)
simulated_data_MA2 = MA2_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_MA2)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_MA2)
plt.show()

# stimulate MA(3) process, and draw corresponding ACF and PACF graphs
ar0 = np.array([1])
ma3 = np.array([1, -0.5, 0.5, -0.3])
MA3_stimulated_pro = stmarma.ArmaProcess(ar0, ma3)
simulated_data_MA3 = MA3_stimulated_pro.generate_sample(nsample=1000)
# ACF
stmplot.plot_acf(simulated_data_MA3)
plt.show()
# PACF
stmplot.plot_pacf(simulated_data_MA3)
plt.show()