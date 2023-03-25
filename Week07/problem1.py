import functionlib
from datetime import datetime
import numpy as np

##### PART 1
# set up the problem
underlying = 165
strike = 165
current_date = datetime(2022,3,13)
expiration = datetime(2022,4,15)
ttm = (expiration - current_date).days / 365
rf = 0.0425
coupon = 0.0053
b = rf - coupon
ivol = 0.2

### for a call
print("For a call option:\n")
# calculation of delta
delta_call_closed = functionlib.delta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")
differential_equation_delta_call = functionlib.partial_derivative(functionlib.gbsm, "underlying", delta=1e-3, order=1)
delta_call_diff = differential_equation_delta_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Delta using closed-form formula: {:.4f}".format(delta_call_closed))
print("Delta using a finite difference derivative function: {:.4f}".format(delta_call_diff))

# calculation of gamma
gamma_call_closed = functionlib.gamma_gbsm(underlying, strike, ttm, rf, b, ivol)
differential_equation_gamma_call = functionlib.partial_derivative(functionlib.gbsm, "underlying", delta=1e-3, order=2)
gamma_call_diff = differential_equation_gamma_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Gamma using closed-form formula: {:.4f}".format(gamma_call_closed))
print("Gamma using a finite difference derivative function: {:.4f}".format(gamma_call_diff))

# calculation of vega
vega_call_closed = functionlib.vega_gbsm(underlying, strike, ttm, rf, b, ivol)
differential_equation_vega_call = functionlib.partial_derivative(functionlib.gbsm, "ivol", delta=1e-3, order=1)
vega_call_diff = differential_equation_vega_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Vega using closed-form formula: {:.4f}".format(vega_call_closed))
print("Vega using a finite difference derivative function: {:.4f}".format(vega_call_diff))

# calculation of theta
theta_call_closed = functionlib.theta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")
differential_equation_theta_call = functionlib.partial_derivative(functionlib.gbsm, "ttm", delta=1e-3, order=1)
theta_call_diff = -differential_equation_theta_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Theta using closed-form formula: {:.4f}".format(theta_call_closed))
print("Theta using a finite difference derivative function: {:.4f}".format(theta_call_diff))

# calculation of rho
rho_call_closed = functionlib.rho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")
differential_equation_rho_call = functionlib.partial_derivative(functionlib.gbsm, "rf", delta=1e-3, order=1)
rho_call_diff = differential_equation_rho_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Rho using closed-form formula: {:.4f}".format(rho_call_closed))
print("Rho using a finite difference derivative function: {:.4f}".format(rho_call_diff))

# calculation of carry rho
crho_call_closed = functionlib.crho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call")
differential_equation_crho_call = functionlib.partial_derivative(functionlib.gbsm, "b", delta=1e-3, order=1)
crho_call_diff = differential_equation_crho_call(underlying, strike, ttm, rf, b, ivol, type="call")
print("Carry Rho using closed-form formula: {:.4f}".format(crho_call_closed))
print("Carry Rho using a finite difference derivative function: {:.4f}".format(crho_call_diff))
print("\n\n\n")



### for a put
print("For a put option:\n")
# calculation of delta
delta_put_closed = functionlib.delta_gbsm(underlying, strike, ttm, rf, b, ivol, type="put")
differential_equation_delta_put = functionlib.partial_derivative(functionlib.gbsm, "underlying", delta=1e-3, order=1)
delta_put_diff = differential_equation_delta_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Delta using closed-form formula: {:.4f}".format(delta_put_closed))
print("Delta using a finite difference derivative function: {:.4f}".format(delta_put_diff))

# calculation of gamma
gamma_put_closed = functionlib.gamma_gbsm(underlying, strike, ttm, rf, b, ivol)
differential_equation_gamma_put = functionlib.partial_derivative(functionlib.gbsm, "underlying", delta=1e-3, order=2)
gamma_put_diff = differential_equation_gamma_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Gamma using closed-form formula: {:.4f}".format(gamma_put_closed))
print("Gamma using a finite difference derivative function: {:.4f}".format(gamma_put_diff))

# calculation of vega
vega_put_closed = functionlib.vega_gbsm(underlying, strike, ttm, rf, b, ivol)
differential_equation_vega_put = functionlib.partial_derivative(functionlib.gbsm, "ivol", delta=1e-3, order=1)
vega_put_diff = differential_equation_vega_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Vega using closed-form formula: {:.4f}".format(vega_put_closed))
print("Vega using a finite difference derivative function: {:.4f}".format(vega_put_diff))

# calculation of theta
theta_put_closed = functionlib.theta_gbsm(underlying, strike, ttm, rf, b, ivol, type="put")
differential_equation_theta_put = functionlib.partial_derivative(functionlib.gbsm, "ttm", delta=1e-3, order=1)
theta_put_diff = -differential_equation_theta_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Theta using closed-form formula: {:.4f}".format(theta_put_closed))
print("Theta using a finite difference derivative function: {:.4f}".format(theta_put_diff))

# calculation of rho
rho_put_closed = functionlib.rho_gbsm(underlying, strike, ttm, rf, b, ivol, type="put")
differential_equation_rho_put = functionlib.partial_derivative(functionlib.gbsm, "rf", delta=1e-3, order=1)
rho_put_diff = differential_equation_rho_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Rho using closed-form formula: {:.4f}".format(rho_put_closed))
print("Rho using a finite difference derivative function: {:.4f}".format(rho_put_diff))

# calculation of carry rho
crho_put_closed = functionlib.crho_gbsm(underlying, strike, ttm, rf, b, ivol, type="put")
differential_equation_crho_put = functionlib.partial_derivative(functionlib.gbsm, "b", delta=1e-3, order=1)
crho_put_diff = differential_equation_crho_put(underlying, strike, ttm, rf, b, ivol, type="put")
print("Carry Rho using closed-form formula: {:.4f}".format(crho_put_closed))
print("Carry Rho using a finite difference derivative function: {:.4f}".format(crho_put_diff))
print("\n\n\n")



##### PART 2
# set up the problem (assume that there are 500 steps)
N = 500
div_date = datetime(2022,4,11)
divAmts = [0.88]
divTimes = [int((div_date - current_date).days / (expiration - current_date).days * N)]

# price the American call without dividends, and compute correpsonding greeks
print("For American call option without dividends:\n")
american_call_nodiv = functionlib.bt_american(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Value of American call option without dividends: {:.4f}".format(american_call_nodiv))
differential_equation_delta_a_call = functionlib.partial_derivative(functionlib.bt_american, "underlying", delta=1e-3, order=1)
delta_a_call_diff = differential_equation_delta_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Delta of American call option without dividends: {:.4f}".format(delta_a_call_diff))
differential_equation_gamma_a_call = functionlib.partial_derivative(functionlib.bt_american, "underlying", delta=1e-3, order=2)
gamma_a_call_diff = differential_equation_gamma_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Gamma of American call option without dividends: {:.4f}".format(gamma_a_call_diff))
differential_equation_vega_a_call = functionlib.partial_derivative(functionlib.bt_american, "ivol", delta=1e-3, order=1)
vega_a_call_diff = differential_equation_vega_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Vega of American call option without dividends: {:.4f}".format(vega_a_call_diff))
differential_equation_theta_a_call = functionlib.partial_derivative(functionlib.bt_american, "ttm", delta=1e-3, order=1)
theta_a_call_diff = -differential_equation_theta_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Theta of American call option without dividends: {:.4f}".format(theta_a_call_diff))
differential_equation_rho_a_call = functionlib.partial_derivative(functionlib.bt_american, "rf", delta=1e-3, order=1)
rho_a_call_diff = differential_equation_rho_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Rho of American call option without dividends: {:.4f}".format(rho_a_call_diff))
differential_equation_crho_a_call = functionlib.partial_derivative(functionlib.bt_american, "b", delta=1e-3, order=1)
crho_a_call_diff = differential_equation_crho_a_call(underlying, strike, ttm, rf, b, ivol, N, otype="call")
print("Carry Rho of American call option without dividends: {:.4f}\n\n\n".format(crho_a_call_diff))

# price the American put without dividends, and compute corresponding greeks
print("For American put option without dividends:\n")
american_put_nodiv = functionlib.bt_american(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Value of American put option without dividends: {:.4f}".format(american_put_nodiv))
differential_equation_delta_a_put = functionlib.partial_derivative(functionlib.bt_american, "underlying", delta=1e-3, order=1)
delta_a_put_diff = differential_equation_delta_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Delta of American put option without dividends: {:.4f}".format(delta_a_put_diff))
differential_equation_gamma_a_put = functionlib.partial_derivative(functionlib.bt_american, "underlying", delta=1e-3, order=2)
gamma_a_put_diff = differential_equation_gamma_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Gamma of American put option without dividends: {:.4f}".format(gamma_a_put_diff))
differential_equation_vega_a_put = functionlib.partial_derivative(functionlib.bt_american, "ivol", delta=1e-3, order=1)
vega_a_put_diff = differential_equation_vega_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Vega of American put option without dividends: {:.4f}".format(vega_a_put_diff))
differential_equation_theta_a_put = functionlib.partial_derivative(functionlib.bt_american, "ttm", delta=1e-3, order=1)
theta_a_put_diff = -differential_equation_theta_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Theta of American put option without dividends: {:.4f}".format(theta_a_put_diff))
differential_equation_rho_a_put = functionlib.partial_derivative(functionlib.bt_american, "rf", delta=1e-3, order=1)
rho_a_put_diff = differential_equation_rho_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Rho of American put option without dividends: {:.4f}".format(rho_a_put_diff))
differential_equation_crho_a_put = functionlib.partial_derivative(functionlib.bt_american, "b", delta=1e-3, order=1)
crho_a_put_diff = differential_equation_crho_a_put(underlying, strike, ttm, rf, b, ivol, N, otype="put")
print("Carry Rho of American put option without dividends: {:.4f}\n\n\n".format(crho_a_put_diff))

# price the American call with dividends, and compute correpsonding greeks
print("For American call option with dividends:\n")
american_call_div = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Value of American call option with dividends: {:.4f}".format(american_call_div))
differential_equation_delta_a_call_div = functionlib.partial_derivative(functionlib.bt_american_div, "underlying", delta=1e-3, order=1)
delta_a_call_diff_div = differential_equation_delta_a_call_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Delta of American call option with dividends: {:.4f}".format(delta_a_call_diff_div))
differential_equation_gamma_a_call_div = functionlib.partial_derivative(functionlib.bt_american_div, "underlying", delta=1e-3, order=2)
gamma_a_call_diff_div = differential_equation_gamma_a_call_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Gamma of American call option with dividends: {:.4f}".format(gamma_a_call_diff_div))
differential_equation_vega_a_call_div = functionlib.partial_derivative(functionlib.bt_american_div, "ivol", delta=1e-3, order=1)
vega_a_call_diff_div = differential_equation_vega_a_call_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Vega of American call option with dividends: {:.4f}".format(vega_a_call_diff_div))
differential_equation_theta_a_call_div = functionlib.partial_derivative(functionlib.bt_american_div, "ttm", delta=1e-3, order=1)
theta_a_call_diff_div = -differential_equation_theta_a_call_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Theta of American call option with dividends: {:.4f}".format(theta_a_call_diff))
differential_equation_rho_a_call_div = functionlib.partial_derivative(functionlib.bt_american_div, "rf", delta=1e-3, order=1)
rho_a_call_diff_div = differential_equation_rho_a_call_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call")
print("Rho of American call option with dividends: {:.4f}".format(rho_a_call_diff_div))
print("Carry Rho of American call option with dividends: not applicable here\n\n\n")

# price the American put with dividends, and compute correpsonding greeks
print("For American put option with dividends:\n")
american_put_div = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Value of American put option with dividends: {:.4f}".format(american_put_div))
differential_equation_delta_a_put_div = functionlib.partial_derivative(functionlib.bt_american_div, "underlying", delta=1e-3, order=1)
delta_a_put_diff_div = differential_equation_delta_a_put_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Delta of American put option with dividends: {:.4f}".format(delta_a_put_diff_div))
differential_equation_gamma_a_put_div = functionlib.partial_derivative(functionlib.bt_american_div, "underlying", delta=1e-3, order=2)
gamma_a_put_diff_div = differential_equation_gamma_a_put_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Gamma of American put option with dividends: {:.4f}".format(gamma_a_put_diff_div))
differential_equation_vega_a_put_div = functionlib.partial_derivative(functionlib.bt_american_div, "ivol", delta=1e-3, order=1)
vega_a_put_diff_div = differential_equation_vega_a_put_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Vega of American put option with dividends: {:.4f}".format(vega_a_put_diff_div))
differential_equation_theta_a_put_div = functionlib.partial_derivative(functionlib.bt_american_div, "ttm", delta=1e-3, order=1)
theta_a_put_diff_div = -differential_equation_theta_a_put_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Theta of American put option with dividends: {:.4f}".format(theta_a_put_diff))
differential_equation_rho_a_put_div = functionlib.partial_derivative(functionlib.bt_american_div, "rf", delta=1e-3, order=1)
rho_a_put_diff_div = differential_equation_rho_a_put_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="put")
print("Rho of American put option with dividends: {:.4f}".format(rho_a_put_diff_div))
print("Carry Rho of American put option with dividends: not applicable here\n\n\n")

# calculate sensitivity of the put and call to dividends
div_delta = 1e-3
divAmts = np.array(divAmts)
call_value1 = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts+div_delta, divTimes, ivol, N, type="call")    
call_value2 = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts-div_delta, divTimes, ivol, N, type="call")  
call_sens = (call_value1 - call_value2) / (2*div_delta)
print("Sensitivity of the American call option is {:.4f}".format(call_sens))

put_value1 = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts+div_delta, divTimes, ivol, N, type="put")    
put_value2 = functionlib.bt_american_div(underlying, strike, ttm, rf, divAmts-div_delta, divTimes, ivol, N, type="put")
put_sens = (put_value1 - put_value2) / (2*div_delta)
print("Sensitivity of the American put option is {:.4f}".format(put_sens))