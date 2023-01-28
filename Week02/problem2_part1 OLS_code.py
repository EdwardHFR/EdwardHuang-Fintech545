import pandas as pd
import statsmodels.tools.tools as stmtools
import statsmodels.regression.linear_model as stmLRmodel
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro

# read data from provided file
data = pd.read_csv("problem2.csv")

# extract X and Y from the data
X = data.loc[:,"x"]
Y = data.loc[:,"y"]

# add constant term to X
X = stmtools.add_constant(X)

# construct the model and run OLS over it
model = stmLRmodel.OLS(Y,X)
result = model.fit()

# get residuals of the model
residuals = result.resid

# run Lilliefors test and Shapiro Wilk test over residuals to test normality
L_p_value = lilliefors(residuals)
SW_p_value = shapiro(residuals)

# draw a histogram for residuals to see whether the distribution looks normal or not
residuals.hist()