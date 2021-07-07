import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression,Ridge
import pylab
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from matplotlib.gridspec import GridSpec

import scipy.stats as stats
from statsmodels.formula.api import ols

def linear_model(X_train, y_train, X_test, y_test):
    """
    Function to build linear regression model
    Step 1: Scaling data
    Step 2: Initializing regression class
    Step 3: fitting the model
    Step 4: predicting values
    Step 5: Finding coefficients
    Step 6: Finding Intercepts
    Step 7: Calculating RMSE and R2 value
    Step 8: Plotting regression graph
    Step 9: Creating Q-Q Plot
    Step 10: Creating homoscedasticity plot
    """
#   Scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
#   Create model 
    model = LinearRegression()
#   Fit the train data in the model
    model.fit(X_train_scaled, y_train)
#   Predict the train and test data
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
#   Find coefficients
    coeff_df = pd.DataFrame(model.coef_,X_train.columns, columns =['Coefficient'])
    print(coeff_df)
    print("-"*10)
#   Find intercept
    print('Intercept: ' + str(model.intercept_))
    print("-"*10)
#   Find RMSE value for the train and test data
    train_rmse = np.sqrt(mean_squared_error(y_train, (train_pred)))
    test_rmse = np.sqrt(mean_squared_error(y_test, (test_pred)))
    print("Train RMSE: "+ str(train_rmse))
    print("Test RMSE: "+ str(test_rmse))
#   Find the R2 value for the train and test data  
    train_r_squared = r2_score(y_train, train_pred)
    test_r_squared = r2_score(y_test, test_pred)
    print("Train R2: "+ str(train_r_squared))
    print("Test R2: "+ str(test_r_squared))
    sns.regplot(x=  test_pred, y = y_test, scatter_kws={"s":10}, line_kws={"color": "green"})
    plt.xlabel("Prediction_Price")
    plt.ylabel("Actual_Price")
    plt.title("Model Regression Plot")
#   Create QQ Plot
    sm.qqplot(train_pred, line = 's')
    pylab.show()
#  Create homoscedasticity plot
#  Code from https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
    model = Ridge()
    visualizer = ResidualsPlot(model)
    visualizer.fit(X_train, y_train)  
    visualizer.score(X_test, y_test) 
    visualizer.show()






