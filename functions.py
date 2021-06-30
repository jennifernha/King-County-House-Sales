import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import seaborn as sns
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.eval_measures as ev
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from scipy.special import boxcox, inv_boxcox


import scipy.stats as stats
from statsmodels.formula.api import ols

def linear_model(X_train, y_train, X_test, y_test):
    """
    Function to build linear regression model
    Step 1: Scaling data
    Step 2: Initializing regression class
    Step 3: fitting the model
    Step 4: predicting values
    Step 5: Calculating RMSE and R2 value
    Step 6: Plotting regression graph
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
#   Finding RMSE value for the train and test data
    train_rmse = np.sqrt(mean_squared_error(y_train, (train_pred)))
    test_rmse = np.sqrt(mean_squared_error(y_test, (test_pred)))
    print("Train RMSE: "+ str(train_rmse))
    print("Test RMSE: "+ str(test_rmse))
#   Finding the R2 value for the train and test data  
    train_r_squared = r2_score(y_train, train_pred)
    test_r_squared = r2_score(y_test, test_pred)
    print("Train R2: "+ str(train_r_squared))
    print("Test R2: "+ str(test_r_squared))
    sns.regplot(x=  test_pred, y = y_test, scatter_kws={"s":10}, line_kws={"color": "green"})
    plt.xlabel("Prediction_Price")
    plt.ylabel("Actual_Price")
    plt.title("Model Regression Plot")




def model_result(y_train, train_preds, y_test, test_preds):
    '''
    This function was written by Lindsey Berlin during Office Hours.
    Prints the R2 Score, Mean Absolute Error and Root Mean Squared Error
    Will unlog to get MAE & RMSE in terms of the original target if log=True

    Inputs:
        y_train: array-like or pandas series
            Actual target values for the train set
        train_preds: array-like or pandas series
            Predicted target values for the train set
        y_test: array-like or pandas series
            Actual target values for the test set
        test_preds: array-like or pandas series
            Predicted target values for the test set
        log: boolean
            Toggles whether the target values have been logged or not
            If True, assumes all other arguments passed into the function have been logged

    Outputs:
        None, just prints the metrics 
    '''

    # Unlogging all variables if you set log=True
    if log == True:
        y_train_unlog = np.expm1(y_train)
        train_preds_unlog = np.expm1(train_preds)
        y_test_unlog = np.expm1(y_test)
        test_preds_unlog = np.expm1(test_preds)

    # Printing train scores
    print("Training Scores")
    print("-"*10)
    print(f'R2: {r2_score(y_train, train_preds): .4f}') #R2 should not be done on unlogged values
    if log == True:
        print(f'RMSE: {mean_squared_error(y_train_unlog, train_preds_unlog, squared = False): .4f}')
        print(f'MAE: {mean_absolute_error(y_train_unlog, train_preds_unlog): .4f}')

    else:
        print(f'RMSE: {mean_squared_error(y_train, train_preds, squared = False): .4f}')
        print(f'MAE: {mean_absolute_error(y_train, train_preds): .4f}')

    print('\n'+'*'*10)

    # Printing test scores
    print('Test Scores:')
    print("-"*10)
    print(f'R2: {r2_score(y_test, test_preds): .4f}') #R2 should not be done on unlogged values
    if log == True:
        print(f'RMSE: {mean_squared_error(y_test_unlog, test_preds_unlog, squared = False): .4f}')
        print(f'MAE: {mean_absolute_error(y_test_unlog, test_preds_unlog): .4f}')

    else:
        print(f'RMSE: {mean_squared_error(y_test, test_preds, squared = False): .4f}')
        print(f'MAE: {mean_absolute_error(y_test, test_preds): .4f}')



##################################################
# Create function to run linear regression model #
##################################################
def model_summary(df):
    '''
    Creates function to run linear regression regression model.
    Inputs:
        df: dataframe that is being used

    Outputs:
        OLS Regression Results, Q-Q Plot, and Homoscedasticity Scatter Plot
    '''
    ## Create a string representing the right side of the ~ in our formula
    independent = ' + '.join(df.drop('price',axis=1).columns)
    
    ## Create the final formula and create the model
    f  = "price~"+independent
    
    # Model regression
    model = smf.ols(f, df).fit()
    display(model.summary())
        # Create QQ plot
    fig, ax = plt.subplots(ncols=2,figsize=(14,6))
    sm.graphics.qqplot(model.resid,dist=stats.norm,fit=True,line='45',\
                       ax=ax[0])
    ax[0].set_title('QQ Plot')
    
    # Create homoscedasticity plot
    resids = model.resid
    sns.scatterplot(x=model.predict(df.drop('price',axis=1), transform=True),\
                    y=model.resid, ax=ax[1])
    ax[1].axhline(0, color='r')
    ax[1].set_title('Homoscedasticity of Residuals')
    ax[1].set_xlabel('Predicted Price')
    ax[1].set_ylabel('Residuals')
    
    
    return model, fig, ax
