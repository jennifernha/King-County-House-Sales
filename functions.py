import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def metrics(y_train, train_preds, y_test, test_preds):
    '''
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



