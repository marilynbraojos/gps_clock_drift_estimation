from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def r2(yhat, y): 
    r2_value = r2_score(yhat, y)
    return r2_value 

def mae(yhat, y): 
    mae_value = mean_absolute_error(yhat, y)
    return mae_value