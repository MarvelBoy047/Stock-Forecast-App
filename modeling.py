from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import sklearn.metrics as sk_metrics


def fit_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def fit_random_forest(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def fit_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

def fit_lstm(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse_train = sk_metrics.mean_squared_error(y_train, y_train_pred)
    mse_test = sk_metrics.mean_squared_error(y_test, y_test_pred)
    mae_train = sk_metrics.mean_absolute_error(y_train, y_train_pred)
    mae_test = sk_metrics.mean_absolute_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = sk_metrics.r2_score(y_train, y_train_pred)
    r2_test = sk_metrics.r2_score(y_test, y_test_pred)
    n = len(y_train)
    p = 1 # Number of features
    adj_r2_train = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)
    n = len(y_test)
    adj_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

    return mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test, adj_r2_train, adj_r2_test, y_train_pred, y_test_pred
def get_X_y(data):
    X = data['Date'].map(lambda x: x.timestamp()).values.reshape(-1, 1) # Convert date to numeric value
    y = data['Close'].values
    return X, y

def split_data(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, y_train, X_test, y_test
