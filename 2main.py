import streamlit as st
from datetime import date
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title('Stock Forecast App')

# Sidebar for user options
st.sidebar.subheader('Select Options')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
models = ('Linear Regression', 'Random Forest', 'Gradient Boosting', 'LSTM')
selected_model = st.sidebar.selectbox('Select model for prediction', models)
evaluation_metrics = ('MSE', 'MAE', 'RMSE', 'R2', 'Adj. R2')
selected_metric = st.sidebar.selectbox('Select metric for evaluation', evaluation_metrics)
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365
epochs = st.sidebar.slider('Epochs for LSTM model:', 1, 50, 10)  # Adjustable epochs for LSTM model training

# Function to load data from Yahoo Finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to extract features and target
def get_X_y(data):
    X = data['Date'].map(lambda x: x.timestamp()).values.reshape(-1, 1)  # Convert date to numeric value
    y = data['Close'].values
    return X, y

# Function to split data into train and test sets
def split_data(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, y_train, X_test, y_test

# Function to fit Linear Regression model
def fit_linear_regression(X_train, y_train):
    X_train = X_train.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to fit Random Forest model
def fit_random_forest(X_train, y_train):
    n_estimators = st.sidebar.slider('Number of Trees:', 1, 100, 10)
    max_depth = st.sidebar.slider('Maximum Depth:', 1, 20, 5)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

# Function to fit Gradient Boosting model
def fit_gradient_boosting(X_train, y_train):
    learning_rate = st.sidebar.slider('Learning Rate:', 0.01, 1.0, 0.1)
    n_estimators = st.sidebar.slider('Number of Trees:', 1, 100, 10)
    max_depth = st.sidebar.slider('Maximum Depth:', 1, 20, 5)
    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

# Function to fit LSTM model
def fit_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Function to predict next 5 days stock prices
def predict_next_5_days(model, data):
    next_dates = pd.date_range(data['Date'].iloc[-1], periods=6, freq='B')[1:]
    next_dates_str = next_dates.strftime('%Y-%m-%d').tolist()
    X_pred = np.array([d.timestamp() for d in next_dates]).reshape(-1, 1)
    next_5_days_predictions = model.predict(X_pred)
    next_5_days_predictions = next_5_days_predictions.flatten()  # Ensure predictions are 1-dimensional
    return next_dates_str, next_5_days_predictions

# Function to evaluate model and calculate metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    metrics = {
        'MSE': sk_metrics.mean_squared_error,
        'MAE': sk_metrics.mean_absolute_error,
        'RMSE': lambda y_true, y_pred: np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)),
        'R2': sk_metrics.r2_score,
        'Adj. R2': lambda y_true, y_pred: 1 - (1 - sk_metrics.r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - X_train.shape[1] - 1)
    }
    results = {metric: metric_func(y_train, y_train_pred) for metric, metric_func in metrics.items()}
    results.update({f'{metric} (test)': metric_func(y_test, y_test_pred) for metric, metric_func in metrics.items()})
    return results, y_train_pred, y_test_pred

# Function to detect upward or downward trend
def detect_trend(y_pred):
    if y_pred[-1] > y_pred[0]:
        return 'Upward'
    elif y_pred[-1] < y_pred[0]:
        return 'Downward'
    else:
        return 'No clear trend'

# Display next 5 days stock prices
def display_next_5_days_prediction(model, data):
    st.subheader('Next 5 Days Stock Price Predictions')
    X, y = get_X_y(data)
    X_train, y_train, X_test, y_test = split_data(X, y)
    next_dates, next_5_days_predictions = predict_next_5_days(model, data)
    next_days_data = pd.DataFrame({'Date': next_dates, 'Predicted Price': next_5_days_predictions})
    st.write(next_days_data)

# Display model evaluation metrics
def display_model_evaluation(results, selected_metric):
    st.subheader('Model Evaluation Metrics')
    st.write(f'Selected Metric: {selected_metric}')
    st.write(f'Training {selected_metric}: {results[selected_metric]:.2f}')
    st.write(f'Testing {selected_metric}: {results[selected_metric + " (test)"]:.2f}')

    # Plot actual vs predicted values as a table
    def plot_actual_vs_predicted_values_table(data, model, y_train_pred, y_test_pred):
        st.subheader('Actual vs Predicted Values')

        # Split data
        X_train, y_train, X_test, y_test = split_data(*get_X_y(data))

        # Concatenate actual values
        actual_values = np.concatenate([y_train, y_test])

        # Concatenate predicted values
        predicted_values = np.concatenate([y_train_pred, y_test_pred])

        # Create DataFrame
        df = pd.DataFrame({'Date': np.concatenate([X_train, X_test]),
                           'Actual Values': actual_values,
                           'Predicted Values': predicted_values})

        # Display as table
        visible_rows = st.slider('Number of Rows to Display:', min_value=5, max_value=len(df), value=10)
        st.dataframe(df.head(visible_rows))

        # Display raw data
        st.subheader('Raw Data')
        st.write(data.tail())

        # Plot raw data
        plot_raw_data(data)

        # Fit selected model and display results
        if selected_model == 'Linear Regression':
            model = fit_linear_regression(*get_X_y(data))
        elif selected_model == 'Random Forest':
            model = fit_random_forest(*get_X_y(data))
        elif selected_model == 'Gradient Boosting':
            model = fit_gradient_boosting(*get_X_y(data))
        elif selected_model == 'LSTM':
            model = fit_lstm(*get_X_y(data))

        # Predict next 5 days
        display_next_5_days_prediction(model, data)

        # Evaluate model
        results, y_train_pred, y_test_pred = evaluate_model(model, *split_data(*get_X_y(data)))

        # Display evaluation metrics
        display_model_evaluation(results, selected_metric)

        # Plot actual vs predicted values as a table
        plot_actual_vs_predicted_values_table(data, model, y_train_pred, y_test_pred)

# Function to plot additional graphs using Matplotlib and Plotly
def plot_additional_graphs(data):
    st.subheader('Additional Graphs')
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(data['Close'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Close Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Close Prices')
    st.pyplot(fig)

    # Plot line chart
    st.write('Plotting a line chart using Matplotlib')
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Closing Price Over Time')
    st.pyplot(fig)

    # Plot candlestick chart
    st.write('Plotting a candlestick chart using Plotly')
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart')
    st.plotly_chart(fig)

    # Plot dataset graphs
    st.subheader('Dataset Graphs')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], mode='lines', name='Low'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(title='OHLC Data Over Time', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
plot_raw_data(data)

# Fit selected model and display results
if selected_model == 'Linear Regression':
    model = fit_linear_regression(*get_X_y(data))
elif selected_model == 'Random Forest':
    model = fit_random_forest(*get_X_y(data))
elif selected_model == 'Gradient Boosting':
    model = fit_gradient_boosting(*get_X_y(data))
elif selected_model == 'LSTM':
    model = fit_lstm(*get_X_y(data))

# Display actual vs predicted values as a table
plot_actual_vs_predicted_values_table(*get_X_y(data), model, *evaluate_model(model, *split_data(*get_X_y(data)))[1:])

# Plot additional graphs
plot_additional_graphs(data)
