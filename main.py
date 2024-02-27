import streamlit as st
from datetime import date
import pandas as pd
import data_loading
import modeling
import visualization
import statsmodels.api as sm

st.title('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Create a sidebar for user options
st.sidebar.subheader('Select options')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
models = ('Linear Regression', 'Random Forest', 'Gradient Boosting', 'LSTM')
selected_model = st.sidebar.selectbox('Select model for prediction', models)
evaluation_metrics = ('MSE', 'MAE', 'RMSE', 'R2', 'Adj. R2')
selected_metric = st.sidebar.selectbox('Select metric for evaluation', evaluation_metrics)

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load data
data_load_state = st.text('Loading data...')
data = data_loading.load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Extract features and target
X, y = modeling.get_X_y(data)

# Split data into train and test sets
X_train, y_train, X_test, y_test = modeling.split_data(X, y)

# Fit model
if selected_model == 'Linear Regression':
    model = modeling.fit_linear_regression(X_train, y_train)
elif selected_model == 'Random Forest':
    model = modeling.fit_random_forest(X_train, y_train)
elif selected_model == 'Gradient Boosting':
    model = modeling.fit_gradient_boosting(X_train, y_train)
elif selected_model == 'LSTM':
    model = modeling.fit_lstm(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train, (X_train.shape[1], 1))

# Evaluate model
mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test, adj_r2_train, adj_r2_test, y_train_pred, y_test_pred = modeling.evaluate_model(model, X_train, y_train, X_test, y_test)

# Display evaluation metrics
st.subheader('Model Evaluation Metrics')
st.write(f'Mean Squared Error (MSE) for train set: {mse_train:.2f}')
st.write(f'Mean Squared Error (MSE) for test set: {mse_test:.2f}')
st.write(f'Mean Absolute Error (MAE) for train set: {mae_train:.2f}')
st.write(f'Mean Absolute Error (MAE) for test set: {mae_test:.2f}')
st.write(f'Root Mean Squared Error (RMSE) for train set: {rmse_train:.2f}')
st.write(f'Root Mean Squared Error (RMSE) for test set: {rmse_test:.2f}')
st.write(f'R-squared (R2) for train set: {r2_train:.2f}')
st.write(f'R-squared (R2) for test set: {r2_test:.2f}')
st.write(f'Adjusted R-squared (Adj. R2) for train set: {adj_r2_train:.2f}')
st.write(f'Adjusted R-squared (Adj. R2) for test set: {adj_r2_test:.2f}')

# Plot actual vs predicted values
st.subheader('Actual vs Predicted Values')
visualization.plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred)

# Display model-specific information
if selected_model == 'Random Forest' or selected_model == 'Gradient Boosting':
    try:
        if selected_model == 'Random Forest':
            feature_importance = model.feature_importances_
        elif selected_model == 'Gradient Boosting':
            feature_importance = model.feature_importances_

        if len(data.columns[:-1]) == len(feature_importance):
            st.subheader(f'Feature Importance for {selected_model} Model')
            df_feature_importance = pd.DataFrame({'Feature': data.columns[:-1], 'Importance': feature_importance})
            df_feature_importance = df_feature_importance.sort_values(by='Importance', ascending=False)
            st.write(df_feature_importance)
        else:
            st.write("Error: Number of features does not match feature importance array. Please check the data.")
    except AttributeError:
        st.write(f"Error: {selected_model} model does not support feature importance analysis.")

elif selected_model == 'Linear Regression':
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    model_sm = sm.OLS(y_train, X_train_sm)
    results = model_sm.fit()
    st.subheader('Summary Table for Linear Regression Model')
    st.write(results.summary())

elif selected_model == 'LSTM':
    st.subheader('Summary of LSTM Model Architecture')
    st.write(
        'LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture, commonly used for sequential data such as time series.')
    st.write(
        'The model architecture consists of multiple LSTM layers followed by one or more fully connected layers (Dense layers).')
    st.write(
        'The input shape of the LSTM layer is determined by the number of time steps (sequence length) and the number of features.')
    st.write('The output of the model is a single value representing the predicted price at the next time step.')
