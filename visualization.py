import streamlit as st
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

def plot_evaluation_metrics(mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train,
                            r2_test, adj_r2_train, adj_r2_test):
    # Plot evaluation metrics
    metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'Adj. R2']
    train_scores = [mse_train, mae_train, rmse_train, r2_train, adj_r2_train]
    test_scores = [mse_test, mae_test, rmse_test, r2_test, adj_r2_test]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    axes[0].bar(metrics, train_scores, color='skyblue', label='Train')
    axes[0].set_title('Train Set')
    axes[0].set_ylabel('Score')
    axes[0].legend()

    axes[1].bar(metrics, test_scores, color='salmon', label='Test')
    axes[1].set_title('Test Set')
    axes[1].set_ylabel('Score')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train, mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_test)), y=y_test, mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train_pred, mode='lines', name='Train Prediction'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_test)), y=y_test_pred, mode='lines', name='Test Prediction'))
    fig.update_layout(title='Actual vs Predicted Values', xaxis_title='Index', yaxis_title='Price')
    st.plotly_chart(fig)
