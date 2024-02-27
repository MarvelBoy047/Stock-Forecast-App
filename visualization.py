import streamlit as st
import plotly.graph_objs as go
import numpy as np

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
