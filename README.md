# Stock Forecast App

This is a Streamlit web application for predicting stock prices using various machine learning models.

## Description

This application allows users to select a stock ticker symbol, choose a machine learning model for prediction, and view the forecasted stock prices along with evaluation metrics.

## Features

- Choose from multiple machine learning models including Linear Regression, Random Forest, Gradient Boosting, and LSTM.
- Evaluate models using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R2), and Adjusted R-squared (Adj. R2).
- Visualize actual vs predicted stock prices over time.
- Display raw data and feature importance for selected models.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-forecast-app.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run main.py
```

4. Access the application in your browser:

```
http://localhost:8501
```


### Long Report: Stock Forecasting Web Application

#### Introduction
This project entails the development of a web application for stock forecasting using machine learning models. The application allows users to select a stock dataset, choose a predictive model, and evaluate the model's performance based on selected evaluation metrics.

#### Features
1. **Selection of Stock Dataset**: Users can choose from a list of stock datasets including Google (GOOG), Apple (AAPL), Microsoft (MSFT), and GameStop (GME).
2. **Model Selection**: Users can select one of the following predictive models: Linear Regression, Random Forest, Gradient Boosting, or Long Short-Term Memory (LSTM) networks.
3. **Evaluation Metrics**: Users can specify evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R2), or Adjusted R-squared (Adj. R2).
4. **Years of Prediction**: Users can select the number of years for which they want to forecast the stock prices.
5. **LSTM Configuration**: If the LSTM model is chosen, users can specify the number of epochs for training.

#### Implementation
- The web application is built using Streamlit, a Python library for creating interactive web applications.
- Data fetching is performed using the Yahoo Finance API (yfinance), allowing users to obtain historical stock price data.
- Predictive modeling is carried out using machine learning models such as Linear Regression, Random Forest, Gradient Boosting, and LSTM networks.
- Evaluation metrics are calculated to assess the performance of the selected model.
- The application provides visualizations including raw data plots and tables displaying actual vs predicted stock prices.

#### Usage
1. **Select Dataset**: Choose one of the available stock datasets from the sidebar.
2. **Choose Model**: Select a predictive model from the available options.
3. **Evaluation Metrics**: Specify the evaluation metric to assess the model's performance.
4. **Years of Prediction**: Adjust the slider to select the number of years for which to forecast the stock prices.
5. **LSTM Configuration**: If LSTM is chosen, configure the number of epochs using the slider.
6. **View Results**: The application displays the raw data, plots, and tables comparing actual vs predicted stock prices, along with evaluation metrics.

#### Conclusion
The Stock Forecasting Web Application provides a user-friendly interface for predicting stock prices using various machine learning models. By enabling users to interactively explore different datasets, models, and evaluation metrics, the application serves as a valuable tool for investors and analysts in making informed decisions.

### Brief Report

**Objective**: Develop a web application for stock price forecasting using machine learning models.

**Features**:
- Selection of stock dataset (GOOG, AAPL, MSFT, GME).
- Choice of predictive models: Linear Regression, Random Forest, Gradient Boosting, LSTM.
- Evaluation metrics: MSE, MAE, RMSE, R2, Adj. R2.
- Adjustable years of prediction.
- Configuration of LSTM model with adjustable epochs.
- Visualizations including raw data plots and tables comparing actual vs predicted stock prices.

**Implementation**:
- Built with Streamlit for creating interactive web applications.
- Data fetched from Yahoo Finance API (yfinance).
- Machine learning models: Linear Regression, Random Forest, Gradient Boosting, LSTM.
- Evaluation metrics calculated for model performance assessment.
- Visualizations include raw data plots and tables displaying actual vs predicted stock prices.

**Usage**:
1. Select dataset.
2. Choose predictive model.
3. Specify evaluation metric.
4. Adjust years of prediction.
5. Configure LSTM model if chosen.
6. View results: raw data, plots, and tables comparing actual vs predicted stock prices, along with evaluation metrics.

**Conclusion**: The Stock Forecasting Web Application facilitates stock price prediction using machine learning models, offering an intuitive interface for investors and analysts to explore different datasets, models, and evaluation metrics to make informed decisions.### Long Report: Stock Forecasting Web Application

#### Introduction
This project entails the development of a web application for stock forecasting using machine learning models. The application allows users to select a stock dataset, choose a predictive model, and evaluate the model's performance based on selected evaluation metrics.

#### Features
1. **Selection of Stock Dataset**: Users can choose from a list of stock datasets including Google (GOOG), Apple (AAPL), Microsoft (MSFT), and GameStop (GME).
2. **Model Selection**: Users can select one of the following predictive models: Linear Regression, Random Forest, Gradient Boosting, or Long Short-Term Memory (LSTM) networks.
3. **Evaluation Metrics**: Users can specify evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R2), or Adjusted R-squared (Adj. R2).
4. **Years of Prediction**: Users can select the number of years for which they want to forecast the stock prices.
5. **LSTM Configuration**: If the LSTM model is chosen, users can specify the number of epochs for training.

#### Implementation
- The web application is built using Streamlit, a Python library for creating interactive web applications.
- Data fetching is performed using the Yahoo Finance API (yfinance), allowing users to obtain historical stock price data.
- Predictive modeling is carried out using machine learning models such as Linear Regression, Random Forest, Gradient Boosting, and LSTM networks.
- Evaluation metrics are calculated to assess the performance of the selected model.
- The application provides visualizations including raw data plots and tables displaying actual vs predicted stock prices.

#### Usage
1. **Select Dataset**: Choose one of the available stock datasets from the sidebar.
2. **Choose Model**: Select a predictive model from the available options.
3. **Evaluation Metrics**: Specify the evaluation metric to assess the model's performance.
4. **Years of Prediction**: Adjust the slider to select the number of years for which to forecast the stock prices.
5. **LSTM Configuration**: If LSTM is chosen, configure the number of epochs using the slider.
6. **View Results**: The application displays the raw data, plots, and tables comparing actual vs predicted stock prices, along with evaluation metrics.

#### Conclusion
The Stock Forecasting Web Application provides a user-friendly interface for predicting stock prices using various machine learning models. By enabling users to interactively explore different datasets, models, and evaluation metrics, the application serves as a valuable tool for investors and analysts in making informed decisions.

### Brief Report

**Objective**: Develop a web application for stock price forecasting using machine learning models.

**Features**:
- Selection of stock dataset (GOOG, AAPL, MSFT, GME).
- Choice of predictive models: Linear Regression, Random Forest, Gradient Boosting, LSTM.
- Evaluation metrics: MSE, MAE, RMSE, R2, Adj. R2.
- Adjustable years of prediction.
- Configuration of LSTM model with adjustable epochs.
- Visualizations including raw data plots and tables comparing actual vs predicted stock prices.

**Implementation**:
- Built with Streamlit for creating interactive web applications.
- Data fetched from Yahoo Finance API (yfinance).
- Machine learning models: Linear Regression, Random Forest, Gradient Boosting, LSTM.
- Evaluation metrics calculated for model performance assessment.
- Visualizations include raw data plots and tables displaying actual vs predicted stock prices.

**Usage**:
1. Select dataset.
2. Choose predictive model.
3. Specify evaluation metric.
4. Adjust years of prediction.
5. Configure LSTM model if chosen.
6. View results: raw data, plots, and tables comparing actual vs predicted stock prices, along with evaluation metrics.

**Conclusion**: The Stock Forecasting Web Application facilitates stock price prediction using machine learning models, offering an intuitive interface for investors and analysts to explore different datasets, models, and evaluation metrics to make informed decisions.
