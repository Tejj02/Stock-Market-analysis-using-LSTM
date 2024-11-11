import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt

# Load stock data (Replace with actual loading logic)
@st.cache_data
def load_data():
    data = pd.read_csv('TATAPOWER.NS_historical_data.csv')  # Example file path, replace with actual data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Load pre-trained LSTM model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('lstm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Prepare data for LSTM
def prepare_data_for_lstm(data, time_steps=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Predict future stock prices
def predict_stock(model, scaler, data, days_to_predict=30):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    X_input = last_60_days_scaled.reshape((1, last_60_days_scaled.shape[0], 1))

    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(X_input)
        predictions.append(pred[0][0])
        X_input = np.append(X_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    last_date = data.index[-1]
    dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    return predictions, dates

# Streamlit UI setup
def app():
    st.title('Stock Price Prediction with LSTM')

    # Load data and model
    data = load_data()
    model, scaler = load_model_and_scaler()

    
    # Create two columns: one for the slider, one for the prediction output
    col1, col2 = st.columns([1, 2])  # The left column will have the slider, and the right column will show predictions

    with col1:
        # User input for number of days to predict
        days_to_predict = st.slider('Select number of days to predict:', 1, 30, 7)

    with col2:
        if st.button('Predict'):
            # Predict the next 'days_to_predict' stock prices
            predictions, dates = predict_stock(model, scaler, data, days_to_predict)

            # Display the predictions in a DataFrame
            prediction_df = pd.DataFrame({
                'Date': dates,
                'Predicted Close': predictions.flatten()
            })

            st.write('Predicted Stock Prices:')
            st.write(prediction_df)

            # Plot the predictions
            plt.figure(figsize=(10, 6))
            plt.plot(dates, predictions, label=f'Predicted {days_to_predict} Days', color='orange')
            plt.title(f'Stock Price Prediction for Next {days_to_predict} Days')
            plt.xlabel('Date')
            plt.ylabel('Predicted Close Price')
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)

if __name__ == '__main__':
    app()
