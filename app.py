import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64
import matplotlib.dates as mdates
import plotly.graph_objects as go


app = Flask(__name__)

# Load the pre-trained model
model = load_model('best_model.h5')

# Prepare the scaler used for scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))

def get_stock_data(ticker):
    # Download stock data for the last 1 year
    stock_data = yf.download(ticker, period="1y", interval="1d")
    return stock_data

def prepare_data_for_prediction(stock_data):
    # Normalize the stock prices using the scaler
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)
    
    # Prepare data for LSTM input (lookback window)
    lookback = 60
    X = []
    y = []
    
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be 3D as required by LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def predict_stock_price(ticker):
    # Get stock data
    stock_data = get_stock_data(ticker)
    
    # Prepare data for prediction
    X, y = prepare_data_for_prediction(stock_data)
    
    # Make predictions using the LSTM model
    predicted_price = model.predict(X[-1].reshape(1, 60, 1))  # Predict for the last window
    
    # Reverse scaling to get the actual price value
    predicted_price = scaler.inverse_transform(predicted_price)
    
    return predicted_price[0][0], stock_data

def get_investment_recommendation(predicted_price, current_price, risk_tolerance,investment_goal):
    # Simple rule-based recommendation system
    if investment_goal == "Short-term":
        # For short-term goals, focus more on immediate trends
        if predicted_price > current_price:
            if risk_tolerance >= 7:
                return "Buy quickly, but monitor closely (High Risk Tolerance)"
            elif risk_tolerance <= 3:
                return "Hold for now, avoid aggressive actions (Low Risk Tolerance)"
            else:
                return "Buy cautiously (Moderate Risk Tolerance)"
        else:
            return "Sell quickly to minimize losses (Short-term Goal)"
    elif investment_goal == "Long-term":
        # For long-term goals, assume market recovery and stability
        if predicted_price > current_price:
            if risk_tolerance >= 7:
                return "Buy and hold for long-term growth (High Risk Tolerance)"
            elif risk_tolerance <= 3:
                return "Hold for now, consider stability (Low Risk Tolerance)"
            else:
                return "Buy conservatively for gradual growth (Moderate Risk Tolerance)"
        else:
            return "Hold, markets often recover over time (Long-term Goal)"
    else:
        return "Invalid investment goal"
        
def generate_stock_plot(ticker, stock_data, predicted_price, current_price, lower_bound, upper_bound):
    plt.figure(figsize=(11, 9))

    # Plot the actual closing prices
    plt.plot(stock_data['Close'], label=f'{ticker} Actual Closing Price', color='blue', linewidth=2)

    # Plot the current closing price
    plt.scatter(stock_data.index[-1], current_price, color='green', s=150, label=f'Current Price: {current_price:.2f}', zorder=5)

    # Plot the predicted next closing price
    plt.scatter(stock_data.index[-1], predicted_price, color='red', s=150, label=f'Predicted Next Price: {predicted_price:.2f}', zorder=5)

    # Plot the confidence interval (shaded area)
    plt.fill_between(stock_data.index[-1:], lower_bound, upper_bound, color='red', alpha=0.3, label='Confidence Interval (3%)')

    # Title and labels
    plt.title(f'{ticker} Stock Price and Predicted Next Close Price with Confidence Interval', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price in USD', fontsize=14)
    
    # Format x-axis date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add a legend to describe each plot element
    plt.legend(loc='best', fontsize=14)

    # Add a text box with prediction value next to the predicted point
    plt.text(stock_data.index[-1], predicted_price + 5, f'{predicted_price:.2f}', fontsize=12, color='red', ha='left', va='bottom')

    # Save plot to a BytesIO object to serve as an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert image to Base64 for rendering in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    ticker = request.form['ticker']
    risk_tolerance = int(request.form['risk_tolerance'])  # Get user risk tolerance
    investment_goal = request.form['investment_goal']  # Get user investment goal

    # Make prediction
    predicted_price, stock_data = predict_stock_price(ticker)
    current_price = stock_data['Close'].iloc[-1].item()

    # Simulate confidence interval
    margin = 0.03 * predicted_price
    lower_bound = predicted_price - margin
    upper_bound = predicted_price + margin

    # Get recommendation based on risk tolerance
    recommendation = get_investment_recommendation(predicted_price, current_price, risk_tolerance,investment_goal)

    # Create plot as before
    img_base64 = generate_stock_plot(ticker, stock_data, predicted_price, current_price, lower_bound, upper_bound)

    return render_template('index.html', ticker=ticker, predicted_price=predicted_price, current_price=current_price, 
                           img_data=img_base64, recommendation=recommendation, investment_goal=investment_goal,
                           risk_tolerance=risk_tolerance,
                           lower_bound = lower_bound,
                           upper_bound=upper_bound)

if __name__ == '__main__':
    app.run(debug=True)
