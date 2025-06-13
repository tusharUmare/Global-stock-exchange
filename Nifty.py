import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Nifty Stock Price Prediction Dashboard")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload Nifty_Stocks.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("⚠️ Please upload the 'Nifty_Stocks.csv' file to proceed.")
    st.stop()

# Preprocessing
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol',
              'Category', 'Daily_Return', 'Price_Range', 'Volatility',
              'Cumulative_Return', 'Average_Price']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)
df = df.drop(columns=['Adj Close', 'Volume'])

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Feature Engineering Functions
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short=12, long=26, signal=9):
    short_ema = data.ewm(span=short, adjust=False).mean()
    long_ema = data.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal

def calculate_volatility(data, window=14):
    return data.pct_change().rolling(window=window).std()

# Apply Features
df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_sma(x, 50))
df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_sma(x, 200))
df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x))
df['MACD'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_macd(x))
df['Volatility'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_volatility(x))

df = df.dropna()

# Encode categorical columns
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# Sidebar selection
st.sidebar.title("📈 Stock Selector")
stock_names = df['Symbol'].unique()
selected_stock_encoded = st.sidebar.selectbox("Select Stock (code)", stock_names)
selected_df = df[df['Symbol'] == selected_stock_encoded]

# Prepare Data
X = selected_df.drop(columns=['Close', 'Date'])
y = selected_df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

# Show Metrics
st.subheader(f"📌 Model Performance for Stock Code: {selected_stock_encoded}")

col1, col2, col3 = st.columns(3)
col1.metric("Linear R²", f"{lr_r2*100:.2f}%")
col2.metric("RF R²", f"{rf_r2*100:.2f}%")
col3.metric("XGB R²", f"{xgb_r2*100:.2f}%")

# Plot Predictions
def plot_predictions(y_true, preds, title):
    fig, ax = plt.subplots()
    ax.plot(y_true.values, label='Actual', linewidth=2)
    ax.plot(preds, label='Predicted', linestyle='--')
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

st.markdown("### 🔍 Predicted vs Actual Prices")
plot_predictions(y_test, lr_pred, "Linear Regression")
plot_predictions(y_test, rf_pred, "Random Forest")
plot_predictions(y_test, xgb_pred, "XGBoost")

# Technical Indicators
st.markdown("### 🧮 Technical Indicators")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(selected_df['Date'], selected_df['Close'], label='Close Price', color='black')
ax.plot(selected_df['Date'], selected_df['SMA_50'], label='SMA 50', color='blue')
ax.plot(selected_df['Date'], selected_df['SMA_200'], label='SMA 200', color='orange')
ax.set_title("Price with SMA")
ax.legend()
st.pyplot(fig)

st.line_chart(selected_df.set_index('Date')[['RSI', 'MACD']])

# Optional: Show Data
with st.expander("📄 Show Data Table"):
    st.dataframe(selected_df.tail(50))
