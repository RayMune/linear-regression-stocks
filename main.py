import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download historical stock price data for Tesla
ticker = "TSLA"
start_date = "2020-01-01"
end_date = "2021-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess the data by selecting the "Close" price and dropping any missing values
data = data[["Close"]].dropna()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Train a linear regression model on the training data
X_train = pd.DataFrame(range(len(train_data)))
y_train = train_data.values
model = LinearRegression()
model.fit(X_train, y_train)

# Use the trained model to make predictions for the test data
X_test = pd.DataFrame(range(len(train_data), len(data)))
y_test = test_data.values
y_pred = model.predict(X_test)

# Print the predicted average price for July
july_index = data.index.get_loc("2021-07-01")
july_price = y_pred[july_index - train_size]
print("Predicted average price for Tesla in July:", july_price)