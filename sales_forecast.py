# ==========================================
# PREDICTIVE ANALYTICS USING HISTORICAL DATA
# SALES FORECASTING WITH LINEAR REGRESSION
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("sales_data.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# ------------------------------------------
# Data Preprocessing
# ------------------------------------------

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing Values
df.dropna(inplace=True)

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Date
df = df.sort_values('Date')

# Create Numerical Feature for Regression
df['Month_Number'] = np.arange(1, len(df)+1)

print("\nProcessed Dataset:")
print(df.head())

# ------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Sales'], marker='o')
plt.title('Historical Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# ------------------------------------------
# Feature Selection
# ------------------------------------------

X = df[['Month_Number']]
y = df['Sales']

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Model Training
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred
})

print("\nSample Predictions:")
print(prediction_df.head())

# ------------------------------------------
# Model Evaluation
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("-" * 30)
print("MAE  :", round(mae,2))
print("MSE  :", round(mse,2))
print("RMSE :", round(rmse,2))
print("R²   :", round(r2,4))

# ------------------------------------------
# Future Forecasting
# ------------------------------------------

future_months = 12

future_X = pd.DataFrame({
    'Month_Number': range(
        len(df)+1,
        len(df)+future_months+1
    )
})

future_predictions = model.predict(future_X)

print("\nFuture 12-Month Forecast:")
for i, value in enumerate(future_predictions, start=1):
    print(f"Month {i}: {round(value,2)}")

# ------------------------------------------
# Create Future Dates
# ------------------------------------------

last_date = df['Date'].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=future_months,
    freq='M'
)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': future_predictions
})

print("\nForecast Table:")
print(forecast_df)

# ------------------------------------------
# Visualization
# ------------------------------------------

plt.figure(figsize=(14,6))

# Historical Sales
plt.plot(
    df['Date'],
    df['Sales'],
    label='Actual Sales',
    marker='o'
)

# Forecasted Sales
plt.plot(
    forecast_df['Date'],
    forecast_df['Forecasted Sales'],
    label='Forecasted Sales',
    marker='s'
)

plt.title('Sales Forecast for Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)

plt.show()

# ------------------------------------------
# Save Forecast Results
# ------------------------------------------

forecast_df.to_csv(
    "future_sales_forecast.csv",
    index=False
)

print("\nForecast saved as 'future_sales_forecast.csv'")
