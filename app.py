# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load CSV
df = pd.read_csv("data/food_price.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Filter only food items
df = df[df["Category"].str.lower() == "food"]

# Choose the commodity
commodity = "Tomato"
df_veg = df[df["Commodity"].str.lower() == commodity.lower()].copy()

# Prepare monthly data
df_veg.loc[:, "MonthStart"] = df_veg["Date"].dt.to_period("M").dt.to_timestamp()
monthly = df_veg.groupby("MonthStart")["Retail Price"].mean().reset_index()
monthly.rename(columns={"MonthStart": "Date"}, inplace=True)

# Drop missing values
monthly = monthly.dropna(subset=["Retail Price"])

# Sort by date
monthly = monthly.sort_values("Date")

# Create features
monthly["Year"] = monthly["Date"].dt.year
monthly["Month"] = monthly["Date"].dt.month

X = monthly[["Year", "Month"]]
y = monthly["Retail Price"]

# Split data & train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future 12 months
last_date = monthly["Date"].max()
future_months = pd.date_range(start=last_date, periods=73, freq='ME')[1:]

future_df = pd.DataFrame({"Date": future_months})
future_df["Year"] = future_df["Date"].dt.year
future_df["Month"] = future_df["Date"].dt.month
future_df["Predicted Price"] = model.predict(future_df[["Year", "Month"]])

# Display predictions
print(future_df)

# Plot Actual vs Predicted Price

# Combine actual and predicted for plotting
plt.figure(figsize=(15, 6))

# Actual prices as bars
plt.bar(monthly["Date"], monthly["Retail Price"], width=20, label="Actual Price", color='skyblue')

# Predicted prices as bars (slightly shifted for visibility)
plt.bar(future_df["Date"], future_df["Predicted Price"], width=20, label="Predicted Price", color='orange')

# Formatting x-axis as Month-Year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.xlabel("Month-Year")
plt.ylabel("Price (₹)")
plt.title(f"Monthly Price Forecasting of {commodity} (Bar Graph)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()