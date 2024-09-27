import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load textile export and reserve data
textile_exports_df = pd.read_excel('/content/drive/MyDrive/RSU50 DataFiles/Textile_Exports.xlsx')  # Load textile export data
reserve_df = pd.read_excel('/content/drive/MyDrive/RSU50 DataFiles/Reserve.xlsx')  # Load reserve data

# Merge datasets on 'Country' and 'Year', and only include years 1990-2018
merged_data = pd.merge(textile_exports_df, reserve_df, on=['Country', 'Year'])
merged_data = merged_data[(merged_data['Year'] >= 1990) & (merged_data['Year'] <= 2018)]

# Rename columns for easier access
merged_data.rename(columns={
    'Textiles and Clothing Export (US$ Thousand)': 'Textile_Exports',
    'Total reserves (includes gold, current US$) [FI.RES.TOTL.CD]': 'Reserves'
}, inplace=True)

# Ensure data is sorted by Country and Year
merged_data.sort_values(by=['Country', 'Year'], inplace=True)

# List of countries of interest
countries_of_interest = ['Sri Lanka', 'Malaysia', 'Thailand', 'Vietnam']

# Exclude Sri Lanka to compute average growth rates of comparator countries
comparator_countries = ['Malaysia', 'Thailand', 'Vietnam']

# Filter data for the countries of interest
df_filtered = merged_data[merged_data['Country'].isin(countries_of_interest)].copy()

df_filtered.head()

"""# 1. Comparative Time Series Analysis
Objective:

Compare the textile export trends and growth rates of Sri Lanka with Malaysia, Thailand, and Vietnam to identify patterns and potential benchmarks.
"""

# Calculate annual growth rates for Textile Exports
df_filtered['Textile_Export_Growth'] = df_filtered.groupby('Country')['Textile_Exports'].pct_change() * 100

# Plot annual textile export growth rates
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['Country'] == country]
    plt.plot(country_data['Year'], country_data['Textile_Export_Growth'], marker='o', label=country)

plt.title('Annual Textile Export Growth Rates')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Since we have annual data, and the time series may not be very long,
# we'll use a moving average to smooth out short-term fluctuations and highlight longer-term trends.

# Calculate a 3-year moving average for Textile Exports
df_filtered['Textile_Exports_MA'] = df_filtered.groupby('Country')['Textile_Exports'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

df_filtered['Textile_Exports_MA_Million'] = df_filtered['Textile_Exports_MA'] / 1000

# Plot the moving averages
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['Country'] == country]
    plt.plot(country_data['Year'], country_data['Textile_Exports_MA_Million'], marker='o', label=country)

plt.title('3-Year Moving Average of Textile Exports')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Comparing Actual Textile Exports

df_filtered['Textile_Exports_Million'] = df_filtered['Textile_Exports'] / 1000

# Plot actual textile exports over time
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['Country'] == country]
    plt.plot(country_data['Year'], country_data['Textile_Exports_Million'], marker='o', label=country)

plt.title('Textile Exports Over Time')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Applying Comparator Countries' Growth Rates to Sri Lanka

df_comparators = df_filtered[df_filtered['Country'].isin(comparator_countries)].copy()

# Calculate average annual growth rates of comparator countries
average_growth_rates = df_comparators.groupby('Year')['Textile_Export_Growth'].mean().reset_index(name='Avg_Comparator_Growth')

# Get Sri Lanka's data
df_sri_lanka = df_filtered[df_filtered['Country'] == 'Sri Lanka'].copy()

# Merge average comparator growth rates with Sri Lanka's data
df_sri_lanka = pd.merge(df_sri_lanka, average_growth_rates, on='Year', how='left')

# Reset index for iteration
df_sri_lanka.reset_index(drop=True, inplace=True)

# Initialize 'Potential_Textile_Exports' column
df_sri_lanka['Potential_Textile_Exports'] = 0.0

# Set the first year's potential exports to actual exports
df_sri_lanka.loc[0, 'Potential_Textile_Exports'] = df_sri_lanka.loc[0, 'Textile_Exports']

# Loop to compute potential exports
for i in range(1, len(df_sri_lanka)):
    prev_potential = df_sri_lanka.loc[i - 1, 'Potential_Textile_Exports']
    growth_rate = df_sri_lanka.loc[i, 'Avg_Comparator_Growth'] / 100
    df_sri_lanka.loc[i, 'Potential_Textile_Exports'] = prev_potential * (1 + growth_rate)

df_sri_lanka['Potential_Textile_Exports_Million'] = df_sri_lanka['Potential_Textile_Exports'] / 1000

plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Textile_Exports_Million'], marker='o', label='Potential Textile Exports (Based on Comparator Growth)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate the difference between potential and actual exports
df_sri_lanka['Export_Difference'] = df_sri_lanka['Potential_Textile_Exports'] - df_sri_lanka['Textile_Exports']

# Calculate total missed export revenue
total_missed_exports = df_sri_lanka['Export_Difference'].sum()
print(f"Total missed textile export revenue from 1990 to 2018(Potential): {total_missed_exports:,.2f} US$ Thousand")

"""# 2. Time Series Forecasting Using Comparator Countries' Patterns
Objective:

Use the textile export patterns of Malaysia, Thailand, and Vietnam to forecast Sri Lanka's textile exports and assess potential future trajectories.
"""

# Calculate average annual growth rate over the entire period for comparator countries
average_growth_rate = df_comparators['Textile_Export_Growth'].mean()
print(f"Average annual growth rate of comparator countries: {average_growth_rate:.2f}%")

# Initialize 'Forecasted_Textile_Exports' column
df_sri_lanka['Forecasted_Textile_Exports'] = 0.0

# Set the first year's forecasted exports to actual exports
df_sri_lanka.loc[0, 'Forecasted_Textile_Exports'] = df_sri_lanka.loc[0, 'Textile_Exports']

# Apply average growth rate to forecast future exports
for i in range(1, len(df_sri_lanka)):
    prev_forecast = df_sri_lanka.loc[i - 1, 'Forecasted_Textile_Exports']
    growth_rate = average_growth_rate / 100  # Convert percentage to decimal
    df_sri_lanka.loc[i, 'Forecasted_Textile_Exports'] = prev_forecast * (1 + growth_rate)

df_sri_lanka['Forecasted_Textile_Exports_Million'] = df_sri_lanka['Forecasted_Textile_Exports'] / 1000

plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Textile_Exports_Million'], marker='o', label='Forecasted Textile Exports (Based on Average Growth Rate)')
plt.title('Actual vs. Forecasted Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate the difference between potential and actual exports
df_sri_lanka['Export_Difference'] = df_sri_lanka['Forecasted_Textile_Exports'] - df_sri_lanka['Textile_Exports']

# Calculate total missed export revenue
total_missed_exports = df_sri_lanka['Export_Difference'].sum()
print(f"Total missed textile export revenue from 1990 to 2018(Forecasted): {total_missed_exports:,.2f} US$ Thousand")

"""#Comparison between approach 1 and 2"""

plt.figure(figsize=(12, 6))

# Plot Actual Textile Exports
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')

# Plot Potential Textile Exports from Option 1
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Textile_Exports_Million'], marker='o', label='Potential Exports (Approach 1)')

# Plot Forecasted Textile Exports from Option 2
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Textile_Exports_Million'], marker='o', label='Forecasted Exports (Approach 2)')

plt.title('Actual vs. Potential and Forecasted Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Difference for Approach 1
df_sri_lanka['Difference_Approach1'] = df_sri_lanka['Potential_Textile_Exports_Million'] - df_sri_lanka['Textile_Exports_Million']

# Difference for Approach 2
df_sri_lanka['Difference_Approach2'] = df_sri_lanka['Forecasted_Textile_Exports_Million'] - df_sri_lanka['Textile_Exports_Million']

plt.figure(figsize=(12, 6))

# Plot Difference for Option 1
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Difference_Approach1'], marker='o', label='Export Difference (Approach 1)')

# Plot Difference for Option 2
plt.plot(df_sri_lanka['Year'], df_sri_lanka['Difference_Approach2'], marker='o', label='Export Difference (Approach 2)')

plt.title('Export Differences Between Actual and Projected Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Export Difference (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""#3. Growth Differential Analysis
Objective:

Compute the difference in textile export growth rates between Sri Lanka and the average of the comparator countries to understand the magnitude of underperformance.
"""

# Merge Sri Lanka's growth rates with average comparator growth rates
growth_comparison = pd.merge(
    df_sri_lanka[['Year', 'Textile_Export_Growth']],
    average_growth_rates,
    on='Year',
    how='left'
)

# Calculate growth differential
growth_comparison['Growth_Differential'] = growth_comparison['Avg_Comparator_Growth'] - growth_comparison['Textile_Export_Growth']

plt.figure(figsize=(12, 6))
plt.bar(growth_comparison['Year'], growth_comparison['Growth_Differential'])
plt.title('Growth Differential Between Comparator Countries and Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Growth Differential (%)')
plt.grid(True)
plt.show()

# Apply the growth differentials cumulatively to estimate potential increases
growth_comparison['Cumulative_Differential'] = growth_comparison['Growth_Differential'].cumsum()

# Plot cumulative growth differential
plt.figure(figsize=(12, 6))
plt.plot(growth_comparison['Year'], growth_comparison['Cumulative_Differential'], marker='o')
plt.title('Cumulative Growth Differential Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Growth Differential (%)')
plt.grid(True)
plt.show()

"""#4. Forecasting Sri Lanka's Textile Exports Using Regression Models
Objective

Build a regression model using the textile export data of Malaysia, Thailand, and Vietnam to predict Sri Lanka's textile exports based on observed relationships.
"""

# Pivot the data to have countries as columns and years as rows
textile_exports_pivot = df_filtered.pivot(index='Year', columns='Country', values='Textile_Exports_Million')

# Ensure all countries are included
textile_exports_pivot = textile_exports_pivot[countries_of_interest]

# Drop rows with missing values
textile_exports_pivot.dropna(inplace=True)

# Display the pivot table
print(textile_exports_pivot.head())

# Independent variables (comparator countries)
X = textile_exports_pivot[comparator_countries]

# Dependent variable (Sri Lanka)
y = textile_exports_pivot['Sri Lanka']

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Print the coefficients
coefficients = pd.Series(model.coef_, index=X.columns)
print("Coefficients:")
print(coefficients)

print(f"Intercept: {model.intercept_:.2f}")

# Predict Sri Lanka's textile exports
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)
print(f"R-squared: {r2:.4f}")

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:,.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Textile Exports (Sri Lanka)')
plt.ylabel('Predicted Textile Exports')
plt.title('Actual vs. Predicted Textile Exports')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()