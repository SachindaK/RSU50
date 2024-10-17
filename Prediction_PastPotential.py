import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score

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

"""#1. Without Incorporating Country-Specific Factors

## 1.1. Comparative Time Series Analysis
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

"""## 1.2. Time Series Forecasting Using Comparator Countries' Patterns
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

# Apply average growth rate to forecast potential exports
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

"""##Comparison between approach 1 and 2"""

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

"""##1.3. Growth Differential Analysis
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

"""##1.4. Forecasting Sri Lanka's Textile Exports Using Regression Models
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

"""##1.5. Analyzing Potential Reserves of Sri Lanka Using Textile Export Predictions"""

# # Step 1: Prepare the Data for Comparator Countries

# df_comparators = df_filtered[df_filtered['Country'].isin(comparator_countries)].copy()

# # Create lagged textile exports (1 to 3 years lag)
# for lag in range(1, 4):
#     df_comparators[f'Lag{lag}'] = df_comparators.groupby('Country')['Textile_Exports'].shift(lag)
#     df_comparators[f'Lag{lag}_Million'] = df_comparators[f'Lag{lag}'] / 1e3  # Convert to Millions

# # Convert Reserves to Millions for consistency
# df_comparators['Reserves_Million'] = df_comparators['Reserves'] / 1e6

# # Drop rows with missing values due to lagging
# df_comparators.dropna(subset=[f'Lag{lag}_Million' for lag in range(1, 4)], inplace=True)

# df_comparators

# # Step 2: Build the Regression Model

# # Independent variables: Lagged Textile Exports
# X = df_comparators[[f'Lag{lag}_Million' for lag in range(1, 4)]]

# # Dependent variable: Reserves
# y = df_comparators['Reserves_Million']

# # Fit the regression model
# model = LinearRegression()
# model.fit(X, y)

# # Print the coefficients and intercept
# coefficients = pd.Series(model.coef_, index=X.columns)
# print("Regression Coefficients:")
# print(coefficients)
# print(f"Intercept: {model.intercept_:.2f}")

# # Evaluate the model
# y_pred = model.predict(X)
# r2 = r2_score(y, y_pred)
# print(f"R-squared: {r2:.4f}")
# mse = mean_squared_error(y, y_pred)
# print(f"Mean Squared Error: {mse:,.2f}")

# # Step 3: Apply the Model to Sri Lanka

# # Sort Sri Lanka's data by Year
# df_sri_lanka.sort_values(by='Year', inplace=True)

# # Using Potential_Textile_Exports

# # Create lagged potential textile exports for Sri Lanka using the same generic names
# for lag in range(1, 4):
#     df_sri_lanka[f'Lag{lag}'] = df_sri_lanka['Potential_Textile_Exports'].shift(lag)
#     df_sri_lanka[f'Lag{lag}_Million'] = df_sri_lanka[f'Lag{lag}'] / 1e3  # Convert to Millions

# # Convert actual reserves to Millions
# df_sri_lanka['Reserves_Million'] = df_sri_lanka['Reserves'] / 1e6

# # Drop rows with missing lagged values
# df_sri_lanka.dropna(subset=[f'Lag{lag}_Million' for lag in range(1, 4)], inplace=True)

# # Prepare independent variables for prediction using the same feature names
# X_sri_lanka = df_sri_lanka[[f'Lag{lag}_Million' for lag in range(1, 4)]]

# # Predict potential reserves
# df_sri_lanka['Potential_Reserves_Million'] = model.predict(X_sri_lanka)

# # Calculate the difference between potential and actual reserves
# df_sri_lanka['Potential_Difference_Million'] = df_sri_lanka['Potential_Reserves_Million'] - df_sri_lanka['Reserves_Million']

# # Using Forecasted_Textile_Exports

# # Create lagged potential textile exports for Sri Lanka using the same generic names
# for lag in range(1, 4):
#     df_sri_lanka[f'Lag{lag}'] = df_sri_lanka['Forecasted_Textile_Exports'].shift(lag)
#     df_sri_lanka[f'Lag{lag}_Million'] = df_sri_lanka[f'Lag{lag}'] / 1e3  # Convert to Millions

# # Convert actual reserves to Millions
# df_sri_lanka['Reserves_Million'] = df_sri_lanka['Reserves'] / 1e6

# # Drop rows with missing lagged values
# df_sri_lanka.dropna(subset=[f'Lag{lag}_Million' for lag in range(1, 4)], inplace=True)

# # Prepare independent variables for prediction using the same feature names
# X_sri_lanka = df_sri_lanka[[f'Lag{lag}_Million' for lag in range(1, 4)]]

# # Predict potential reserves
# df_sri_lanka['Forecasted_Reserves_Million'] = model.predict(X_sri_lanka)

# # Calculate the difference between potential and actual reserves
# df_sri_lanka['Forecasted_Difference_Million'] = df_sri_lanka['Forecasted_Reserves_Million'] - df_sri_lanka['Reserves_Million']

# # Plot Actual vs. Potential Reserves
# plt.figure(figsize=(12, 6))
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Reserves_Million'], marker='o', label='Actual Reserves')
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Reserves_Million'], marker='o', label='Potential Reserves')
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Reserves_Million'], marker='o', label='Forecasted Reserves')
# plt.title('Actual vs. Potential and Forecasted Reserves for Sri Lanka')
# plt.xlabel('Year')
# plt.ylabel('Reserves (US$ Million)')
# plt.legend()
# plt.grid(True)
# plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
# plt.show()

# # Plot the difference in reserves over time
# plt.figure(figsize=(12, 6))
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Difference_Million'], marker='o', label='Potential Difference')
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Difference_Million'], marker='o', label='Forecasted Difference')
# plt.title('Difference Between Potential/Forecasted  and Actual Reserves for Sri Lanka')
# plt.xlabel('Year')
# plt.ylabel('Difference in Reserves (US$ Million)')
# plt.grid(True)
# plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
# plt.show()

# # Calculate total potential increase in reserves
# total_potential_increase = df_sri_lanka['Potential_Difference_Million'].sum()
# print(f"Total potential increase in reserves from {int(df_sri_lanka['Year'].min())} to {int(df_sri_lanka['Year'].max())} (Potential): {total_potential_increase:,.2f} US$ Million")

# total_potential_increase = df_sri_lanka['Forecasted_Difference_Million'].sum()
# print(f"Total potential increase in reserves from {int(df_sri_lanka['Year'].min())} to {int(df_sri_lanka['Year'].max())} (Forecasted): {total_potential_increase:,.2f} US$ Million")

"""Results are unrealistic. Hence tring to incorporate county specific details to normalize.

#2. Incorporate Country-Specific Factors
"""

# Load GDP and Population data
population_and_gdp_df = pd.read_excel('/content/drive/MyDrive/RSU50 DataFiles/Population_and_GDP.xlsx')

# Merge GDP and Population data with the main dataset
merged_data = pd.merge(df_filtered, population_and_gdp_df, on=['Country', 'Year'])

# Rename columns for clarity
merged_data.rename(columns={
    'GDP (current US$) [NY.GDP.MKTP.CD]': 'GDP',
    'Population, total [SP.POP.TOTL]': 'Population'
}, inplace=True)

# Convert variables to consistent units
merged_data['Textile_Exports_Million'] = merged_data['Textile_Exports'] / 1e3  # Convert to Millions
merged_data['Reserves_Million'] = merged_data['Reserves'] / 1e6  # Convert to Millions

# Calculate per capita and percentage measures
merged_data['Textile_Exports_per_Capita'] = merged_data['Textile_Exports'] / merged_data['Population']
merged_data['Reserves_per_Capita'] = merged_data['Reserves'] / merged_data['Population']
merged_data['Textile_Exports_to_GDP'] = merged_data['Textile_Exports'] / merged_data['GDP']
merged_data['Reserves_to_GDP'] = merged_data['Reserves'] / merged_data['GDP']
merged_data['GDP_per_Capita'] = merged_data['GDP'] / merged_data['Population']

"""##2.1 Potential Textile Exports using Annual AVG growth of compatitors"""

# Filter data for comparator countries
df_comparators = merged_data[merged_data['Country'].isin(comparator_countries)].copy()

# Calculate annual growth rates for normalized textile exports
df_comparators['Textile_Exports_per_Capita_Growth'] = df_comparators.groupby('Country')['Textile_Exports_per_Capita'].pct_change() * 100
df_comparators['Textile_Exports_to_GDP_Growth'] = df_comparators.groupby('Country')['Textile_Exports_to_GDP'].pct_change() * 100

# Calculate average growth rates per year
average_growth_rates_per_capita = df_comparators.groupby('Year')['Textile_Exports_per_Capita_Growth'].mean().reset_index(name='Avg_Growth_per_Capita')
average_growth_rates_to_GDP = df_comparators.groupby('Year')['Textile_Exports_to_GDP_Growth'].mean().reset_index(name='Avg_Growth_to_GDP')

# Get Sri Lanka's data
df_sri_lanka1 = merged_data[merged_data['Country'] == 'Sri Lanka'].copy()

# Merge average comparator growth rates with Sri Lanka's data
df_sri_lanka1 = pd.merge(df_sri_lanka1, average_growth_rates_per_capita[['Year', 'Avg_Growth_per_Capita']], on='Year', how='left')
df_sri_lanka1 = pd.merge(df_sri_lanka1, average_growth_rates_to_GDP[['Year', 'Avg_Growth_to_GDP']], on='Year', how='left')

# Reset index
df_sri_lanka1.reset_index(drop=True, inplace=True)

# Initialize potential normalized textile exports columns
df_sri_lanka1['Potential_Textile_Exports_per_Capita'] = 0.0
df_sri_lanka1['Potential_Textile_Exports_to_GDP'] = 0.0

# Set the first year's potential normalized exports to actual normalized exports
df_sri_lanka1.loc[0, 'Potential_Textile_Exports_per_Capita'] = df_sri_lanka1.loc[0, 'Textile_Exports_per_Capita']
df_sri_lanka1.loc[0, 'Potential_Textile_Exports_to_GDP'] = df_sri_lanka1.loc[0, 'Textile_Exports_to_GDP']

# Calculate potential normalized textile exports using average growth rates
for i in range(1, len(df_sri_lanka1)):
    # Per Capita
    prev_potential_per_capita = df_sri_lanka1.loc[i - 1, 'Potential_Textile_Exports_per_Capita']
    growth_rate_per_capita = df_sri_lanka1.loc[i, 'Avg_Growth_per_Capita'] / 100
    df_sri_lanka1.loc[i, 'Potential_Textile_Exports_per_Capita'] = prev_potential_per_capita * (1 + growth_rate_per_capita)

    # To GDP Ratio
    prev_potential_to_GDP = df_sri_lanka1.loc[i - 1, 'Potential_Textile_Exports_to_GDP']
    growth_rate_to_GDP = df_sri_lanka1.loc[i, 'Avg_Growth_to_GDP'] / 100
    df_sri_lanka1.loc[i, 'Potential_Textile_Exports_to_GDP'] = prev_potential_to_GDP * (1 + growth_rate_to_GDP)

# Potential Textile Exports in absolute terms
df_sri_lanka1['Potential_Textile_Exports_per_Capita_Absolute'] = df_sri_lanka1['Potential_Textile_Exports_per_Capita'] * df_sri_lanka1['Population']
df_sri_lanka1['Potential_Textile_Exports_to_GDP_Absolute'] = df_sri_lanka1['Potential_Textile_Exports_to_GDP'] * df_sri_lanka1['GDP']

# Convert to Millions for plotting
df_sri_lanka1['Textile_Exports_Million'] = df_sri_lanka1['Textile_Exports'] / 1e3  # From US$ Thousand to Million
df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'] = df_sri_lanka1['Potential_Textile_Exports_per_Capita_Absolute'] / 1e3
df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'] = df_sri_lanka1['Potential_Textile_Exports_to_GDP_Absolute'] / 1e3

# Plot Actual vs. Potential Textile Exports (Per Capita)
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'], marker='o', label='Potential Textile Exports (Calculated using \'Per Capita\' Annual average growth)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot Actual vs. Potential Textile Exports (To GDP)
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'], marker='o', label='Potential Textile Exports (Calculated using \'To GDP\' Annual average growth)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot Actual vs. Potential Textile Exports (To GDP)
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'], marker='o', label='Potential Textile Exports (Per Capita)')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'], marker='o', label='Potential Textile Exports (To GDP)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka (Per Capita/To GDP)')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate Export Differences (Per Capita)
df_sri_lanka1['Export_Difference_per_Capita_Million'] = df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'] - df_sri_lanka1['Textile_Exports_Million']

# Calculate Export Differences (To GDP)
df_sri_lanka1['Export_Difference_to_GDP_Million'] = df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'] - df_sri_lanka1['Textile_Exports_Million']

# Plot the Differences (Per Capita)
plt.figure(figsize=(12, 6))
plt.bar(df_sri_lanka1['Year'], df_sri_lanka1['Export_Difference_per_Capita_Million'])
plt.title('Difference Between Potential and Actual Textile Exports (Per Capita)')
plt.xlabel('Year')
plt.ylabel('Difference in Textile Exports (US$ Million)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot the Differences (To GDP)
plt.figure(figsize=(12, 6))
plt.bar(df_sri_lanka1['Year'], df_sri_lanka1['Export_Difference_to_GDP_Million'])
plt.title('Difference Between Potential and Actual Textile Exports (To GDP)')
plt.xlabel('Year')
plt.ylabel('Difference in Textile Exports (US$ Million)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

plt.figure(figsize=(12, 6))

# Plot Difference for Option 1
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Export_Difference_per_Capita_Million'], marker='o', label='Export Difference (Per Capita)')

# Plot Difference for Option 2
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Export_Difference_to_GDP_Million'], marker='o', label='Export Difference (To GDP)')

plt.title('Export Differences Between Actual and Potntial Exports (Per Capita/To GDP) for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Export Difference (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Total potential increase in textile exports (Per Capita)
total_potential_export_increase_per_capita = df_sri_lanka1['Export_Difference_per_Capita_Million'].sum()
print(f"Total potential increase in textile exports from {int(df_sri_lanka1['Year'].min())} to {int(df_sri_lanka1['Year'].max())} (Per Capita): {total_potential_export_increase_per_capita:,.2f} US$ Million")

# Total potential increase in textile exports (To GDP)
total_potential_export_increase_to_gdp = df_sri_lanka1['Export_Difference_to_GDP_Million'].sum()
print(f"Total potential increase in textile exports from {int(df_sri_lanka1['Year'].min())} to {int(df_sri_lanka1['Year'].max())} (To GDP): {total_potential_export_increase_to_gdp:,.2f} US$ Million")

"""##2.2. Reserves Prediction - Not a proper approch"""

# For comparator countries, create lagged variables
# Option A: Per Capita
for lag in range(1, 3):
    df_comparators[f'Lag{lag}_Per_Capita'] = df_comparators.groupby('Country')['Textile_Exports_per_Capita'].shift(lag)

# Option B: To GDP
for lag in range(1, 3):
    df_comparators[f'Lag{lag}_To_GDP'] = df_comparators.groupby('Country')['Textile_Exports_to_GDP'].shift(lag)

# Prepare datasets
# Option A: Per Capita
df_comp_per_capita = df_comparators.dropna(subset=[f'Lag{lag}_Per_Capita' for lag in range(1, 3)]).copy()
X_per_capita = df_comp_per_capita[[f'Lag{lag}_Per_Capita' for lag in range(1, 3)]].copy()
# X_per_capita['GDP_per_Capita'] = df_comp_per_capita['GDP_per_Capita']
y_per_capita = df_comp_per_capita['Reserves_per_Capita']

# Option B: To GDP
df_comp_to_gdp = df_comparators.dropna(subset=[f'Lag{lag}_To_GDP' for lag in range(1, 3)]).copy()
X_to_gdp = df_comp_to_gdp[[f'Lag{lag}_To_GDP' for lag in range(1, 3)]].copy()
X_to_gdp['GDP_per_Capita'] = df_comp_to_gdp['GDP_per_Capita']
y_to_gdp = df_comp_to_gdp['Reserves_to_GDP']

df_comp_to_gdp.info()

# Fit regression models
# Option A: Per Capita
model_per_capita = LinearRegression()
model_per_capita.fit(X_per_capita, y_per_capita)
print("Per Capita Model Coefficients:")
coefficients_per_capita = pd.Series(model_per_capita.coef_, index=X_per_capita.columns)
print(coefficients_per_capita)
print(f"Intercept: {model_per_capita.intercept_:.6f}")

# Option B: To GDP
model_to_gdp = LinearRegression()
model_to_gdp.fit(X_to_gdp, y_to_gdp)
print("To GDP Model Coefficients:")
coefficients_to_gdp = pd.Series(model_to_gdp.coef_, index=X_to_gdp.columns)
print(coefficients_to_gdp)
print(f"Intercept: {model_to_gdp.intercept_:.6f}")

# Prepare Sri Lanka's data
# Option A: Per Capita
for lag in range(1,3):
    df_sri_lanka1[f'Lag{lag}_Per_Capita'] = df_sri_lanka1['Potential_Textile_Exports_per_Capita'].shift(lag)
df_sri_lanka_per_capita = df_sri_lanka1.dropna(subset=[f'Lag{lag}_Per_Capita' for lag in range(1, 3)]).copy()
X_sri_lanka_per_capita = df_sri_lanka_per_capita[[f'Lag{lag}_Per_Capita' for lag in range(1, 3)]].copy()
# X_sri_lanka_per_capita['GDP_per_Capita'] = df_sri_lanka_per_capita['GDP_per_Capita']

# Option B: To GDP
for lag in range(1, 3):
    df_sri_lanka1[f'Lag{lag}_To_GDP'] = df_sri_lanka1['Potential_Textile_Exports_to_GDP'].shift(lag)
df_sri_lanka_to_gdp = df_sri_lanka1.dropna(subset=[f'Lag{lag}_To_GDP' for lag in range(1, 3)]).copy()
X_sri_lanka_to_gdp = df_sri_lanka_to_gdp[[f'Lag{lag}_To_GDP' for lag in range(1, 3)]].copy()
X_sri_lanka_to_gdp['GDP_per_Capita'] = df_sri_lanka_to_gdp['GDP_per_Capita']

# Predict potential reserves
# Option A: Per Capita
df_sri_lanka_per_capita['Potential_Reserves_per_Capita'] = model_per_capita.predict(X_sri_lanka_per_capita)
df_sri_lanka_per_capita['Potential_Reserves'] = df_sri_lanka_per_capita['Potential_Reserves_per_Capita'] * df_sri_lanka_per_capita['Population']
df_sri_lanka_per_capita['Reserves_Difference'] = df_sri_lanka_per_capita['Potential_Reserves'] - df_sri_lanka_per_capita['Reserves']
df_sri_lanka_per_capita['Reserves_Million'] = df_sri_lanka_per_capita['Reserves'] / 1e6
df_sri_lanka_per_capita['Potential_Reserves_Million'] = df_sri_lanka_per_capita['Potential_Reserves'] / 1e6
df_sri_lanka_per_capita['Reserves_Difference_Million'] = df_sri_lanka_per_capita['Potential_Reserves_Million'] - df_sri_lanka_per_capita['Reserves_Million']

# Option B: To GDP
df_sri_lanka_to_gdp['Potential_Reserves_to_GDP'] = model_to_gdp.predict(X_sri_lanka_to_gdp)
df_sri_lanka_to_gdp['Potential_Reserves'] = df_sri_lanka_to_gdp['Potential_Reserves_to_GDP'] * df_sri_lanka_to_gdp['GDP']
df_sri_lanka_to_gdp['Reserves_Difference'] = df_sri_lanka_to_gdp['Potential_Reserves'] - df_sri_lanka_to_gdp['Reserves']
df_sri_lanka_to_gdp['Reserves_Million'] = df_sri_lanka_to_gdp['Reserves'] / 1e6
df_sri_lanka_to_gdp['Potential_Reserves_Million'] = df_sri_lanka_to_gdp['Potential_Reserves'] / 1e6
df_sri_lanka_to_gdp['Reserves_Difference_Million'] = df_sri_lanka_to_gdp['Potential_Reserves_Million'] - df_sri_lanka_to_gdp['Reserves_Million']

# Plot Actual vs. Potential Reserves (Per Capita)
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka_per_capita['Year'], df_sri_lanka_per_capita['Reserves_Million'], marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka_per_capita['Year'], df_sri_lanka_per_capita['Potential_Reserves_Million'], marker='o', label='Potential Reserves (Per Capita)')
plt.title('Actual vs. Potential Reserves for Sri Lanka (Per Capita)')
plt.xlabel('Year')
plt.ylabel('Reserves (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot Actual vs. Potential Reserves (To GDP)
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka_to_gdp['Year'], df_sri_lanka_to_gdp['Reserves_Million'], marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka_to_gdp['Year'], df_sri_lanka_to_gdp['Potential_Reserves_Million'], marker='o', label='Potential Reserves (To GDP)')
plt.title('Actual vs. Potential Reserves for Sri Lanka (To GDP)')
plt.xlabel('Year')
plt.ylabel('Reserves (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot the Differences in Reserves (Per Capita)
plt.figure(figsize=(12, 6))
plt.bar(df_sri_lanka_per_capita['Year'], df_sri_lanka_per_capita['Reserves_Difference_Million'])
plt.title('Difference Between Potential and Actual Reserves (Per Capita)')
plt.xlabel('Year')
plt.ylabel('Difference in Reserves (US$ Million)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot the Differences in Reserves (To GDP)
plt.figure(figsize=(12, 6))
plt.bar(df_sri_lanka_to_gdp['Year'], df_sri_lanka_to_gdp['Reserves_Difference_Million'])
plt.title('Difference Between Potential and Actual Reserves (To GDP)')
plt.xlabel('Year')
plt.ylabel('Difference in Reserves (US$ Million)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate total potential increase in reserves
# Option A: Per Capita
total_potential_increase_per_capita = df_sri_lanka_per_capita['Reserves_Difference'].sum()
print(f"Total potential increase in reserves from {int(df_sri_lanka_per_capita['Year'].min())} to {int(df_sri_lanka_per_capita['Year'].max())} (Per Capita): {total_potential_increase_per_capita:,.2f} US$")

# Option B: To GDP
total_potential_increase_to_gdp = df_sri_lanka_to_gdp['Reserves_Difference'].sum()
print(f"Total potential increase in reserves from {int(df_sri_lanka_to_gdp['Year'].min())} to {int(df_sri_lanka_to_gdp['Year'].max())} (To GDP): {total_potential_increase_to_gdp:,.2f} US$")

"""##2.3 Potential Textile Exports without using AVG growth of compatitors

###2.3.1. Incorrect Approch - use local indicators on trainned model
"""

merged_data.info()

# Prepare the panel dataset
panel_data = merged_data.copy()

# # Convert data types
panel_data['Year'] = panel_data['Year'].astype(int)
panel_data['Country'] = panel_data['Country'].astype(str)

# # Set the index for panel data
# panel_data.set_index(['Country', 'Year'], inplace=True)

panel_data.info()

# Define the regression formula
formula = 'Textile_Exports_Million ~ GDP + Population + C(Country)'
# Fit the fixed effects model
model = smf.ols(formula=formula, data=panel_data).fit()
# Print the summary
# print(model.summary())

# Create a DataFrame with Sri Lanka's data
sri_lanka_data = panel_data[panel_data['Country'] == 'Sri Lanka'].copy()

# Use the model to predict
sri_lanka_data['Predicted_Textile_Exports'] = model.predict(sri_lanka_data)

plt.figure(figsize=(12, 6))
plt.plot(sri_lanka_data['Year'], sri_lanka_data['Textile_Exports_Million'], marker='o', label='Actual Textile Exports')
plt.plot(sri_lanka_data['Year'], sri_lanka_data['Predicted_Textile_Exports'], marker='o', label='Predicted Textile Exports')
plt.title('Actual vs. Potential (Using GDP and Population) Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

model.summary()

"""###2.3.2. Estimating Potential Beyond Current Constraints - Use Relative or Normalized Indicators - Linear Regression

When estimating potential textile exports, the goal is to assess what Sri Lanka could achieve under different conditions or by performing at the level of comparator countries. Using Sri Lanka's actual economic indicators (like GDP, population, etc.) may not fully capture this potential if those indicators themselves reflect current limitations or underperformance.
"""

# merged_data.info()

# create a list of our conditions
conditions = [
    (merged_data['Country'] == 'Malaysia'),
    (merged_data['Country'] == 'Thailand'),
    (merged_data['Country'] == 'Vietnam'),
    (merged_data['Country'] == 'Sri Lanka')
    ]

# create a list of the values we want to assign for each condition
values = [merged_data['Population']/330803, merged_data['Population']/513120, merged_data['Population']/331690, merged_data['Population']/65610]

# create a new column and use np.select to assign values to it using our lists as arguments
merged_data['Population_Density'] = np.select(conditions, values)

# Calculate lagged textile exports (previous year's exports)
merged_data['Lag_Textile_Exports'] = merged_data.groupby('Country')['Textile_Exports'].shift(1)

# Fill missing lagged exports with current exports
merged_data['Lag_Textile_Exports'].fillna(merged_data['Textile_Exports'], inplace=True)

# Calculate log-transformed textile exports
merged_data['Log_Textile_Exports'] = np.log(merged_data['Textile_Exports'])
merged_data['Log_Lag_Textile_Exports'] = np.log(merged_data['Lag_Textile_Exports'])

# Calculate growth rate using log differences
merged_data['Log_Textile_Export_Growth'] = merged_data.groupby('Country')['Log_Textile_Exports'].diff().fillna(0)

# Drop missing values due to lagging and differencing
# merged_data.dropna(subset=['Log_Textile_Export_Growth', 'Log_Lag_Textile_Exports', 'Log_Textile_Exports'], inplace=True)

merged_data['Textile_Export_Growth'] = merged_data.groupby('Country')['Textile_Export_Growth'].ffill().bfill()

comparator_countries=['Malaysia', 'Thailand', 'Vietnam']
# Filter data for comparator countries
df_comparators = merged_data[merged_data['Country'].isin(comparator_countries)].copy()

# Key indicators to consider
indicators = [
    'GDP_per_Capita', 'Textile_Exports_per_Capita', 'Population', 'GDP',
    'Textile_Exports_to_GDP', 'Reserves_per_Capita', 'Reserves_to_GDP', 'Textile_Exports','Textile_Export_Growth','Textile_Exports_MA'
    ,'Population_Density','Log_Textile_Export_Growth', 'Log_Lag_Textile_Exports', 'Log_Textile_Exports', 'Lag_Textile_Exports'
]

# Compute average indicators for comparator countries
average_indicators = df_comparators.groupby('Year')[indicators].mean().reset_index()
# average_indicators = merged_data.groupby('Year')[indicators].mean().reset_index()
average_indicators.rename(columns={col: 'Avg_' + col for col in indicators}, inplace=True)

# Filter data for Sri Lanka
df_sri_lanka2 = merged_data[merged_data['Country'] == 'Sri Lanka'].copy()

# Merge average indicators with Sri Lanka's data
df_sri_lanka2 = pd.merge(df_sri_lanka2, average_indicators, on='Year', how='left')

# Compute relative indicators
for col in indicators:
    df_sri_lanka2['Rel_' + col] = df_sri_lanka2[col] / df_sri_lanka2['Avg_' + col]

# Compute means and standard deviations for comparator countries
means = df_comparators[indicators].mean()
stds = df_comparators[indicators].std()

# Standardize indicators for comparator countries
for col in indicators:
    df_comparators['Std_' + col] = (df_comparators[col] - means[col]) / stds[col]

# Standardize indicators for Sri Lanka using comparator countries' stats
for col in indicators:
    df_sri_lanka2['Std_' + col] = (df_sri_lanka2[col] - means[col]) / stds[col]

# independent_vars = ['GDP_per_Capita', 'Population_Density', 'Log_Textile_Export_Growth'] # 2nd Best Approach

independent_vars = ['GDP_per_Capita', 'Population_Density', 'Log_Textile_Export_Growth', 'Reserves_per_Capita'] # The Best
# independent_vars = ['GDP_per_Capita', 'Population_Density', 'Textile_Export_Growth']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Population']#, 'Std_Textile_Exports_to_GDP', 'Std_Reserves_per_Capita']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Textile_Exports_to_GDP', 'Std_Reserves_per_Capita']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Reserves_per_Capita']

# df_comparators.dropna(subset=independent_vars, inplace=True)

# Prepare data for regression
X = df_comparators[independent_vars]
y = df_comparators['Std_Textile_Exports_per_Capita']

# Add constant term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Prepare Sri Lanka's data for prediction
X_sri_lanka = df_sri_lanka2[independent_vars]
X_sri_lanka = sm.add_constant(X_sri_lanka)

# Predict standardized textile exports
df_sri_lanka2['Std_Predicted_Textile_Exports_per_Capita'] = model.predict(X_sri_lanka)

# Convert predictions back to actual values
# df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] = df_sri_lanka2['Std_Predicted_Textile_Exports'] * stds['Textile_Exports'] + means['Textile_Exports']
df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] = (df_sri_lanka2['Std_Predicted_Textile_Exports_per_Capita'] * stds['Textile_Exports_per_Capita'] + means['Textile_Exports_per_Capita']) * df_sri_lanka2['Population']
# df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] = df_sri_lanka2['Std_Predicted_Textile_Exports_per_Capita']*df_sri_lanka2['Population']

# Plot actual vs. predicted textile exports
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Potential Textile Exports (Linear Regression)')
plt.title('Actual vs. Potential Textile Exports (Linear Regression) for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate the difference
df_sri_lanka2['Export_Difference'] = df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] - df_sri_lanka2['Textile_Exports']

# Total potential increase
total_potential_increase = df_sri_lanka2['Export_Difference'].sum()
print(f"Total potential increase in textile exports from {df_sri_lanka2['Year'].min()} to {df_sri_lanka2['Year'].max()}: {total_potential_increase / 1e3:,.2f} US$ Million")

# Plot the differences
plt.figure(figsize=(12, 6))
plt.bar(df_sri_lanka2['Year'], df_sri_lanka2['Export_Difference'] / 1e3)
plt.title('Difference Between Potential and Actual Textile Exports for Sri Lanka (Normalized Indicators)')
plt.xlabel('Year')
plt.ylabel('Difference in Textile Exports (US$ Million)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""###2.3.3. Random Forest"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_comparators[independent_vars]
y = df_comparators['Std_Textile_Exports_per_Capita']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

rf_regressor = RandomForestRegressor(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all available cores
)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_TX_rfr = rf_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_TX_rfr)
r2 = r2_score(y_test, y_pred_TX_rfr)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_TX_rfr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual Textile Exports')
plt.ylabel('Predicted Actual Textile Exports')
plt.title('Actual vs Predicted Textile Exports (Random Forest)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Prepare Sri Lanka's data for prediction
X_sri_lanka_rfr = df_sri_lanka2[independent_vars]

# Predict standardized textile exports
predicted_TX_rfr = rf_regressor.predict(X_sri_lanka_rfr)

# Convert predictions back to actual values
df_sri_lanka2['Predicted_Textile_Exports_RF'] = (predicted_TX_rfr * stds['Textile_Exports_per_Capita'] + means['Textile_Exports_per_Capita']) * df_sri_lanka2['Population']

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_RF'] / 1e3, label='Potential Textile Exports (Random Forest)', marker='o')
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual vs. Potential Textile Exports (Random Forest) for Sri Lanka')
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""###2.3.4. Support Vector Regressor (SVR):
Selected because it is said to be good for smaller datasets.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X = df_comparators[independent_vars]
y = df_comparators['Std_Textile_Exports_per_Capita']

# Split the data into Training and Testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

svr_regressor = SVR(kernel='rbf', C=100, epsilon=0.1)

# Train the model
svr_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_TX_svr = svr_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_TX_svr)
r2 = r2_score(y_test, y_pred_TX_svr)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_TX_svr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual Textile Exports')
plt.ylabel('Predicted Actual Textile Exports')
plt.title('Actual vs Predicted Textile Exports (SVR)')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Prepare Sri Lanka's data for prediction
X_sri_lanka_svr = df_sri_lanka2[independent_vars]

# Predict standardized textile exports
predicted_TX_svr = svr_regressor.predict(X_sri_lanka_svr)

# Convert predictions back to actual values
df_sri_lanka2['Predicted_Textile_Exports_SVR'] = (predicted_TX_svr * stds['Textile_Exports_per_Capita'] + means['Textile_Exports_per_Capita']) * df_sri_lanka2['Population']

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, label='Actual Textile Exports', marker='o')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_SVR'] / 1e3, label='Potential Textile Exports (SVR)', marker='o')
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual vs. Potential Textile Exports (SVR) for Sri Lanka')
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##2.4. Comparison Between Textile Export Potential Predictions"""

# Plot actual vs. Potential textile exports
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Potential Textile Exports (Linear Regression)')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_RF'] / 1e3, marker='o', label='Potential Textile Exports (Random Forest)')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_SVR'] / 1e3, marker='o', label='Potential Textile Exports (SVR)')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'], marker='o', label='Potential Textile Exports (Calculated using \'Per Capita\' Annual average growth)')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'], marker='o', label='Potential Textile Exports (Calculated using \'To GDP\' Annual average growth)')

# # Plot Potential Textile Exports from Option 1
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Textile_Exports_Million'], marker='o', label='Potential Exports (Approach 1)')

# # Plot Forecasted Textile Exports from Option 2
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Textile_Exports_Million'], marker='o', label='Forecasted Exports (Approach 2)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot actual vs. Potential textile exports
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Potential Textile Exports (Linear Regression)')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_RF'] / 1e3, marker='o', label='Potential Textile Exports (Random Forest)')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_SVR'] / 1e3, marker='o', label='Potential Textile Exports (SVR)')
# plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'], marker='o', label='Potential Textile Exports (Calculated using \'Per Capita\' Annual average growth)')
# plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'], marker='o', label='Potential Textile Exports (Calculated using \'To GDP\' Annual average growth)')

# # Plot Potential Textile Exports from Option 1
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Potential_Textile_Exports_Million'], marker='o', label='Potential Exports (Approach 1)')

# # Plot Forecasted Textile Exports from Option 2
# plt.plot(df_sri_lanka['Year'], df_sri_lanka['Forecasted_Textile_Exports_Million'], marker='o', label='Forecasted Exports (Approach 2)')
plt.title('Actual vs. Potential Textile Exports for Sri Lanka')
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

plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Predicted Textile Exports')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million'], marker='o', label='Potential Textile Exports (Per Capita)')
plt.plot(df_sri_lanka1['Year'], df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million'], marker='o', label='Potential Textile Exports (To GDP)')
plt.title('Textile Exports Over Time')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""#3. Analyzing Potential Reserves of Sri Lanka Using Textile Export Predictions

##3.1. Using Compatitor Indicators
"""

# Select standardized independent variables
# independent_vars = ['Std_GDP_per_Capita', 'Std_Population']#, 'Std_Textile_Exports_to_GDP', 'Std_Reserves_per_Capita']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Textile_Exports_to_GDP', 'Std_Reserves_per_Capita']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Reserves_per_Capita']
# independent_vars = ['Std_GDP_per_Capita', 'Std_Textile_Export_Growth']#, 'Std_Population']#, 'Textile_Export_Growth']

# independent_vars = ['GDP_per_Capita', 'Population_Density', 'Log_Textile_Export_Growth'] # 2nd Best Approach

# independent_vars = ['GDP_per_Capita', 'Population_Density', 'Log_Textile_Export_Growth', 'Textile_Exports_per_Capita'] # The Best
independent_vars = ['GDP_per_Capita', 'Population_Density', 'Textile_Exports_per_Capita']
# independent_vars = ['Log_Textile_Export_Growth', 'Log_Lag_Textile_Exports']#,'Std_Reserves_to_GDP']

# df_comparators.dropna(subset=independent_vars, inplace=True)

# Prepare data for regression
X = df_comparators[independent_vars]
y = df_comparators['Std_Reserves_per_Capita']

# Add constant term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Prepare Sri Lanka's data for prediction
X_sri_lanka = df_sri_lanka2[independent_vars]
X_sri_lanka = sm.add_constant(X_sri_lanka)

# Predict standardized textile exports
df_sri_lanka2['Std_Reserves_per_Capita'] = model.predict(X_sri_lanka)

# Convert predictions back to actual values
df_sri_lanka2['Predicted_Reserves'] = (df_sri_lanka2['Std_Reserves_per_Capita'] * stds['Reserves_per_Capita'] + means['Reserves_per_Capita']) * df_sri_lanka2['Population']

# Plot actual vs. predicted Reserves
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Reserves'] / 1e6, marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Potential Reserves')
plt.title('Actual vs. Predicted Potential Reserves for Sri Lanka (Normalized Indicators)')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Calculate the difference
df_sri_lanka2['Reserve_Difference'] = df_sri_lanka2['Predicted_Reserves'] - df_sri_lanka2['Reserves']

# Plot actual vs. predicted Reserves
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Reserves'] / 1e6, marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Predicted Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Reserves')
plt.title('Actual vs. Predicted Reserves for Sri Lanka (Normalized Indicators)')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##3.2. Using SL indicators - Linear Regression"""

# Select standardized independent variables
independent_vars = ['Textile_Export_Growth','Textile_Exports'] #Best


df_sri_lanka3 = df_sri_lanka2.copy()

# Prepare data for regression
X = df_sri_lanka3[independent_vars]
y = df_sri_lanka3['Reserves']

# Add constant term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

df_results=df_sri_lanka3['Year'].values
df_results = pd.DataFrame(df_results)
df_results['Actual_Textile_Exports'] = df_sri_lanka2['Textile_Exports'] / 1e3
df_results['Predicted_Potential_Textile_Exports_Using_Normalized_Indicators(2.4))'] = df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3
df_results['Predicted_Potential_Textile_Exports_Using_Per_Capita_Annual_Average_Growth(2.2))'] = df_sri_lanka1['Potential_Textile_Exports_per_Capita_Million']
df_results['Predicted_Potential_Textile_Exports_Using_To_GDP_Annual_Average_Growth(2.3))'] = df_sri_lanka1['Potential_Textile_Exports_to_GDP_Million']
df_results['Actual_Reserves'] = df_sri_lanka3['Reserves'] / 1e6
df_results['Predicted_Potential_Reserves_Using_Selected_Predicted_Textile_Exports(4.2)'] = df_sri_lanka3['Predicted_Reserves'] / 1e6
df_results['Predicted_Potential_Reserves_Using_Normalized_Compatitor_Indicators(4.1)'] = df_sri_lanka2['Predicted_Reserves'] / 1e6

df_results.head()

# df_results.to_excel('/content/drive/MyDrive/RSU50 DataFiles/Predictions.xlsx', index=False)

df_sri_lanka3['Textile_Exports'] = df_sri_lanka3['Predicted_Textile_Exports_per_Capita'].values

df_sri_lanka3['Textile_Export_Growth'] = df_sri_lanka3['Textile_Exports'].pct_change() * 100

df_sri_lanka3['Textile_Export_Growth'].bfill(inplace=True)

# Prepare Sri Lanka's data for prediction
X_sri_lanka = df_sri_lanka3[independent_vars]
X_sri_lanka = sm.add_constant(X_sri_lanka)

# Predict standardized textile exports
df_sri_lanka3['Predicted_Reserves'] = model.predict(X_sri_lanka)

# Plot actual vs. predicted Reserves
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka3['Year'], df_sri_lanka3['Reserves'] / 1e6, marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka3['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Potential Reserves (Linear Regression)')
# plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Reserves 1')
plt.title('Actual vs. Predicted Potential Reserves (Linear Regression) for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot actual vs. predicted Reserves
plt.figure(figsize=(12, 6))
plt.plot(df_sri_lanka3['Year'], df_sri_lanka3['Reserves'] / 1e6, marker='o', label='Actual Reserves')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka3['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Potential Reserves (Linear Regression)')
# plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Reserves'] / 1e6, marker='o', label='Predicted Reserves 1')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Textile_Exports'] / 1e3, marker='o', label='Actual Textile Exports')
plt.plot(df_sri_lanka2['Year'], df_sri_lanka2['Predicted_Textile_Exports_per_Capita'] / 1e3, marker='o', label='Potential Textile Exports')
plt.title('Actual vs. Predicted Potential Reserves and Textile Exports for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('Textile Exports (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##3.3. Using SL indicators - Random Forest"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df_sri_lanka_potential_rfr = df_sri_lanka3[['Year']].copy()
df_sri_lanka_potential_rfr['Predicted_Reserves'] = (df_sri_lanka3['Predicted_Reserves']/1e6).copy()
df_sri_lanka_potential_rfr['Actual_Textile_Export_Growth'] = df_sri_lanka3['Textile_Export_Growth'].copy()
df_sri_lanka_potential_rfr['Actual_Reserves'] = (df_sri_lanka3['Reserves']/1e6).copy()
df_sri_lanka_potential_rfr['Predicted_Textile_Exports'] = (df_sri_lanka3['Predicted_Textile_Exports_per_Capita']/1e3).copy()
df_sri_lanka_potential_rfr['Actual_Textile_Exports'] = (df_sri_lanka2['Textile_Exports']/1e3).copy()

df_sri_lanka_potential_rfr['Predicted_Textile_Export_Growth'] = df_sri_lanka_potential_rfr['Predicted_Textile_Exports'].pct_change() * 100

df_sri_lanka_potential_rfr['Predicted_Textile_Export_Growth'].bfill(inplace=True)

# Define features and target
training_features = ['Actual_Textile_Exports', 'Actual_Textile_Export_Growth']
# training_features = ['Actual_Textile_Exports']
target_variable = 'Actual_Reserves'

X = df_sri_lanka_potential_rfr[training_features]
y = df_sri_lanka_potential_rfr[target_variable]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

rf_regressor = RandomForestRegressor(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all available cores
)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rfr = rf_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_rfr)
r2 = r2_score(y_test, y_pred_rfr)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_rfr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel('Actual Reserves')
plt.ylabel('Predicted Actual Reserves')
plt.title('Actual vs Predicted Reserves')
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

prediction_features = ['Predicted_Textile_Exports', 'Predicted_Textile_Export_Growth']
# prediction_features = ['Predicted_Textile_Exports']

X_new= pd.DataFrame()
# Prepare the input data for prediction
X_new[training_features] = df_sri_lanka_potential_rfr[prediction_features]

predicted_reserves = rf_regressor.predict(X_new)

df_sri_lanka_potential_rfr['Predicted_Reserves_RF'] = predicted_reserves

# Ensure that the index is a DatetimeIndex
if not isinstance(df_sri_lanka_potential_rfr.index, pd.DatetimeIndex):
    df_sri_lanka_potential_rfr.index = pd.to_datetime(df_sri_lanka_potential_rfr.index)

# Sort the DataFrame by Date
df_sri_lanka_potential_rfr.sort_index(inplace=True)

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Actual_Reserves'], label='Actual Reserves', marker='o')
plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Predicted_Reserves_RF'], label='Predicted Potential Reserves (Random Forest)', marker='o')
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual and Predicted Potential Reserves (Random Forest)', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Actual_Reserves'], label='Actual Reserves', marker='o')
plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Predicted_Reserves_RF'], label='Predicted Reserves (Random Forest)', marker='o')
plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Actual_Textile_Exports'], label='Actual Textile Exports', marker='o')
plt.plot(df_sri_lanka_potential_rfr['Year'], df_sri_lanka_potential_rfr['Predicted_Textile_Exports'], label='Predicted Textile Exports', marker='o')
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual and Predicted Potential Reserves (Random Forest) & Textile Exports Over Time', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##3.4. Using SL indicators - Support Vector Regressor (SVR):
Selected because it is said to be good for smaller datasets.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df_sri_lanka_potential_svr = df_sri_lanka_potential_rfr.copy()

training_features = ['Actual_Textile_Exports', 'Actual_Textile_Export_Growth']
target_variable = 'Actual_Reserves'

X = df_sri_lanka_potential_svr[training_features]
y = df_sri_lanka_potential_svr[target_variable]

# Split the data into Training and Testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

svr_regressor = SVR(kernel='rbf', C=100, epsilon=0.1)

# Train the model
svr_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svr = svr_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

prediction_features = ['Predicted_Textile_Exports', 'Predicted_Textile_Export_Growth']

# Prepare the input data for prediction
X_new = df_sri_lanka_potential_svr[prediction_features]

# Rename the predicted features to match the actual feature names used during training
X_new.rename(columns={
    'Predicted_Textile_Exports': 'Actual_Textile_Exports',
    'Predicted_Textile_Export_Growth': 'Actual_Textile_Export_Growth'
}, inplace=True)


# Predict using the trained SVR model
predicted_reserves_svr = svr_regressor.predict(X_new)

# Add the predictions to the dataframe (optional)
df_sri_lanka_potential_svr['Predicted_Reserves_SVR'] = predicted_reserves_svr

# print("\nPredicted Reserves based on Predicted_* features using SVR:")
# print(df_sri_lanka_potential_svr[['Predicted_Reserves', 'Predicted_Reserves_SVR']])

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Actual_Reserves'], label='Actual Reserves', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Reserves_SVR'], label='Predicted Potential Reserves (SVR)', marker='o')

# Enhancements
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual and Predicted Potential Reserves (SVR)', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Actual_Reserves'], label='Actual Reserves', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Reserves_SVR'], label='Predicted Potential Reserves (SVR)', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Actual_Textile_Exports'], label='Actual Textile Exports', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Textile_Exports'], label='Predicted Textile Exports', marker='o')

# Enhancements
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual and Predicted Potential Reserves (SVR) & Textile Exports Over Time', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##3.5. Comparison of the predicted reserves"""

plt.figure(figsize=(12, 6))

plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Actual_Reserves'], label='Actual Reserves', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Reserves'], label='Predicted Potential Reserves (Linear Regression)', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Reserves_RF'], label='Predicted Potential Reserves (Random Forest)', marker='o')
plt.plot(df_sri_lanka3['Year'], df_sri_lanka_potential_svr['Predicted_Reserves_SVR'], label='Predicted Potential Reserves (SVR)', marker='o')

# Enhancements
plt.xlabel('Year', fontsize=12)
plt.ylabel('US$ Million', fontsize=12)
plt.title('Actual and All the Predicted Potential Reserves', fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout() 
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""#4. Time Series

##4.1. statsmodels - ARIMA - Textile Exports
"""

df_sri_lanka_potential_TS = df_sri_lanka3.copy()
# df_sri_lanka_potential_TS.info()

df_sri_lanka_potential_TS = df_sri_lanka3[['Year','Textile_Exports','Reserves','Predicted_Textile_Exports_per_Capita','Predicted_Reserves']].copy()

# Convert 'Year' to datetime and set as index
df_sri_lanka_potential_TS['Year'] = pd.to_datetime(df_sri_lanka_potential_TS['Year'], format='%Y')
df_sri_lanka_potential_TS.set_index('Year', inplace=True)

# Extract the time series data
textile_exports_ts = df_sri_lanka_potential_TS['Predicted_Textile_Exports_per_Capita']/ 1e3
reserves_ts = df_sri_lanka_potential_TS['Predicted_Reserves']/ 1e6

plt.figure(figsize=(12, 6))
plt.plot(textile_exports_ts, label='Potential Textile Exports')
plt.title('Potential Textile Exports Time Series')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

def adf_test(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

# Apply ADF test
adf_test(textile_exports_ts)

"""the series is non-stationary, so difference it."""

# First-order differencing
textile_exports_ts_diff = textile_exports_ts.diff().dropna()
adf_test(textile_exports_ts_diff)

# First-order differencing
textile_exports_ts_diff2 = textile_exports_ts_diff.diff().dropna()
adf_test(textile_exports_ts_diff2)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12,6))
plt.subplot(121)
plot_acf(textile_exports_ts_diff2, ax=plt.gca(), lags=20)
plt.subplot(122)
plot_pacf(textile_exports_ts_diff2, ax=plt.gca(), lags=10)
plt.show()

# Assuming p=1, d=1 (since we differenced twice), q=1
model_textile_exports = ARIMA(textile_exports_ts, order=(1,2,1))
model_textile_exports_fit = model_textile_exports.fit()
print(model_textile_exports_fit.summary())

# Forecasting the next 10 years
forecast_steps = 10
forecast_textile_exports = model_textile_exports_fit.forecast(steps=forecast_steps)

# Create a new date index for future dates
last_year = textile_exports_ts.index[-1].year
future_years = pd.date_range(start=f'{last_year+1}', periods=forecast_steps, freq='Y')

# Combine historical and forecasted data
forecast_textile_exports.index = future_years

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(textile_exports_ts, label='Historical Potential Textile Exports')
plt.plot(forecast_textile_exports, label='Forecasted Potential Textile Exports', linestyle='--')
plt.title('Forecasted Potential Textile Exports')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Assuming p=1, d=1 (since we differenced twice), q=1
model_resrves = ARIMA(reserves_ts, order=(1,2,1))
model_resrves_fit = model_resrves.fit()
print(model_resrves_fit.summary())

forecast_resrves = model_resrves_fit.forecast(steps=forecast_steps)

# Create a new date index for future dates
last_year = reserves_ts.index[-1].year
future_years = pd.date_range(start=f'{last_year+1}', periods=forecast_steps, freq='Y')

# Combine historical and forecasted data
forecast_resrves.index = future_years

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(reserves_ts, label='Historical Potential Reserves')
plt.plot(forecast_resrves, label='Forecasted Potential Reserves', linestyle='--')
plt.title('Forecasted Potential Reserves')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(reserves_ts, label='Historical Potential Reserves')
plt.plot(forecast_resrves, label='Forecasted Potential Reserves', linestyle='--')
plt.plot(textile_exports_ts, label='Historical Potential Textile Exports')
plt.plot(forecast_textile_exports, label='Forecasted Potential Textile Exports', linestyle='--')
plt.title('Forecasted Potential Textile Exports and Reserves (ARIMA)')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##4.2. Using pmdarima for Auto ARIMA"""

#install pmdarima
!pip install pmdarima

import pmdarima as pm

# For potential textile exports
model_textile_exports_auto = pm.auto_arima(textile_exports_ts, seasonal=False, stepwise=True, suppress_warnings=True)
print(model_textile_exports_auto.summary())

# Forecast
forecast_textile_exports_auto = model_textile_exports_auto.predict(n_periods=forecast_steps)
# forecast_textile_exports_auto = pd.Series(forecast_textile_exports_auto, index=future_years)

# For potential reserves
model_reserves_auto = pm.auto_arima(reserves_ts, seasonal=False, stepwise=True, suppress_warnings=True)
print(model_reserves_auto.summary())

# Forecast
forecast_reserves_auto = model_reserves_auto.predict(n_periods=forecast_steps)
# forecast_reserves_auto = pd.Series(forecast_reserves_auto, index=future_years)

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(textile_exports_ts, label='Historical Potential Textile Exports')
plt.plot(forecast_textile_exports_auto, label='Forecasted Potential Textile Exports', linestyle='--')
plt.title('Forecasted Potential Textile Exports')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot the forecast
plt.figure(figsize=(12,6))
plt.plot(reserves_ts, label='Historical Potential Reserves')
plt.plot(forecast_reserves_auto, label='Forecasted Potential Reserves', linestyle='--')
plt.title('Forecasted Potential Reserves')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""##4.3. Using Prophet"""

from prophet import Prophet

"""###4.3.1. Predict future using actual past"""

# For actual textile exports
df_exports_actual = df_sri_lanka2.reset_index()[['Year', 'Textile_Exports']]
df_exports_actual.columns = ['ds', 'y']

df_exports_actual['y'] = df_exports_actual['y'] / 1e3

# Convert 'Year' to datetime
df_exports_actual['ds'] = pd.to_datetime(df_exports_actual['ds'], format='%Y')

model_exports_actual = Prophet()
model_exports_actual.fit(df_exports_actual)

future_exports_actual = model_exports_actual.make_future_dataframe(periods=forecast_steps, freq='Y')
forecast_exports_actual = model_exports_actual.predict(future_exports_actual)

# Plot forecast
model_exports_actual.plot(forecast_exports_actual)
plt.title('Forecasted Actual Textile Exports using Prophet')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# For potential reserves
df_reserves_actual = df_sri_lanka2.reset_index()[['Year', 'Reserves']]
df_reserves_actual.columns = ['ds', 'y']

df_reserves_actual['y'] = df_reserves_actual['y'] / 1e6

# Convert 'Year' to datetime
df_reserves_actual['ds'] = pd.to_datetime(df_reserves_actual['ds'], format='%Y')

model_reserves_actual = Prophet()
model_reserves_actual.fit(df_reserves_actual)

future_reserves_actual = model_reserves_actual.make_future_dataframe(periods=forecast_steps, freq='Y')
forecast_reserves_actual = model_reserves_actual.predict(future_reserves_actual)

# Plot forecast
model_reserves_actual.plot(forecast_reserves_actual)
plt.title('Forecasted Actual Reserves using Prophet')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""###4.3.2. Predict future using predicted past potential"""

# For potential textile exports
df_exports_past_potential = df_sri_lanka3.reset_index()[['Year', 'Predicted_Textile_Exports_per_Capita']]
df_exports_past_potential.columns = ['ds', 'y']

df_exports_past_potential['y'] = df_exports_past_potential['y'] / 1e3

# Convert 'Year' to datetime
df_exports_past_potential['ds'] = pd.to_datetime(df_exports_past_potential['ds'], format='%Y')

model_exports_past_potential = Prophet()
model_exports_past_potential.fit(df_exports_past_potential)

future_exports_past_potential = model_exports_past_potential.make_future_dataframe(periods=forecast_steps, freq='Y')
forecast_exports_past_potential = model_exports_past_potential.predict(future_exports_past_potential)

# Plot forecast
model_exports_past_potential.plot(forecast_exports_past_potential)
plt.title('Forecasted Potential Textile Exports using Prophet')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# For potential reserves
df_reserves_past_potential = df_sri_lanka3.reset_index()[['Year', 'Predicted_Reserves']]
df_reserves_past_potential.columns = ['ds', 'y']

df_reserves_past_potential['y'] = df_reserves_past_potential['y'] / 1e6

# Convert 'Year' to datetime
df_reserves_past_potential['ds'] = pd.to_datetime(df_reserves_past_potential['ds'], format='%Y')

model_reserves_past_potential = Prophet()
model_reserves_past_potential.fit(df_reserves_past_potential)

future_reserves_past_potential = model_reserves_past_potential.make_future_dataframe(periods=forecast_steps, freq='Y')
forecast_reserves_past_potential = model_reserves_past_potential.predict(future_reserves_past_potential)

# Plot forecast
model_reserves_past_potential.plot(forecast_reserves_past_potential)
plt.title('Forecasted Potential Reserves using Prophet')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot
plt.figure(figsize=(12, 6))

# Plot Potential Textile Exports
plt.plot(forecast_exports_past_potential['ds'], forecast_exports_past_potential['yhat'], label='Potential Textile Exports Forecast', color='blue', linestyle='dashed')
plt.fill_between(forecast_exports_past_potential['ds'], forecast_exports_past_potential['yhat_lower'], forecast_exports_past_potential['yhat_upper'], color='blue', alpha=0.2)

# Plot Potential Reserves
plt.plot(forecast_reserves_past_potential['ds'], forecast_reserves_past_potential['yhat'], label='Potential Reserves Forecast', color='green', linestyle='dashed')
plt.fill_between(forecast_reserves_past_potential['ds'], forecast_reserves_past_potential['yhat_lower'], forecast_reserves_past_potential['yhat_upper'], color='green', alpha=0.2)

# Plot historical data for reference
plt.plot(df_exports_past_potential['ds'], df_exports_past_potential['y'], label='Past Potential - Textile Exports', color='blue')
plt.plot(df_reserves_past_potential['ds'], df_reserves_past_potential['y'], label='Past Potential - Reserves', color='green')

# Customizing the plot
plt.title('Forecasted Potential Textile Exports and Reserves (Prophet)')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot settings
plt.figure(figsize=(14, 8))

# Plot Potential Textile Exports
plt.plot(forecast_exports_actual['ds'], forecast_exports_actual['yhat'], label='Potential Textile Exports Forecast', color='blue', linestyle='dashed')
plt.fill_between(forecast_exports_actual['ds'], forecast_exports_actual['yhat_lower'], forecast_exports_actual['yhat_upper'], color='blue', alpha=0.2)

# Plot Potential Reserves
plt.plot(forecast_reserves_actual['ds'], forecast_reserves_actual['yhat'], label='Potential Reserves Forecast', color='green', linestyle='dashed')
plt.fill_between(forecast_reserves_actual['ds'], forecast_reserves_actual['yhat_lower'], forecast_reserves_actual['yhat_upper'], color='green', alpha=0.2)

# Plot historical data for reference
plt.plot(df_exports_actual['ds'], df_exports_actual['y'], label='Actual - Textile Exports', color='blue')
plt.plot(df_reserves_actual['ds'], df_reserves_actual['y'], label='Actual - Reserves', color='green')

# Customizing the plot
plt.title('Prophet Forecasted Actual Textile Exports and Actual Reserves for Sri Lanka')
plt.xlabel('Year')
plt.ylabel('US$ Million')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()

"""#5. Analysis using pyts"""

!pip install --upgrade pyts

import pyts
print(f"pyts version: {pyts.__version__}")

from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation
from pyts.metrics import dtw
# from pyts.shapelets import ShapeletModel

df_sri_lanka_potential_pyts = df_sri_lanka3[['Year','Predicted_Reserves','Textile_Export_Growth']].copy()
# For actual textile exports
df_sri_lanka_potential_pyts['Actual_Reserves'] = df_sri_lanka3['Reserves'].copy()
df_sri_lanka_potential_pyts['Predicted_Textile_Exports'] = df_sri_lanka3['Predicted_Textile_Exports_per_Capita'].copy()
df_sri_lanka_potential_pyts['Actual_Textile_Exports'] = df_sri_lanka2['Textile_Exports'].copy()

# Convert 'Year' to datetime and set as index
df_sri_lanka_potential_pyts['Year'] = pd.to_datetime(df_sri_lanka_potential_pyts['Year'], format='%Y')
df_sri_lanka_potential_pyts.set_index('Year', inplace=True)

df_sri_lanka_potential_pyts.info()

# # Extract the time series data
# textile_exports_paa = df_sri_lanka_potential_pyts['Predicted_Textile_Exports_per_Capita']/ 1e3
# reserves_paa = df_sri_lanka_potential_pyts['Predicted_Reserves']/ 1e6

# # Instantiate PAA with desired number of segments
# paa = PiecewiseAggregateApproximation(window_size=2)

# # Apply PAA to Potential Textile Exports
# textile_exports_paa = paa.transform(textile_exports_paa.values.reshape(1, -1))
# reserves_paa = paa.transform(reserves_paa.values.reshape(1, -1))

# Instantiate PAA with desired number of segments
paa = PiecewiseAggregateApproximation(window_size=5, output_size=None, overlapping=True)

# Apply PAA to Potential Textile Exports
textile_exports_actual_paa = paa.fit_transform((df_sri_lanka_potential_pyts['Actual_Textile_Exports'].values/ 1e3).reshape(1, -1))
textile_exports_potential_paa = paa.fit_transform((df_sri_lanka_potential_pyts['Predicted_Textile_Exports'].values/ 1e3).reshape(1, -1))

print("PAA Transformed Actual Textile Exports:")
print(textile_exports_actual_paa)

print("PAA Transformed Potential Textile Exports:")
print(textile_exports_potential_paa)

# Instantiate SAX with desired parameters
sax = SymbolicAggregateApproximation(n_bins=3, strategy='quantile', raise_warning=True, alphabet=None)

# Apply SAX to Potential Textile Exports
textile_exports_actual_sax = sax.fit_transform((df_sri_lanka_potential_pyts['Actual_Textile_Exports'].values/ 1e3).reshape(1, -1))
textile_exports_potential_sax = sax.fit_transform((df_sri_lanka_potential_pyts['Predicted_Textile_Exports'].values/ 1e3).reshape(1, -1))

print("SAX Transformed Actual Textile Exports:")
print(textile_exports_actual_sax)

print("SAX Transformed Potential Textile Exports:")
print(textile_exports_potential_sax)

from pyts.transformation import ShapeletTransform

# Define sliding window parameters
window_size = 5
stride = 1

# Function to create sliding windows
def create_sliding_windows(data, target_data, window_size, stride=1):
    # n_windows = (len(data) - window_size) // stride + 1
    # windows = np.array([data[i*stride : i*stride + window_size] for i in range(n_windows)])
    # return windows
    n_windows = (len(data) - window_size) // stride
    windows = np.array([data[i*stride : i*stride + window_size] for i in range(n_windows)])
    targets = np.array([target_data[i*stride + window_size] for i in range(n_windows)])
    return windows, targets

# # Function to create sliding windows
# def create_sliding_windows(data, window_size, stride=1):
#     n_windows = (len(data) - window_size) // stride + 1
#     windows = np.array([data[i*stride : i*stride + window_size] for i in range(n_windows)])
#     return windows

# Create sliding windows and targets for Actual Textile Exports
textile_windows_actual, y_reserves_actual = create_sliding_windows(
    df_sri_lanka_potential_pyts['Actual_Textile_Exports'].values/ 1e3,
    df_sri_lanka_potential_pyts['Actual_Reserves'].values/ 1e6,
    window_size,
    stride
)

# Create sliding windows and targets for Potential Textile Exports
textile_windows_potential, y_reserves_potential = create_sliding_windows(
    df_sri_lanka_potential_pyts['Predicted_Textile_Exports'].values/ 1e3,
    df_sri_lanka_potential_pyts['Predicted_Reserves'].values/ 1e6,
    window_size,
    stride
)

print("Textile Windows Actual Shape:", textile_windows_actual.shape)
print("Target Variable Actual Shape:", y_reserves_actual.shape)

print("Textile Windows Potential Shape:", textile_windows_potential.shape)
print("Target Variable Potential Shape:", y_reserves_potential.shape)

textile_windows_actual

textile_windows_potential

y_reserves_actual

y_reserves_potential

# Initialize ShapeletTransform
shapelet_transform = ShapeletTransform(
    n_shapelets=3,               # Number of shapelets to extract
    criterion='anova',           # Use 'anova' for regression
    window_sizes='auto',         # Automatically determine window sizes
    window_steps=None,           # Default step size
    remove_similar=True,         # Remove similar shapelets
    sort=False,                  # Do not sort shapelets by score
    verbose=0,                   # No verbosity
    random_state=42,             # For reproducibility
    n_jobs=-1                    # Utilize all available CPU cores
)

# Fit the ShapeletTransform and transform the data
X_shapelet_actual = shapelet_transform.fit_transform(textile_windows_actual, y_reserves_actual)
X_shapelet_potential = shapelet_transform.fit_transform(textile_windows_potential, y_reserves_actual)

print("Shapelet Features Shape:", X_shapelet_actual.shape)
print("Shapelet Features:")
print(X_shapelet_actual)

print("Shapelet Features Shape:", X_shapelet_potential.shape)
print("Shapelet Features:")
print(X_shapelet_potential)

# Initialize and evaluate the regression model
regressor_potential = LinearRegression()
regressor_actual = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor_potential, X_shapelet_potential, y_reserves_potential, cv=kf, scoring='r2')

print(f'Cross-validated R-squared scores with Shapelet Features: {cv_scores}')
print(f'Average R-squared: {np.mean(cv_scores):.4f}')

# Fit the model on the entire dataset using Shapelet features
regressor_potential.fit(X_shapelet_potential, y_reserves_potential)
regressor_actual.fit(X_shapelet_actual, y_reserves_actual)

# Predict reserves using the fitted model
y_pred_potential = regressor_potential.predict(X_shapelet_potential)
y_pred_actual = regressor_actual.predict(X_shapelet_actual)

# Extract the years corresponding to y_reserves
years = df_sri_lanka_potential_rfr['Year'].values[window_size:]

plt.figure(figsize=(12, 6))
plt.plot(years, y_reserves_actual, marker='o', label='Actual Reserves')
plt.plot(years, y_reserves_potential, marker='o', label='Potential Reserves')
plt.plot(years, y_pred_potential, marker='x', label='Predicted Potential Reserves (Shapelet Features)', linestyle='--')
plt.plot(years, y_pred_actual, marker='x', label='Predicted Actual Reserves (Shapelet Features)', linestyle='--')
plt.title('Actual vs. Predicted Reserves Using Shapelet Features')
plt.xlabel('Year')
plt.ylabel('Reserves (US$ Million)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.show()