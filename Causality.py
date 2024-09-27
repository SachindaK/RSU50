
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

# Load textile export and reserve data
textile_exports_df = pd.read_excel('/content/drive/MyDrive/RSU50 DataFiles/Textile_Exports.xlsx')  # Load textile export data
reserve_df = pd.read_excel('/content/drive/MyDrive/RSU50 DataFiles/Reserve.xlsx')  # Load reserve data

textile_exports_df.head()

reserve_df.head()

# Merge datasets on 'Country' and 'Year', and only include years 1990-2018
merged_data = pd.merge(textile_exports_df, reserve_df, on=['Country', 'Year'])
merged_data = merged_data[(merged_data['Year'] >= 1990) & (merged_data['Year'] <= 2018)]

merged_data.head()

merged_data.info()

# Define a function to perform Granger causality test for each country
def granger_test_for_country(country_data, max_lag=3):
    # Ensure there are no missing values in the data for the causality test
    country_data = country_data.dropna()

    # Select the relevant columns for textile exports and reserves
    causality_data = country_data[['Total reserves (includes gold, current US$) [FI.RES.TOTL.CD]', 'Textiles and Clothing Export (US$ Thousand)']]

    # Perform the Granger causality test
    result = grangercausalitytests(causality_data, max_lag, verbose=True)

    return result

# # Loop over each country and apply the Granger causality test
# countries = merged_data['Country'].unique()
# granger_results = {}

# for country in countries:
#     print(f"\nGranger Causality Test for {country}")

#     # Filter data for the specific country
#     country_data = merged_data[merged_data['Country'] == country]

#     # Perform the Granger causality test for the country
#     granger_results[country] = granger_test_for_country(country_data)

# # granger_results will contain the causality test results for each country

# List of countries to be analyzed
countries_to_analyze = ['Cambodia', 'China', 'India', 'Korea, Rep.', 'Malaysia', 'Singapore', 'Thailand', 'Vietnam', 'Sri Lanka']

# Define a function to perform Granger causality test for a country and return p-values
def granger_test_for_country(country_data, max_lag=3):
    # Ensure there are no missing values in the data for the causality test
    country_data = country_data.dropna()
    
    # Select the relevant columns for textile exports and reserves
    causality_data = country_data[['Total reserves (includes gold, current US$) [FI.RES.TOTL.CD]', 'Textiles and Clothing Export (US$ Thousand)']]
    
    # Store p-values for each lag
    p_values = []

    # Perform Granger causality test and extract p-values
    result = grangercausalitytests(causality_data, max_lag, verbose=False)
    for lag, res in result.items():
        p_value = res[0]['ssr_ftest'][1]  # Extract the p-value from the ssr F-test
        p_values.append((lag, p_value))
    
    return p_values

# Plot p-values for each country
for country in countries_to_analyze:
    print(f"\nGranger Causality Test for {country}")
    
    # Filter the data for the specific country
    country_data = merged_data[merged_data['Country'] == country]
    
    # Perform Granger causality test and get p-values
    try:
        p_values = granger_test_for_country(country_data)
        
        # Convert p-values to a DataFrame for easier plotting
        p_values_df = pd.DataFrame(p_values, columns=['Lag', 'p-value'])
        
        # Plot the p-values
        plt.figure(figsize=(8, 5))
        plt.plot(p_values_df['Lag'], p_values_df['p-value'], marker='o', linestyle='--', color='b')
        plt.axhline(y=0.05, color='r', linestyle='-', label='Significance Level (0.05)')
        plt.title(f'Granger Causality p-values for {country}')
        plt.xlabel('Lag')
        plt.ylabel('p-value')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError as e:
        print(f"Error with {country}: {e}")

# Plot p-values for multiple countries in a single graph
plt.figure(figsize=(10, 6))

for country in countries_to_analyze:
    print(f"\nGranger Causality Test for {country}")
    
    # Filter the data for the specific country
    country_data = merged_data[merged_data['Country'] == country]
    
    # Perform Granger causality test and get p-values
    try:
        p_values = granger_test_for_country(country_data)
        
        # Convert p-values to a DataFrame for easier plotting
        p_values_df = pd.DataFrame(p_values, columns=['Lag', 'p-value'])
        
        # Plot the p-values for the country
        plt.plot(p_values_df['Lag'], p_values_df['p-value'], marker='o', linestyle='--', label=country)
    
    except ValueError as e:
        print(f"Error with {country}: {e}")

# Add a horizontal line for the 0.05 significance level
plt.axhline(y=0.05, color='r', linestyle='-', label='Significance Level (0.05)')

# Add titles and labels
plt.title('Granger Causality p-values for Multiple Countries')
plt.xlabel('Lag')
plt.ylabel('p-value')

# Display legend
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# Display grid and show the plot
plt.grid(True)
plt.show()

# Loop over the countries
for country in countries_to_analyze:
    print(f"\nGranger Causality Test for {country}")

    # Filter the data for the specific country
    country_data = merged_data[merged_data['Country'] == country]

    # Perform Granger causality test for the country
    try:
        granger_test_for_country(country_data)
    except ValueError as e:
        print(f"Error with {country}: {e}")

# Get SL data
sl_data = merged_data[merged_data['Country'] == 'Sri Lanka']
sl_data.info()

sl_data.set_index('Year', inplace=True)

# Define rolling window parameters
window_size = 10
max_lag = 1  # Adjust based on lag selection criteria
p_values = []

# Perform rolling window Granger causality tests
for start_year in range(sl_data.index.min(), sl_data.index.max() - window_size + 2):
    end_year = start_year + window_size - 1
    window_data = sl_data.loc[start_year:end_year]

    if len(window_data) < window_size:
        continue  # Skip if not enough data

    # Prepare data for Granger causality test
    test_data = window_data[['Total reserves (includes gold, current US$) [FI.RES.TOTL.CD]', 'Textiles and Clothing Export (US$ Thousand)']].dropna()

    # Perform the test
    try:
        result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        # Extract p-value for the desired lag (e.g., max_lag)
        p_value = result[max_lag][0]['ssr_ftest'][1]
    except Exception as e:
        p_value = None  # Handle exceptions (e.g., singular matrix)

    p_values.append({'End_Year': end_year, 'p_value': p_value})

# Convert results to DataFrame
p_values_df = pd.DataFrame(p_values)

p_values_df

# Plot p-values over time
plt.figure(figsize=(10, 6))
plt.plot(p_values_df['End_Year'], p_values_df['p_value'], marker='o')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
plt.title('Rolling Window Granger Causality p-values for Sri Lanka')
plt.xlabel('End Year of Window')
plt.ylabel('p-value')
plt.legend()
plt.grid(True)
plt.show()