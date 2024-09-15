import pandas as pd
import numpy as np

# Define countries, years, and economic indicators
countries = ['CountryA', 'CountryB', 'CountryC']
years = [2018, 2019, 2020]
indicators = ['GDP Growth', 'Inflation Rate', 'Unemployment Rate', 'Interest Rate', 'Exchange Rate']

# Initialize an empty list to store data
data = []

# Set a seed for reproducibility
np.random.seed(0)

# Generate sample data
for country in countries:
    for year in years:
        # Random values for economic indicators
        indicator_values = np.random.rand(len(indicators)) * 10  # Values between 0 and 10
        # Random value for textile earnings
        textile_earnings = np.random.rand() * 1000  # Values between 0 and 1000
        # Combine all data into a single row
        row = [country, year] + indicator_values.tolist() + [textile_earnings]
        data.append(row)

# Create a DataFrame
columns = ['Country', 'Year'] + indicators + ['Textile Earnings']
df = pd.DataFrame(data, columns=columns)

# Display the DataFrame
print("Sample Dataset:")
print(df)

# Calculate correlations between each indicator and textile earnings
print("\nCorrelation between Economic Indicators and Textile Earnings:")
for indicator in indicators:
    correlation = df[indicator].corr(df['Textile Earnings'])
    print(f"{indicator}: {correlation:.2f}")

# Sample Dataset:
#     Country  Year  GDP Growth  Inflation Rate  Unemployment Rate  Interest Rate  Exchange Rate  Textile Earnings
# 0  CountryA  2018    5.488135        7.151894           6.027634       5.448832       4.236548        645.894113
# 1  CountryA  2019    4.375872        8.917730           9.636628       3.834415       7.917250        528.894920
# 2  CountryA  2020    5.680446        9.255966           0.710361       0.871293       0.202184        832.619846
# 3  CountryB  2018    7.781568        8.700121           9.786183       7.991586       4.614794        780.529176
# 4  CountryB  2019    1.182744        6.399210           1.433533       9.446689       5.218483        414.661940
# 5  CountryB  2020    2.645556        7.742337           4.561503       5.684339       0.187898        617.635497
# 6  CountryC  2018    6.120957        6.169340           9.437481       6.818203       3.595079        437.031954
# 7  CountryC  2019    6.976312        0.602255           6.667667       6.706379       2.103826        128.926298
# 8  CountryC  2020    3.154284        3.637108           5.701968       4.386015       9.883738        102.044811


# Correlation between Economic Indicators and Textile Earnings:
# GDP Growth: 0.23
# Inflation Rate: 0.90
# Unemployment Rate: -0.10
# Interest Rate: -0.25
# Exchange Rate: -0.46
