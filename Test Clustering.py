import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the World Development Indicators dataset
world_indicators_file = '/content/drive/MyDrive/RSU50 DataFiles/World_Development_Indicators.xlsx'
world_indicators_df = pd.read_excel(world_indicators_file)

# Rename columns for consistency
world_indicators_df.columns = ['Country', 'Year'] + list(world_indicators_df.columns[2:])

# Remove 'Hong Kong' from the dataset since lot of null values
world_indicators_df = world_indicators_df[world_indicators_df['Country'] != 'Hong Kong, China']

# Remove columns with more than 50% missing values across the entire dataset
threshold = len(world_indicators_df) * 0.99
world_indicators_clean = world_indicators_df.dropna(thresh=threshold, axis=1)

# Replace remaining null values with the mean for each country
world_indicators_clean = world_indicators_clean.groupby('Country').transform(lambda x: x.fillna(x.mean()))
world_indicators_clean.head()

world_indicators_clean=world_indicators_clean[world_indicators_clean['Year'].isnull()==False]

# Merge back with 'Country' and 'Year' columns after transforming
world_indicators_clean['Country'] = world_indicators_df['Country']
world_indicators_clean['Year'] = world_indicators_df['Year']
world_indicators_clean.head()

# Loop through each year and perform clustering
years = world_indicators_clean['Year'].unique()
clusters_by_year = {}

for year in years:
    # Filter data for the current year
    year_data_clean = world_indicators_clean[world_indicators_clean['Year'] == year]

    # Remove any remaining columns with NaN values
    #year_data_clean = year_data.dropna(axis=1)

    # Extract the numeric columns for clustering
    numeric_data = year_data_clean.select_dtypes(include=['float64', 'int64'])

    # Standardize the data (mean = 0, variance = 1)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)

    # Apply K-means clustering (set number of clusters, e.g., 3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(standardized_data)

    # Store the clusters for each country
    year_data_clean['Cluster'] = kmeans.labels_

    # Save the results for this year
    clusters_by_year[year] = year_data_clean[['Country', 'Cluster']]

# Visualize Clustering for a specific year (e.g., 2020)
year_to_visualize = 2010
if year_to_visualize in clusters_by_year:
    plt.figure(figsize=(10, 6))
    clusters = clusters_by_year[year_to_visualize]['Cluster'].value_counts()
    clusters.plot(kind='bar')
    plt.title(f'Country Clusters for {year_to_visualize}')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    plt.show()
else:
    print(f'No data for the year {year_to_visualize}')

