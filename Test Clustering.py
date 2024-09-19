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
world_indicators_clean=world_indicators_clean[world_indicators_clean['Year'].isnull()==False]
world_indicators_clean['Country'] = world_indicators_df['Country']
world_indicators_clean['Year'] = world_indicators_df['Year']

# Standardize the entire dataset (mean = 0, variance = 1)
numeric_data = world_indicators_clean.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
standardized_data = scaler.fit_transform(numeric_data)

# Apply PCA to the entire dataset
pca = PCA(n_components=3)  # Choose the number of components based on explained variance
pca_data = pca.fit_transform(standardized_data)

# Add the PCA components back to the DataFrame
world_indicators_clean[['PCA1', 'PCA2', 'PCA3']] = pca_data

# Loop through each year and perform clustering on the PCA-transformed data
years = world_indicators_clean['Year'].unique()
clusters_by_year = {}

for year in years:
    # Filter data for the current year and create a copy
    year_data_clean = world_indicators_clean[world_indicators_clean['Year'] == year].copy()

    # Extract the PCA components for clustering
    pca_columns = year_data_clean[['PCA1', 'PCA2', 'PCA3']]

    # Apply K-means clustering on the PCA-transformed data
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pca_columns)

    # Assign the clusters using .loc on the copied DataFrame
    year_data_clean.loc[:, 'Cluster'] = kmeans.labels_

    # Save the results for this year
    clusters_by_year[year] = year_data_clean[['Country', 'Cluster']]

# Visualize Clustering for a specific year (e.g., 2010)
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