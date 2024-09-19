#Approch 2 
#   - standardize_and_apply_pca per country
#   - select only direct economic indicators
#   - Fit K-Means to the entire dataset across all years to create a consistent set of clusters.
#   - Assign cluster labels for each year using the same clustering model.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Load the World Development Indicators dataset
world_indicators_file = '/content/drive/MyDrive/RSU50 DataFiles/World_Development_Indicators.xlsx'
world_indicators_df = pd.read_excel(world_indicators_file)

# Rename columns for consistency
world_indicators_df.columns = ['Country', 'Year'] + list(world_indicators_df.columns[2:])

# Remove 'Hong Kong' from the dataset
world_indicators_df = world_indicators_df[world_indicators_df['Country'] != 'Hong Kong, China']

# Define a list of keywords that are commonly associated with economic indicators
economic_keywords = ['GDP', 'inflation', 'trade', 'employment', 'unemployment', 'business', 'expenditure',
                     'income', 'investment', 'exports', 'imports', 'interest', 'monetary', 'price index', 'balance of payments']

# Create a boolean mask that checks if any keyword appears in the column names
economic_columns_mask = world_indicators_df.columns.str.contains('|'.join(economic_keywords), case=False)

# Filter the columns based on the economic indicators
economic_indicators_df = world_indicators_df.loc[:, economic_columns_mask].copy()

# Add back 'Country' and 'Year' columns to keep the structure using .loc to avoid the warning
economic_indicators_df['Country'] = world_indicators_df.loc[:, 'Country']
economic_indicators_df['Year'] = world_indicators_df.loc[:, 'Year']

# Display the filtered economic indicators
economic_indicators_df.head()

# Remove columns with more than 50% missing values across the entire dataset
threshold = len(economic_indicators_df) * 0.95
world_indicators_clean = economic_indicators_df.dropna(thresh=threshold, axis=1)

# Replace remaining null values with the mean for each country
#world_indicators_clean = world_indicators_clean.groupby('Country').transform(lambda x: x.fillna(x.mean()))
# Forward fill missing values with previous years' data, then backward fill for any remaining gaps
world_indicators_clean = world_indicators_clean.groupby('Country').ffill().bfill()
world_indicators_clean=world_indicators_clean[world_indicators_clean['Year'].isnull()==False]
world_indicators_clean['Country'] = economic_indicators_df['Country']
world_indicators_clean['Year'] = economic_indicators_df['Year']
#Drop columns with missing values across all rows for a given country (e.g., no valid data to propagate forward or backward)
world_indicators_clean = world_indicators_clean.dropna(axis=1)

world_indicators_clean.head()

# Define a function to standardize and apply PCA for each country
def standardize_and_apply_pca(group):
    scaler = StandardScaler()
    
    # Select only numeric columns (economic indicators)
    numeric_columns = group.select_dtypes(include=['float64', 'int64']).columns
    
    # Standardize the data for the country
    standardized_data = scaler.fit_transform(group[numeric_columns])
    
    # Apply PCA
    pca = PCA(n_components=3)  # Adjust the number of components based on your needs
    pca_data = pca.fit_transform(standardized_data)
    
    # Add the PCA components back to the DataFrame
    pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2', 'PCA3'], index=group.index)
    
    # Return the group with the PCA components included
    return pd.concat([group, pca_df], axis=1)

# Apply the function to each country group using groupby and apply
world_indicators_with_pca = world_indicators_clean.groupby('Country').apply(standardize_and_apply_pca).reset_index(drop=True)

# Display the DataFrame with the PCA components added
world_indicators_with_pca.head()

# Initialize a DataFrame to store cluster assignments over years
cluster_tracking = pd.DataFrame()

# Extract PCA components for all countries and all years
pca_columns = world_indicators_with_pca[['PCA1', 'PCA2', 'PCA3']]

# Fit K-Means on the entire dataset (across all years) to create consistent clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(pca_columns)

# Assign the clusters based on the fitted model
world_indicators_with_pca['Cluster'] = kmeans.labels_

# Initialize a DataFrame to track cluster assignments over years
cluster_tracking = world_indicators_with_pca[['Country', 'Year', 'Cluster']]

# Pivot the data so that we can track clusters over time (rows = countries, columns = years)
cluster_tracking_pivot = cluster_tracking.pivot(index='Country', columns='Year', values='Cluster')

# Fill any missing values in the pivot table (in case any countries are missing for specific years)
cluster_tracking_pivot = cluster_tracking_pivot.fillna(-1).astype(int)  # Use -1 for missing years

# Display the pivot table for cluster tracking over time
print(cluster_tracking_pivot.head())

# Visualize the pivoted data using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_tracking_pivot, cmap="coolwarm", annot=True, cbar=True, linewidths=.5)
plt.title("Country Cluster Transitions Over Time (Consistent Cluster Definition)")
plt.ylabel("Country")
plt.xlabel("Year")
plt.show()
