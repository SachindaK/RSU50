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
threshold = len(economic_indicators_df) * 0.99
world_indicators_clean = economic_indicators_df.dropna(thresh=threshold, axis=1)

# Replace remaining null values with the mean for each country
world_indicators_clean = world_indicators_clean.groupby('Country').transform(lambda x: x.fillna(x.mean()))
world_indicators_clean=world_indicators_clean[world_indicators_clean['Year'].isnull()==False]
world_indicators_clean['Country'] = economic_indicators_df['Country']
world_indicators_clean['Year'] = economic_indicators_df['Year']

world_indicators_clean.head()

# Standardize the entire dataset (mean = 0, variance = 1)
numeric_data = world_indicators_clean.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
standardized_data = scaler.fit_transform(numeric_data)

# Apply PCA to the entire dataset
pca = PCA(n_components=3)
pca_data = pca.fit_transform(standardized_data)

# Add the PCA components back to the DataFrame
world_indicators_clean[['PCA1', 'PCA2', 'PCA3']] = pca_data

# Initialize a DataFrame to store cluster assignments over years
cluster_tracking = pd.DataFrame()

# Loop through each year and perform clustering on the PCA-transformed data
years = world_indicators_clean['Year'].unique()

for year in years:
    # Filter data for the current year and create a copy
    year_data_clean = world_indicators_clean[world_indicators_clean['Year'] == year].copy()

    # Extract the PCA components for clustering
    pca_columns = year_data_clean[['PCA1', 'PCA2', 'PCA3']]

    # Apply K-means clustering on the PCA-transformed data
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pca_columns)

    # Assign the clusters using .loc on the copied DataFrame
    year_data_clean['Cluster'] = kmeans.labels_

    # Store cluster assignments in the tracking DataFrame
    year_data_clean = year_data_clean[['Country', 'Year', 'Cluster']]
    cluster_tracking = pd.concat([cluster_tracking, year_data_clean])

# Pivot the data so that we can track clusters over time (rows = countries, columns = years)
cluster_tracking_pivot = cluster_tracking.pivot(index='Country', columns='Year', values='Cluster')

# Fill any missing values in the pivot table (in case any countries are missing for specific years)
cluster_tracking_pivot = cluster_tracking_pivot.fillna(-1).astype(int)  # Use -1 for missing years

# Display the pivot table for cluster tracking over time
print(cluster_tracking_pivot.head())

# Visualize transitions using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_tracking_pivot, cmap="coolwarm", annot=True, cbar=True)
plt.title("Country Cluster Transitions Over Time")
plt.ylabel("Country")
plt.xlabel("Year")
plt.show()

# Prepare data for the Sankey diagram
sources = []
targets = []
values = []

# Loop through years to create transitions between clusters
for i in range(len(years) - 1):
    year1 = years[i]
    year2 = years[i + 1]
    
    for cluster_from in range(3):  # Assuming 3 clusters
        for cluster_to in range(3):
            # Count transitions from cluster_from in year1 to cluster_to in year2
            transition_count = cluster_tracking_pivot[(cluster_tracking_pivot[year1] == cluster_from) & 
                                                      (cluster_tracking_pivot[year2] == cluster_to)].shape[0]
            sources.append(cluster_from + i * 3)
            targets.append(cluster_to + (i + 1) * 3)
            values.append(transition_count)

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=[f"{year}: Cluster {i}" for year in years for i in range(3)]
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
))

fig.update_layout(title_text="Country Transitions Between Clusters Over Time", font_size=10)
fig.show()
