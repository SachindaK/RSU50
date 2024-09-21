import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the World Development Indicators dataset
world_indicators_file = '/content/drive/MyDrive/RSU50 DataFiles/World_Development_Indicators.xlsx'
world_indicators_df = pd.read_excel(world_indicators_file)

# Rename columns for consistency
world_indicators_df.columns = ['Country', 'Year'] + list(world_indicators_df.columns[2:])

world_indicators_df.shape

# Remove indicators in local currency
world_indicators_df = world_indicators_df.drop(columns=[col for col in world_indicators_df.columns if 'LCU' in col])

# Define a list of keywords that are commonly associated with economic indicators
economic_keywords = ['GDP ', 'inflation', 'exports']

# Create a boolean mask that checks if any keyword appears in the column names
economic_columns_mask = world_indicators_df.columns.str.contains('|'.join(economic_keywords), case=False)

# Filter the columns based on the economic indicators
economic_indicators_df = world_indicators_df.loc[:, economic_columns_mask].copy()

# Add back 'Country' and 'Year' columns to keep the structure using .loc to avoid the warning
economic_indicators_df['Country'] = world_indicators_df.loc[:, 'Country']
economic_indicators_df['Year'] = world_indicators_df.loc[:, 'Year']

# Remove columns with more than 10% missing values across the entire dataset
threshold = len(economic_indicators_df) * 0.9
world_indicators_clean = economic_indicators_df.dropna(thresh=threshold, axis=1)

# Replace remaining null values with the mean for each country
# world_indicators_clean = world_indicators_clean.groupby('Country').transform(lambda x: x.fillna(x.mean()))

# Forward fill missing values with previous years' data, then backward fill for any remaining gaps
world_indicators_clean = world_indicators_clean.groupby('Country').ffill().bfill()
world_indicators_clean=world_indicators_clean[world_indicators_clean['Year'].isnull()==False]
world_indicators_clean['Country'] = economic_indicators_df['Country']
world_indicators_clean['Year'] = economic_indicators_df['Year']

# Drop columns with missing values across all rows for a given country (e.g., no valid data to propagate forward or backward)
world_indicators_clean = world_indicators_clean.dropna(axis=1)

world_indicators_clean.head()

# Select numeric columns (excluding 'Country' and 'Year')
numeric_columns = world_indicators_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['Country', 'Year']]

"""#Correlation of indicators"""

corr_matrix = world_indicators_clean[numeric_columns].corr()

# Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Economic Indicators')
# plt.show()

# Define a threshold for removing highly correlated variables
threshold = 0.9

# Find index of feature columns with correlation greater than the threshold
to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]

# Remove highly correlated columns
reduced_data = world_indicators_clean.drop(columns=to_drop)

# # Now apply clustering to the reduced dataset
# kmeans = KMeans(n_clusters=3, random_state=42)
# reduced_data['Cluster'] = kmeans.fit_predict(reduced_data.select_dtypes(include=[np.number]))

# # Optionally, you can visualize the clustering results or analyze cluster centroids
# print(reduced_data.head())

corr_matrix = world_indicators_clean[numeric_columns].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Economic Indicators')
plt.show()

# Define a threshold for removing highly correlated variables
threshold = 0.9

# Create a set to hold the names of columns to drop
to_drop = set()

# Iterate through the correlation matrix to find columns to drop
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            to_drop.add(colname)

len(to_drop)

# Reduce highly correlated columns
reduced_data = world_indicators_clean[numeric_columns].drop(columns=to_drop)
reduced_data

# Standardize the data across all years
scaler = StandardScaler()
standardized_data = scaler.fit_transform(reduced_data)

# Apply PCA on the entire dataset
pca = PCA(n_components=3)
pca_data = pca.fit_transform(standardized_data)

# Apply K-Means clustering on the entire dataset
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(pca_data)

# Assign clusters to each data point
world_indicators_clean['Cluster'] = kmeans.labels_

# Prepare the cluster tracking DataFrame
cluster_assignments = world_indicators_clean[['Country', 'Year', 'Cluster']]

# Pivot the cluster assignments to track clusters over years
cluster_tracking_pivot = cluster_assignments.pivot(index='Country', columns='Year', values='Cluster')

# Fill missing values if necessary
cluster_tracking_pivot = cluster_tracking_pivot.fillna(-1).astype(int)

# Display the pivot table
#print(cluster_tracking_pivot.head())

# Visualize the cluster transitions over time
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_tracking_pivot, cmap="viridis", annot=True, cbar=True, linewidths=.5)
plt.title("Country Cluster Transitions Over Time (Consistent Clusters)")
plt.ylabel("Country")
plt.xlabel("Year")
plt.show()

"""# Check cluster centroids to identify clusters"""

columns_to_include = ['Cluster'] + reduced_data.columns.tolist()

# Create a DataFrame with the necessary columns
cluster_data = world_indicators_clean[columns_to_include].copy()
cluster_data

# Calculate the centroids in the original feature space
cluster_centroids = cluster_data.groupby('Cluster')[cluster_data.select_dtypes(include=['float64', 'int64']).columns.tolist()].mean()

# Display the centroids
print("Cluster Centroids in Original Feature Space:")
cluster_centroids

# Define cluster labels based on interpretation
cluster_labels = {
    2: 'H',
    0: 'M',
    1: 'L'
}

# Map the cluster labels to the data
cluster_data['Cluster Label'] = cluster_data['Cluster'].map(cluster_labels)

world_indicators_clean['Cluster Label'] = world_indicators_clean['Cluster'].map(cluster_labels)

# # Transpose the centroids for easier plotting
# centroids_transposed = cluster_centroids.T

# # Plotting the centroids
# centroids_transposed.plot(kind='bar', figsize=(12, 8))
# plt.title('Cluster Centroids of Economic Indicators')
# plt.xlabel('Economic Indicators')
# plt.ylabel('Average Value')
# plt.legend(title='Cluster')
# plt.show()

plt.figure(figsize=(30, 6))
sns.heatmap(cluster_centroids, annot=True, fmt=".2f", cmap='viridis')
plt.title('Heatmap of Cluster Centroids')
plt.xlabel('Economic Indicators')
plt.ylabel('Cluster')
plt.show()

# Prepare the cluster tracking DataFrame with labels
cluster_assignments = world_indicators_clean[['Country', 'Year', 'Cluster Label']]

# Pivot the cluster assignments to get the cluster tracking over years
cluster_tracking_pivot = cluster_assignments.pivot(index='Country', columns='Year', values='Cluster Label')

# Display the pivot table
# print(cluster_tracking_pivot.head())

from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# Define the custom color map for 'H', 'L', and 'M'
cmap = ListedColormap(['#77dd77', '#ffb3b0', '#fff4b3'])  # Green for 'H'', Yellow for 'M', Red for 'L

# Define your labels
labels = ['H', 'L', 'M']

# Create label encoding
le = LabelEncoder()
le.fit(labels)

# Apply encoding to the data
cluster_tracking_encoded = cluster_tracking_pivot.applymap(lambda x: le.transform([x])[0] if pd.notnull(x) else -1)

# Create a heatmap with the custom colormap
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_tracking_encoded, cmap=cmap, annot=cluster_tracking_pivot, fmt='', cbar=True, linewidths=.5, cbar_kws={'ticks': [0, 1, 2]})

# Set tick labels for the color bar to 'H', 'L', 'M'
cbar = plt.gca().collections[0].colorbar
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(labels)

# Add plot labels and title
plt.title("Country Cluster Transitions Over Time (Consistent Clusters)")
plt.ylabel("Country")
plt.xlabel("Year")

# Display the plot
plt.show()

"""# PCA Biplot"""

# Create a DataFrame for principal components
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2', 'PC3'])
pca_df['Country'] = world_indicators_clean['Country']
pca_df['Year'] = world_indicators_clean['Year']
pca_df['Cluster'] = world_indicators_clean['Cluster']

# Get the loadings (coefficients of the linear combinations)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a biplot
def biplot(score, coeff, labels=None, clusters=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(xs, ys, c=clusters, cmap='viridis', alpha=0.7)

    # Plot variable vectors
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0]*5, coeff[i, 1]*5, color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0]*5.2, coeff[i, 1]*5.2, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0]*5.2, coeff[i, 1]*5.2, labels[i], color='g', ha='center', va='center')

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Biplot")
    plt.grid()
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Get the labels for the variables
variable_labels = numeric_columns

# Prepare cluster labels (ensure consistent labeling)
clusters = world_indicators_clean['Cluster']

# Call the biplot function
biplot(pca_data, loadings, labels=variable_labels, clusters=clusters)
