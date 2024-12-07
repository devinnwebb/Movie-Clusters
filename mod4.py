import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "movies.csv" 
data = pd.read_csv(file_path)

# Data Cleaning
# Remove rows with missing values in relevant columns
data_cleaned = data.dropna(subset=["genre", "rating", "votes", "budget", "gross", "runtime"])

# Convert genre into one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(data_cleaned[["genre"]])
genre_columns = encoder.get_feature_names_out(["genre"])
genre_df = pd.DataFrame(genre_encoded, columns=genre_columns)

# Normalize numerical columns
scaler = StandardScaler()
numerical_columns = ["votes", "budget", "gross", "runtime"]
scaled_numerical = scaler.fit_transform(data_cleaned[numerical_columns])
scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)

# Combine numerical and genre features
cluster_data = pd.concat([scaled_df, genre_df], axis=1)

# Determine the optimal number of clusters (k) using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Select k based on the Elbow Method (adjust k as needed)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data_cleaned["cluster"] = kmeans.fit_predict(cluster_data)

# Silhouette Score for validation
silhouette_avg = silhouette_score(cluster_data, data_cleaned["cluster"])
print(f"Silhouette Score for k={optimal_k}: {silhouette_avg}")

# Analyze Clusters
# Select only numeric columns for aggregation
numeric_cols = data_cleaned.select_dtypes(include=["number"]).columns
cluster_summary = data_cleaned.groupby("cluster")[numeric_cols].mean()
print(cluster_summary)


# Visualize Clusters
sns.pairplot(data_cleaned, hue="cluster", vars=numerical_columns, palette="tab10")
plt.show()

# Example Data Points in Clusters
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(data_cleaned[data_cleaned["cluster"] == cluster][["name", "rating", "genre"]].head(2))

# Save results
data_cleaned.to_csv("clustered_movies.csv", index=False)
