# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Load dataset
df = pd.read_csv("iris.csv")

# Display first rows
print(df.head())


# Select numerical features
X = df.select_dtypes(include=['int64', 'float64'])

# Remove missing values
X = X.dropna()


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method to find optimal clusters
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# Plot Elbow Method
plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()


# Apply K-Means (choose optimal number after elbow graph)
kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(X_scaled)


# Add cluster labels
X["Cluster"] = clusters


# Reduce dimensions for visualization
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# Create scatter plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters,
    palette="viridis"
)

plt.title("Customer Segments using K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()


# Display cluster counts
print(X["Cluster"].value_counts())