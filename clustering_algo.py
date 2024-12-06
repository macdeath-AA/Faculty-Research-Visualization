import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('faculty_list.csv')

# Remove professors with empty research areas
df = df[df['Research Areas'].notna() & (df['Research Areas'] != '[]')]

# Preprocess and vectorize the research areas
tfidf = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1,3))
tfidf_matrix = tfidf.fit_transform(df['Research Areas'])

# Clustering
n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())

# Function to get top terms for each cluster
def get_top_terms(tfidf_matrix, feature_names, cluster_labels, n_terms=2):
    cluster_terms = {}
    for cluster in range(n_clusters):
        cluster_docs = tfidf_matrix[cluster_labels == cluster]
        tfidf_sums = cluster_docs.sum(axis=0).A1
        top_indices = tfidf_sums.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        cluster_terms[cluster] = ', '.join(top_terms)
    return cluster_terms

# Get cluster names
feature_names = tfidf.get_feature_names_out()
cluster_names = get_top_terms(tfidf_matrix, feature_names, cluster_labels)

# Create a circular layout
theta = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
x = np.cos(theta)
y = np.sin(theta)

# Create the scatter plot
plt.figure(figsize=(5, 5))
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    cluster_points = tfidf_matrix[cluster_labels == i].toarray()
    
    # Project points onto the circular layout
    projected_x = np.random.normal(x[i], 0.1, size=cluster_points.shape[0])
    projected_y = np.random.normal(y[i], 0.1, size=cluster_points.shape[0])
    
    plt.scatter(projected_x, projected_y, c=[colors[i]], label=f'{cluster_names[i]}')


plt.title("Research Areas Cluster Visualization", fontsize=16)
plt.legend()
plt.show()