import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from readability import Readability

# Load FKGL + Words Replaced results
mtcs = pd.read_csv('nsga2_wordnet_results.csv')
filtered_mtcs = mtcs[mtcs["Run"] == 25]

# Load Pareto solution text files
file_list = sorted(glob.glob(os.path.join("pareto_solutions", "run25_*.txt")))
def extract_number(file_path):
    match = re.search(r'run25_pareto_(\d+)', os.path.basename(file_path))
    return int(match.group(1)) if match else float('inf')
sorted_files = sorted(file_list, key=extract_number)

# Read paragraph contents from files
documents = []
for file_path in sorted_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        content = raw.split("\n\n", 1)[1] if "\n\n" in raw else ""
        documents.append(content.strip())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# PCA for Visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)


# Compute average FKGL per cluster
cluster_fkgls = []
for i in range(3):
    idx = np.where(clusters == i)[0]
    avg_fkgl = filtered_mtcs.iloc[idx]["Readability Score"].mean()
    cluster_fkgls.append((i, avg_fkgl))

# Sort clusters by FKGL (ascending)
sorted_clusters = sorted(cluster_fkgls, key=lambda x: x[1])

# Assign human-readable labels
cluster_labels = {
    sorted_clusters[0][0]: "Beginner",
    sorted_clusters[1][0]: "Intermediate",
    sorted_clusters[2][0]: "Advanced"
}

print("\n📘 Cluster Label Mapping:")
for cid, label in cluster_labels.items():
    print(f"Cluster {cid} → {label}")



# Silhouette Score (Clustering Validation)
sil_score = silhouette_score(X, clusters)
print(f"\n🧪 Silhouette Score (clustering quality): {sil_score:.3f}")

# Plot: Cluster Visualization
plt.figure(figsize=(10, 7))
cmap = get_cmap('tab10')
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap=cmap,
                      s=120, edgecolor='black', linewidth=0.6, alpha=0.9)
plt.title("TF-IDF + PCA Clustering of Pareto Text Variants", fontsize=16, weight='bold')
plt.xlabel("Principal Component 1", fontsize=13)
plt.ylabel("Principal Component 2", fontsize=13)
for i in range(len(documents)):
    words_replaced = filtered_mtcs.iloc[i]["Words Replaced"]
    cluster_id = clusters[i]
    label_name = cluster_labels[cluster_id]
    plt.annotate(f'{label_name[0]}: {int(words_replaced)}', (X_reduced[i, 0], X_reduced[i, 1]), fontsize=10, weight='bold', alpha=0.85)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/cluster_plot_paper.png")
plt.show()

# 💡 Centroid Text Identification
print("\n📌 Cluster Representative (Centroid) Texts:")
for label in range(3):
    indices = np.where(clusters == label)[0]
    cluster_vectors = X[indices].toarray()
    center = kmeans.cluster_centers_[label]
    dists = np.linalg.norm(cluster_vectors - center, axis=1)
    nearest_idx = indices[np.argmin(dists)]
    file_name = os.path.basename(sorted_files[nearest_idx])
    label_name = cluster_labels[label]
    file_name = os.path.basename(sorted_files[nearest_idx])
    print(f"\nCluster '{label_name}' representative:\nFile: {file_name}\nText:\n{documents[nearest_idx]}\n")

    with open(f"outputs/{label_name}_centroid.txt", "w", encoding="utf-8") as f:
        f.write(f"File: {file_name}\n\n{documents[nearest_idx]}")



# 🔍 Readability–Consistency Trade-off Plot
plt.figure(figsize=(7, 5))
scatter = plt.scatter(filtered_mtcs["Words Replaced"], filtered_mtcs["Readability Score"], c=clusters, cmap='Set1', s=100, edgecolors='black', alpha=0.85)
# Add proper legend using label names
unique_cids = sorted(np.unique(clusters))
handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_labels[cid], markerfacecolor=scatter.cmap(scatter.norm(cid)), markersize=10) for cid in unique_cids]
plt.legend(handles=handles, title='Reading Level', loc='upper right')
plt.xlabel("Words Replaced", fontsize=12)
plt.ylabel("Readability Score (FKGL)", fontsize=12)
plt.title("Readability vs Word Replacement", fontsize=14, weight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/readability_tradeoff_plot.png")
plt.show()

# 📈 Lexical Progression Analysis (Cluster Stats)
print("\n📊 Cluster-level FKGL & Word Replacement Averages:")
for label in range(3):
    indices = np.where(clusters == label)[0]
    avg_fkgl = filtered_mtcs.iloc[indices]["Readability Score"].mean()
    avg_wr = filtered_mtcs.iloc[indices]["Words Replaced"].mean()
    print(f"Cluster {label}: Readability={avg_fkgl:.2f}, Words Replaced={avg_wr:.2f}")


# Save cluster stats to LaTeX table
latex_table = r"""\begin{table}[h]
\centering
\caption{Cluster-wise Averages of Readability and Word Replacement}
\begin{tabular}{c|c|c}
\hline
\textbf{Cluster} & \textbf{Average FKGL} & \textbf{Average Words Replaced} \\
\hline
"""

for label in range(3):
    indices = np.where(clusters == label)[0]
    avg_fkgl = filtered_mtcs.iloc[indices]["Readability Score"].mean()
    avg_wr = filtered_mtcs.iloc[indices]["Words Replaced"].mean()
    latex_table += f"{label} & {avg_fkgl:.2f} & {avg_wr:.2f} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\label{tab:cluster_stats}
\end{table}"""

# Save LaTeX table to file
with open("outputs/cluster_stats_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("\n✅ LaTeX table saved to: outputs/cluster_stats_table.tex")

# Map cluster labels to sorted FKGL levels
cluster_fkgls = []
for i in range(3):
    idx = np.where(clusters == i)[0]
    avg_fkgl = filtered_mtcs.iloc[idx]["Readability Score"].mean()
    cluster_fkgls.append((i, avg_fkgl))

# Sort by FKGL to get labels
sorted_clusters = sorted(cluster_fkgls, key=lambda x: x[1])  # low to high FKGL
cluster_labels = {sorted_clusters[0][0]: "Beginner",
                  sorted_clusters[1][0]: "Intermediate",
                  sorted_clusters[2][0]: "Advanced"}

# Print mapping
print("\n📘 Cluster Label Mapping:")
for cid, label in cluster_labels.items():
    print(f"Cluster {cid} → {label}")

# Step 1: Compute average FKGL per cluster
cluster_fkgls = []
for i in range(3):
    idx = np.where(clusters == i)[0]
    avg_fkgl = filtered_mtcs.iloc[idx]["Readability Score"].mean()
    cluster_fkgls.append((i, avg_fkgl))

# Step 2: Sort clusters by FKGL (ascending)
sorted_clusters = sorted(cluster_fkgls, key=lambda x: x[1])

# Step 3: Assign labels
cluster_labels = {
    sorted_clusters[0][0]: "Beginner",
    sorted_clusters[1][0]: "Intermediate",
    sorted_clusters[2][0]: "Advanced"
}

print("\n📘 Cluster Label Mapping:")
for cid, label in cluster_labels.items():
    print(f"Cluster {cid} → {label}")

