import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    _HAS_MATPLOTLIB = False


def plot_clusters(embeddings, labels, out_file=None, method="pca", random_state=42):
    """Create a 2D cluster plot using PCA or t-SNE."""
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install it with 'pip install matplotlib'.")

    if method not in {"pca", "tsne"}:
        raise ValueError("plot_clusters method must be 'pca' or 'tsne'.")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
    else:
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.75, s=50)
    plt.title(f"Cluster visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.gca().add_artist(legend1)
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
        print(f"Saved cluster plot to {out_file}")
    else:
        plt.show()

    plt.close()


def cluster_questions(csv_file_path, num_clusters=4):
    # 1. Load data
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # file not found, print an error message and exit
        print(f"Error: Could not find '{csv_file_path}'")
        return
    
    # Drop rows with empty generations
    df = df[df["Generated_Questions"].notna() & (df["Generated_Questions"] != "")]
    
    print("Loading embedding model (this may take a moment on the first run)...")
    # 2. Generate Sentence Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = df["Generated_Questions"].tolist()
    embeddings = model.encode(sentences)
    
    # 3. Perform K-Means Clustering
    print(f"Clustering questions into {num_clusters} distinct themes...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    df["Cluster"] = labels

    # 3.5 Create a 2D visualization of the clusters
    try:
        plot_clusters(embeddings, labels, method="pca")
    except Exception as e:
        print(f"Warning: failed to create cluster plot: {e}")
    
    print("\n==================================================")
    # 4. Analyze and Print the Clusters
    print("           THEMATIC CLUSTER ANALYSIS              ")
    print("==================================================\n")
    
    for cluster_id in range(num_clusters):
        cluster_data = df[df["Cluster"] == cluster_id]
        
        print(f"--- CLUSTER #{cluster_id} ---")
        print(f"Total Questions in Cluster: {len(cluster_data)}")
        
        # Show a few sample questions from this cluster to see what the theme is
        print("\nSample Questions:")
        samples = cluster_data["Generated_Questions"].head(3).tolist()
        for sample in samples:
            # Print just the first line/question as a snippet
            snippet = sample.split('\n')[0] if '\n' in sample else sample
            print(f"  - \"{snippet[:80]}...\"")
            
        # Count which demographics are most prevalent in this cluster
        print("\nDemographic Distribution in this Cluster:")
        demo_counts = Counter(cluster_data["Demographic"]) if "Demographic" in cluster_data.columns else Counter()
        for demo, count in demo_counts.most_common():
            percentage = (count / len(cluster_data)) * 100
            print(f"  * {demo}: {count} times ({percentage:.1f}%)")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    cluster_questions("demographic_bias_results.csv", num_clusters=4)