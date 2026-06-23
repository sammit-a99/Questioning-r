import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

def cluster_questions(csv_file_path, num_clusters=4):
    # 1. Load data
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
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
    df["Cluster"] = kmeans.fit_predict(embeddings)
    
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
        demo_counts = Counter(cluster_data["Demographic"])
        for demo, count in demo_counts.most_common():
            percentage = (count / len(cluster_data)) * 180
            print(f"  * {demo}: {count} times")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    cluster_questions("demographic_bias_results.csv", num_clusters=4)