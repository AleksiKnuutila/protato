from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def run_topic_model(texts, embeddings, n_topics, min_topic_size):
    # Configure UMAP
    umap_model = UMAP(
        n_neighbors=15,      # Reduced for more local structure
        n_components=5,      # Keep this to preserve structure
        min_dist=0.01,      # Reduced for tighter clusters
        metric='cosine',
        random_state=42
    )
    
    # Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,          # Reduced for more clusters
        min_samples=5,               # Reduced for less conservative clustering
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Run BERTopic
    print(f"\nRunning BERTopic with n_topics={n_topics}")
    topic_model = BERTopic(
        n_gram_range=(1, 2),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
#        nr_topics=33
    )
    
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    return topics, topic_model  # Make sure we return both values