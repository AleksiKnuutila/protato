import tomotopy as tp
from tqdm import tqdm

def run_lda_model(texts, n_topics=None, min_topic_size=5):
    # Create and train the LDA model
    lda_model = tp.LDAModel(k=n_topics if n_topics is not None else 0, min_cf=min_topic_size)
    
    # Add documents to the model
    for text in tqdm(texts, desc="Adding documents to LDA model"):
        lda_model.add_doc(text.split())
    
    print("Training LDA model...")
    for _ in tqdm(range(100), desc="LDA training iterations"):
        lda_model.train(1)
    
    # Get topic assignments
    topics = []
    for idx in tqdm(range(len(texts)), desc="Inferring topics"):
        doc_topics = lda_model.infer(lda_model.docs[idx])[0]  # Use the document object stored in the model
        most_likely_topic = max(range(len(doc_topics)), key=lambda i: doc_topics[i])
        topics.append(most_likely_topic)
    
    return topics, lda_model