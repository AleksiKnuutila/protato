import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tomotopy as tp
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
from biterm.btm import oBTM
from gsdmm.gsdmm import MovieGroupProcess

# Import functions from other files
from extract_summaries import extract_summaries
from embed_text import embed_text
from topic_model import run_topic_model
from lda_model import run_lda_model
from check_consistency import calculate_nmi

def truncate_event_type(event_type, frequencies, threshold=5):
    parts = str(event_type).split('.')
    
    # First round of truncation
    if frequencies[event_type] < threshold and len(parts) > 1:
        truncated = '.'.join(parts[:-1])
        return truncated
    
    return event_type

def double_truncate_event_types(df, threshold=5):
    # First round of truncation
    event_type_frequencies = df['Event Type'].value_counts()
    df['Truncated_Event_Type'] = df['Event Type'].apply(lambda x: truncate_event_type(x, event_type_frequencies, threshold))
    
    # Recalculate frequencies after first truncation
    truncated_frequencies = df['Truncated_Event_Type'].value_counts()
    
    # Second round of truncation
    df['Truncated_Event_Type'] = df['Truncated_Event_Type'].apply(lambda x: truncate_event_type(x, truncated_frequencies, threshold))
    
    return df

def truncate_event_type_last_part(event_type):
    parts = str(event_type).split('.')
    if parts[-1].lower() == 'n/a' and len(parts) > 1:
        return parts[-2]
    return parts[-1]

def truncate_event_types_last_part(df):
    df['Truncated_Event_Type'] = df['Event Type'].apply(truncate_event_type_last_part)
    return df

def calculate_purity(true_labels, predicted_labels):
    # Convert labels to numeric format
    le = LabelEncoder()
    
    # Fit the encoder on both true and predicted labels to ensure all labels are known
    all_labels = list(set(true_labels) | set(predicted_labels))
    le.fit(all_labels)
    
    true_labels_encoded = le.transform(true_labels)
    predicted_labels_encoded = le.transform(predicted_labels)
    
    cm = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
    
    # Purity
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    
    # Inverse Purity
    inverse_purity = np.sum(np.amax(cm, axis=1)) / np.sum(cm)
    
    # Harmonic mean of Purity and Inverse Purity
    if purity + inverse_purity > 0:
        harmonic_mean = 2 * (purity * inverse_purity) / (purity + inverse_purity)
    else:
        harmonic_mean = 0
    
    return harmonic_mean

def run_benchmark(dataset='rams', use_summaries=True, use_lda=False, use_ptm=False, use_btm=False, use_gsdmm=False, use_sentences=False, n_topics=20, min_topic_size=5, truncation_method='last_part', summary_length=5, use_generated_summary=False):
    # Load the data based on dataset parameter
    if dataset == 'rams':
        df = pd.read_csv('rams_russia_events.csv')
        event_type_col = 'Event Type'
    elif dataset == 'wiki':
        df = pd.read_csv('wikievents_actions_summaries.csv')
        df = df[df['Role'] != 'Participant']
        event_type_col = 'Role'
        n_topics = len(df['Role'].unique())
    else:
        raise ValueError("Dataset must be either 'rams' or 'wiki'")
    
    # Truncate event types based on the specified method
    if truncation_method == 'last_part':
        df = truncate_event_types_last_part(df)
    elif truncation_method == 'double':
        df = double_truncate_event_types(df, threshold=5)
    else:
        raise ValueError("Invalid truncation method")

    # Drop event types with frequency below 5
    event_type_counts = df['Truncated_Event_Type'].value_counts()
    event_types_to_keep = event_type_counts[event_type_counts >= 5].index
    df = df[df['Truncated_Event_Type'].isin(event_types_to_keep)]
    n_topics = len(event_types_to_keep)
    
    # Drop event types with frequency below 20
    #event_type_counts = df['Event_Type'].value_counts()
    #event_types_to_keep = event_type_counts[event_type_counts >= 20].index
    #df = df[df['Event_Type'].isin(event_types_to_keep)]
    
    # Determine which text to process
    if dataset == 'wiki' and use_summaries and not use_generated_summary:
        text_to_process = df['Summary'].tolist()
        text_type = "summary"
    elif use_summaries:  # For RAMS or Wiki with generated summary
        summaries = extract_summaries(df['Full Text'], df['Doc ID'], dataset=dataset, summary_length=summary_length)
        text_to_process = summaries
        text_type = f"summaries_{summary_length}words"
    elif use_sentences:
        text_to_process = df['Sentence'].tolist()
        text_type = "sentences"
    else:
        text_to_process = df['Full Text'].tolist()
        text_type = "full_text"
    
    # Remove any None values from text_to_process
    text_to_process = [t for t in text_to_process if t is not None]
    
    if use_gsdmm:
        # Run GSDMM modeling
        print(f"\nRunning GSDMM with {n_topics} topics")
        
        # Preprocess documents into word lists
        docs = [doc.split() for doc in text_to_process]
        
        # Initialize and train GSDMM
        mgp = MovieGroupProcess(K=n_topics)
        topics = mgp.fit(docs, vocab_size=15)
        model = mgp
    elif use_btm:
        # Run BTM modeling
        print(f"\nRunning BTM with {n_topics} topics")
        
        # Create document-term matrix
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(text_to_process).toarray()
        
        # Get vocabulary and biterms
        vocab = np.array(vectorizer.get_feature_names_out())
        biterms = vec_to_biterms(X)
        
        # Train BTM
        btm = oBTM(num_topics=n_topics, V=vocab)
        topic_distributions = btm.fit_transform(biterms, iterations=100)
        
        # Get the most likely topic for each document
        topics = np.argmax(topic_distributions, axis=1)
        model = btm
    elif use_ptm:
        # Run PTModel
        print(f"\nRunning PTModel with {n_topics} topics")
        
        # Initialize the model
        #ptm = tp.PTModel(k=n_topics, seed=42, tw=tp.TermWeight.IDF, min_cf=5, min_df=2, rm_top=20, p=5 * n_topics, alpha=0.1, eta=0.01)
        ptm = tp.PTModel(k=n_topics, seed=42)
        
        # Add documents to the model
        for doc in text_to_process:
            ptm.add_doc(doc.split())  # Split text into words
            
        # Set training parameters
        num_iterations = 500
        step_size = 20  # Number of steps per call
        convergence_threshold = 1e-4  # Stopping criterion for log-likelihood change

        # Initialize variables for convergence tracking
        prev_ll = None

        for i in range(0, num_iterations, step_size):
            ptm.train(step_size)

            # Calculate and print log-likelihood
            current_ll = ptm.ll_per_word
            print(f'Iteration: {i}, log-likelihood: {current_ll}')

            # Check for convergence
            if prev_ll is not None and abs(current_ll - prev_ll) < convergence_threshold:
                print("Converged.")
                break
            
            # Update previous log-likelihood
            prev_ll = current_ll
            
        # Get topic assignments for each document
        topics = [doc.get_topics(top_n=1)[0][0] for doc in ptm.docs]
        model = ptm
    elif use_lda:
        # Run LDA modeling
        topics, model = run_lda_model(text_to_process, n_topics, min_topic_size)
    else:
        # Embed the text
        embeddings = embed_text(text_to_process, text_type=text_type)
        
        # Run BERTopic modeling
        topics, model = run_topic_model(text_to_process, embeddings, n_topics, min_topic_size)
    
    # Add topics to the DataFrame
    df['Topic'] = topics
    
    
    # Calculate metrics for Event Type
    event_nmi = calculate_nmi(df['Truncated_Event_Type'], topics)
    event_purity = calculate_purity(df['Truncated_Event_Type'].astype(str), [str(t) for t in topics])
    event_ari = adjusted_rand_score(df['Truncated_Event_Type'], topics)
    
    # Calculate metrics for Role if it's the Wiki dataset
    if dataset == 'wiki':
        role_nmi = calculate_nmi(df['Role'], topics)
        role_purity = calculate_purity(df['Role'].astype(str), [str(t) for t in topics])
        role_ari = adjusted_rand_score(df['Role'], topics)
    else:
        role_nmi = None
        role_purity = None
        role_ari = None

    # Prepare results with dataset info and both sets of metrics
    results = {
        'dataset': dataset,
        'use_summaries': use_summaries,
        'use_sentences': use_sentences,
        'model_type': 'PTModel' if use_ptm else ('LDA' if use_lda else 'BERTopic'),
        'text_type': text_type,
        'n_topics': n_topics,
        'min_topic_size': min_topic_size,
        'truncation_method': truncation_method,
        'summary_length': summary_length if use_summaries and dataset == 'rams' else None,
        'event_nmi': event_nmi,
        'event_purity': event_purity,
        'event_ari': event_ari,
        'role_nmi': role_nmi,
        'role_purity': role_purity,
        'role_ari': role_ari,
        'timestamp': datetime.now().isoformat()
    }
    
    log_results(results)
    return results

def log_results(results):
    filename = 'benchmark_results.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(results)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    # Process RAMS dataset
    df_rams = pd.read_csv('rams_russia_events.csv')
    df_rams = truncate_event_types_last_part(df_rams)
    event_type_counts = df_rams['Truncated_Event_Type'].value_counts()
    event_types_to_keep = event_type_counts[event_type_counts >= 10].index
    n_event_types_rams = len(event_types_to_keep)
    print(f"RAMS dataset: Number of event types with 10+ instances: {n_event_types_rams}")
    
    # Process Wiki dataset
    df_wiki = pd.read_csv('wikievents_actions_summaries.csv')
    df_wiki = df_wiki[df_wiki['Role'] != 'Participant']
    n_event_types_wiki = len(df_wiki['Role'].unique())
    print(f"Wiki dataset: Number of event types: {n_event_types_wiki}")
    
    # Run Wiki dataset benchmarks
    print("\nRunning Wiki dataset benchmarks...")
    
    # Run each model type on different text types for Wiki dataset
    models = ['lda', 'gsdmm', 'bertopic']
    text_types = ['summary', 'sentence', 'full_text']
    
    for model in models:
        for text_type in text_types:
            print(f"\nRunning {model.upper()} benchmark on Wiki {text_type}...")
            results = run_benchmark(
                dataset='wiki',
                use_summaries=(text_type == 'summary'),
                use_lda=(model == 'lda'),
                use_gsdmm=(model == 'gsdmm'),
                use_sentences=(text_type == 'sentence'),
                n_topics=n_event_types_wiki,
                min_topic_size=5,
                truncation_method='last_part'
            )
            print(f"Results with {model.upper()} on {text_type}:", results)
    
    # Run RAMS dataset benchmarks
    print("\nRunning RAMS dataset benchmarks...")
    
    # Run LDA and GSDMM on full text and sentences for RAMS
    for model in ['lda', 'gsdmm']:
        for use_sentences in [True, False]:
            text_type = 'sentences' if use_sentences else 'full_text'
            print(f"\nRunning {model.upper()} benchmark on RAMS {text_type}...")
            results = run_benchmark(
                dataset='rams',
                use_summaries=False,
                use_lda=(model == 'lda'),
                use_gsdmm=(model == 'gsdmm'),
                use_sentences=use_sentences,
                n_topics=n_event_types_rams,
                min_topic_size=5,
                truncation_method='last_part'
            )
            print(f"Results with {model.upper()} on {text_type}:", results)
    
    # Run BERTopic with different summary lengths for RAMS
    summary_lengths = [1, 2, 3, 4, 5, 6, 8, 10]
    
    for length in summary_lengths:
        print(f"\nRunning BERTopic benchmark with {length}-word summaries on RAMS...")
        results = run_benchmark(
            dataset='rams',
            use_summaries=True,
            use_sentences=False,
            n_topics=n_event_types_rams,
            min_topic_size=5,
            truncation_method='last_part',
            summary_length=length
        )
        print(f"Results with BERTopic and {length}-word summaries:", results)
    
    # Also run BERTopic on full text and sentences for RAMS
    for use_sentences in [True, False]:
        text_type = 'sentences' if use_sentences else 'full_text'
        print(f"\nRunning BERTopic benchmark on RAMS {text_type}...")
        results = run_benchmark(
            dataset='rams',
            use_summaries=False,
            use_sentences=use_sentences,
            n_topics=n_event_types_rams,
            min_topic_size=5,
            truncation_method='last_part'
        )
        print(f"Results with BERTopic on {text_type}:", results)
    