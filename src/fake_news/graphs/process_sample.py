#!/usr/bin/env python3
"""
Process a small sample of the real dataset for Phase 3 testing.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

from fake_news.graphs.graph_construction import UserEngagementGraph
from fake_news.graphs.profile_features import UserProfileExtractor
from fake_news.features.text_preprocessing import preprocess_news_data
from fake_news.features.embeddings import EmbeddingExtractor
from fake_news.utils.logging import get_logger
from fake_news.utils.paths import PROCESSED_DIR, GRAPHS_DIR

logger = get_logger(__name__)

def process_sample_dataset(n_samples: int = 100):
    """Process a small sample of the real dataset."""
    logger.info(f"Processing sample dataset with {n_samples} articles...")
    
    # Load real dataset
    gossipcop_fake = pd.read_csv('dataset/gossipcop_fake.csv')
    gossipcop_real = pd.read_csv('dataset/gossipcop_real.csv')
    
    # Take a small sample
    sample_fake = gossipcop_fake.head(n_samples // 2)
    sample_real = gossipcop_real.head(n_samples // 2)
    
    # Add labels
    sample_fake['label'] = 1  # Fake
    sample_real['label'] = 0  # Real
    
    # Combine
    news_data = pd.concat([sample_fake, sample_real], ignore_index=True)
    
    logger.info(f"Sample dataset: {len(news_data)} articles")
    logger.info(f"Fake: {news_data['label'].sum()}, Real: {len(news_data) - news_data['label'].sum()}")
    
    # Create synthetic engagement data
    n_users = 50
    user_data = []
    for i in range(n_users):
        user_data.append({
            'user_id': f'user_{i}',
            'screen_name': f'user{i}',
            'followers': int(np.random.lognormal(mean=6, sigma=2)),
            'friends': int(np.random.lognormal(mean=5, sigma=1.5)),
            'listed': int(np.random.lognormal(mean=2, sigma=1)),
            'favourites': int(np.random.lognormal(mean=4, sigma=1.5)),
            'statuses': int(np.random.lognormal(mean=4, sigma=1.5)),
            'created_at': '2010-01-01',
            'verified': np.random.choice([True, False], p=[0.1, 0.9])
        })
    
    user_df = pd.DataFrame(user_data)
    
    # Create synthetic tweets
    tweet_data = []
    for idx, row in news_data.iterrows():
        tweet_ids = str(row['tweet_ids']).split('\t')
        if len(tweet_ids) > 0 and tweet_ids[0] != 'nan':
            for tweet_id in tweet_ids[:5]:  # Limit to 5 tweets per article
                if tweet_id.strip():
                    tweet_data.append({
                        'tweet_id': tweet_id.strip(),
                        'user_id': f'user_{np.random.randint(0, n_users)}',
                        'created_at': '2023-01-01',
                        'text': f"Tweet about: {row['title'][:50]}...",
                        'retweet_of': None,
                        'in_reply_to': None,
                        'lang': 'en'
                    })
    
    tweet_df = pd.DataFrame(tweet_data)
    
    logger.info(f"Created engagement data: {len(user_df)} users, {len(tweet_df)} tweets")
    
    # Text preprocessing
    logger.info("Preprocessing text...")
    news_processed = preprocess_news_data(news_data, text_columns=['title'])
    
    # Extract text embeddings
    logger.info("Extracting text embeddings...")
    extractor = EmbeddingExtractor()
    
    # Extract spaCy embeddings
    spacy_embeddings = extractor.extract_spacy_embeddings(
        news_processed['title_processed'].tolist()
    )
    
    # Extract BERT embeddings
    bert_embeddings = extractor.extract_bert_embeddings(
        news_processed['title_processed'].tolist()
    )
    
    # Extract profile features
    logger.info("Extracting profile features...")
    profile_extractor = UserProfileExtractor()
    news_with_profiles = profile_extractor.extract_profile_features(
        news_processed, tweet_df, user_df
    )
    
    # Create F3 features (spaCy + Profile)
    logger.info("Creating F3 features (spaCy + Profile)...")
    profile_cols = [col for col in news_with_profiles.columns if col.startswith('profile_')]
    profile_features = news_with_profiles[profile_cols].values
    
    # Combine spaCy embeddings with profile features
    f3_features = np.hstack([spacy_embeddings, profile_features])
    
    # Build engagement graphs
    logger.info("Building engagement graphs...")
    graph_builder = UserEngagementGraph()
    graph_result = graph_builder.process_dataset(
        news_with_profiles, tweet_df, user_df, save_graphs=True
    )
    
    # Save results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    news_with_profiles.to_csv(PROCESSED_DIR / 'sample_news_processed.csv', index=False)
    np.save(PROCESSED_DIR / 'sample_spacy_embeddings.npy', spacy_embeddings)
    np.save(PROCESSED_DIR / 'sample_bert_embeddings.npy', bert_embeddings)
    np.save(PROCESSED_DIR / 'sample_f3_features.npy', f3_features)
    
    # Save metadata
    metadata = {
        'n_articles': int(len(news_data)),
        'n_fake': int(news_data['label'].sum()),
        'n_real': int(len(news_data) - news_data['label'].sum()),
        'spacy_shape': list(spacy_embeddings.shape),
        'bert_shape': list(bert_embeddings.shape),
        'f3_shape': list(f3_features.shape),
        'n_graphs': int(len(graph_result['graphs'])),
        'n_users': int(len(user_df)),
        'n_tweets': int(len(tweet_df))
    }
    
    import json
    with open(PROCESSED_DIR / 'sample_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Sample processing completed successfully!")
    logger.info(f"Processed {len(news_data)} articles")
    logger.info(f"F3 features shape: {f3_features.shape}")
    logger.info(f"Created {len(graph_result['graphs'])} engagement graphs")
    
    return {
        'news_data': news_with_profiles,
        'spacy_embeddings': spacy_embeddings,
        'bert_embeddings': bert_embeddings,
        'f3_features': f3_features,
        'graphs': graph_result['graphs'],
        'user_data': user_df,
        'tweet_data': tweet_df
    }

if __name__ == "__main__":
    result = process_sample_dataset(50)  # Process 50 articles
    print("Sample processing completed successfully!")
    print(f"Processed {len(result['news_data'])} articles")
    print(f"F3 features shape: {result['f3_features'].shape}")
    print(f"Created {len(result['graphs'])} engagement graphs")
