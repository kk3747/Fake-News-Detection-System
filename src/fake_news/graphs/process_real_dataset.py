#!/usr/bin/env python3
"""
Process real dataset for Phase 3: Graph Construction + Profile Features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

from fake_news.graphs.graph_construction import UserEngagementGraph
from fake_news.graphs.profile_features import UserProfileExtractor
from fake_news.features.text_preprocessing import TextPreprocessor
from fake_news.features.embeddings import EmbeddingExtractor
from fake_news.utils.logging import get_logger
from fake_news.utils.paths import PROCESSED_DIR, GRAPHS_DIR

logger = get_logger(__name__)

def load_real_dataset():
    """Load the real GossipCop and PolitiFact datasets."""
    logger.info("Loading real dataset...")
    
    # Load GossipCop data
    gossipcop_fake = pd.read_csv('dataset/gossipcop_fake.csv')
    gossipcop_real = pd.read_csv('dataset/gossipcop_real.csv')
    
    # Load PolitiFact data
    politifact_fake = pd.read_csv('dataset/politifact_fake.csv')
    politifact_real = pd.read_csv('dataset/politifact_real.csv')
    
    # Add labels and source
    gossipcop_fake['label'] = 1  # Fake
    gossipcop_real['label'] = 0  # Real
    politifact_fake['label'] = 1  # Fake
    politifact_real['label'] = 0  # Real
    
    gossipcop_fake['source'] = 'gossipcop'
    gossipcop_real['source'] = 'gossipcop'
    politifact_fake['source'] = 'politifact'
    politifact_real['source'] = 'politifact'
    
    # Combine all data
    all_data = pd.concat([
        gossipcop_fake, gossipcop_real,
        politifact_fake, politifact_real
    ], ignore_index=True)
    
    logger.info(f"Loaded dataset: {len(all_data)} articles")
    logger.info(f"  GossipCop: {len(gossipcop_fake + gossipcop_real)} articles")
    logger.info(f"  PolitiFact: {len(politifact_fake + politifact_real)} articles")
    logger.info(f"  Fake: {all_data['label'].sum()}, Real: {len(all_data) - all_data['label'].sum()}")
    
    return all_data

def create_synthetic_engagement_data_for_real_news(news_data: pd.DataFrame) -> tuple:
    """
    Create synthetic user and tweet data for real news articles.
    This simulates the engagement data that would come from Twitter API.
    """
    logger.info("Creating synthetic engagement data for real news...")
    
    # Create synthetic users
    n_users = 100
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
    
    # Create synthetic tweets for each news article
    tweet_data = []
    for idx, row in news_data.iterrows():
        # Parse tweet IDs from the real data
        tweet_ids = str(row['tweet_ids']).split('\t')
        if len(tweet_ids) > 0 and tweet_ids[0] != 'nan':
            # Create synthetic tweet data for each tweet ID
            for tweet_id in tweet_ids[:10]:  # Limit to first 10 tweets per article
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
    
    logger.info(f"Created synthetic engagement data: {len(user_df)} users, {len(tweet_df)} tweets")
    return user_df, tweet_df

def process_phase3_pipeline():
    """Run the complete Phase 3 pipeline."""
    logger.info("Starting Phase 3 pipeline...")
    
    # Load real dataset
    news_data = load_real_dataset()
    
    # Create synthetic engagement data
    user_data, tweet_data = create_synthetic_engagement_data_for_real_news(news_data)
    
    # Text preprocessing
    logger.info("Preprocessing text...")
    from fake_news.features.text_preprocessing import preprocess_news_data
    news_processed = preprocess_news_data(news_data, text_columns=['title'])
    
    # Extract text embeddings
    logger.info("Extracting text embeddings...")
    extractor = EmbeddingExtractor()
    
    # Extract spaCy embeddings
    spacy_embeddings = extractor.extract_spacy_embeddings(
        news_processed['title_processed'].tolist(),
        save_path=str(PROCESSED_DIR / 'real_spacy_embeddings.pkl')
    )
    
    # Extract BERT embeddings
    bert_embeddings = extractor.extract_bert_embeddings(
        news_processed['title_processed'].tolist(),
        save_path=str(PROCESSED_DIR / 'real_bert_embeddings.pkl')
    )
    
    # Extract profile features
    logger.info("Extracting profile features...")
    profile_extractor = UserProfileExtractor()
    news_with_profiles = profile_extractor.extract_profile_features(
        news_processed, tweet_data, user_data
    )
    
    # Create F3 features (spaCy + Profile)
    logger.info("Creating F3 features (spaCy + Profile)...")
    profile_cols = [col for col in news_with_profiles.columns if col.startswith('profile_')]
    profile_features = news_with_profiles[profile_cols].values
    
    # Combine spaCy embeddings with profile features
    f3_features = np.hstack([spacy_embeddings, profile_features])
    
    # Save F3 features
    np.save(PROCESSED_DIR / 'f3_features.npy', f3_features)
    logger.info(f"Saved F3 features: {f3_features.shape}")
    
    # Build engagement graphs
    logger.info("Building engagement graphs...")
    graph_builder = UserEngagementGraph()
    graph_result = graph_builder.process_dataset(
        news_with_profiles, tweet_data, user_data, save_graphs=True
    )
    
    # Save processed data
    news_with_profiles.to_csv(PROCESSED_DIR / 'real_news_processed.csv', index=False)
    
    # Save metadata
    metadata = {
        'n_articles': len(news_data),
        'n_fake': news_data['label'].sum(),
        'n_real': len(news_data) - news_data['label'].sum(),
        'spacy_shape': spacy_embeddings.shape,
        'bert_shape': bert_embeddings.shape,
        'f3_shape': f3_features.shape,
        'n_graphs': len(graph_result['graphs']),
        'n_users': len(user_data),
        'n_tweets': len(tweet_data)
    }
    
    import json
    with open(PROCESSED_DIR / 'phase3_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Phase 3 pipeline completed successfully!")
    logger.info(f"Processed {len(news_data)} articles")
    logger.info(f"Created {len(graph_result['graphs'])} engagement graphs")
    logger.info(f"F3 features shape: {f3_features.shape}")
    
    return {
        'news_data': news_with_profiles,
        'spacy_embeddings': spacy_embeddings,
        'bert_embeddings': bert_embeddings,
        'f3_features': f3_features,
        'graphs': graph_result['graphs'],
        'user_data': user_data,
        'tweet_data': tweet_data
    }

if __name__ == "__main__":
    result = process_phase3_pipeline()
    print("Phase 3 pipeline completed successfully!")
    print(f"Processed {len(result['news_data'])} articles")
    print(f"F3 features shape: {result['f3_features'].shape}")
    print(f"Created {len(result['graphs'])} engagement graphs")
