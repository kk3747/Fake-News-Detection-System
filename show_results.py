#!/usr/bin/env python3
"""
Demonstration script to show Phase 3 results to professor
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import pickle
import json
import networkx as nx

def show_processed_data():
    """Show the processed news data"""
    print("=" * 60)
    print("📊 PROCESSED NEWS DATA")
    print("=" * 60)
    
    # Load processed data
    df = pd.read_csv('src/data/processed/sample_news_processed.csv')
    
    print(f"Total articles processed: {len(df)}")
    print(f"Fake news: {df['label'].sum()}")
    print(f"Real news: {len(df) - df['label'].sum()}")
    print()
    
    print("Sample articles:")
    print("-" * 40)
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"{i+1}. {row['title']}")
        print(f"   Processed: {row['title_processed']}")
        print(f"   Label: {'FAKE' if row['label'] == 1 else 'REAL'}")
        print()

def show_embeddings():
    """Show embedding information"""
    print("=" * 60)
    print("🔤 TEXT EMBEDDINGS")
    print("=" * 60)
    
    # Load embeddings
    spacy_emb = np.load('src/data/processed/sample_spacy_embeddings.npy')
    bert_emb = np.load('src/data/processed/sample_bert_embeddings.npy')
    f3_features = np.load('src/data/processed/sample_f3_features.npy')
    
    print(f"spaCy embeddings: {spacy_emb.shape} (300D word2vec)")
    print(f"BERT embeddings: {bert_emb.shape} (768D contextual)")
    print(f"F3 features: {f3_features.shape} (315D combined)")
    print()
    
    print("Embedding statistics:")
    print(f"  spaCy range: [{spacy_emb.min():.3f}, {spacy_emb.max():.3f}]")
    print(f"  BERT range: [{bert_emb.min():.3f}, {bert_emb.max():.3f}]")
    print(f"  F3 range: [{f3_features.min():.3f}, {f3_features.max():.3f}]")
    print()

def show_graphs():
    """Show graph structure information"""
    print("=" * 60)
    print("🔗 ENGAGEMENT GRAPHS")
    print("=" * 60)
    
    # Load graphs
    with open('src/data/graphs/retweet_networks.pkl', 'rb') as f:
        graphs = pickle.load(f)
    
    with open('src/data/graphs/graph_features.pkl', 'rb') as f:
        graph_features = pickle.load(f)
    
    print(f"Total graphs created: {len(graphs)}")
    print()
    
    # Show sample graph
    sample_id = list(graphs.keys())[0]
    sample_graph = graphs[sample_id]
    
    print(f"Sample graph ({sample_id}):")
    print(f"  Nodes: {sample_graph.number_of_nodes()}")
    print(f"  Edges: {sample_graph.number_of_edges()}")
    
    # Get node types
    node_types = nx.get_node_attributes(sample_graph, 'node_type')
    news_nodes = sum(1 for nt in node_types.values() if nt == 'news')
    user_nodes = sum(1 for nt in node_types.values() if nt == 'user')
    print(f"  News nodes: {news_nodes}")
    print(f"  User nodes: {user_nodes}")
    
    # Show graph features
    if sample_id in graph_features:
        features = graph_features[sample_id]
        print(f"  Density: {features['density']:.3f}")
        print(f"  Avg degree: {features['avg_degree']:.3f}")
        print(f"  Clustering: {features['clustering_coeff']:.3f}")
    print()

def show_profile_features():
    """Show profile features"""
    print("=" * 60)
    print("👤 USER PROFILE FEATURES")
    print("=" * 60)
    
    df = pd.read_csv('src/data/processed/sample_news_processed.csv')
    profile_cols = [col for col in df.columns if col.startswith('profile_')]
    
    print(f"Total profile features: {len(profile_cols)}")
    print("Feature names:")
    for i, col in enumerate(profile_cols):
        print(f"  {i+1:2d}. {col}")
    print()
    
    # Show statistics
    profile_data = df[profile_cols]
    print("Feature statistics:")
    print(f"  Average followers: {profile_data['profile_followers_count'].mean():.3f}")
    print(f"  Average friends: {profile_data['profile_friends_count'].mean():.3f}")
    print(f"  Verified users: {profile_data['profile_verified_status'].sum()}")
    print(f"  Users with descriptions: {profile_data['profile_has_description'].sum()}")
    print()

def show_file_locations():
    """Show where the result files are located"""
    print("=" * 60)
    print("📁 RESULT FILE LOCATIONS")
    print("=" * 60)
    
    files = [
        ("Processed News Data", "src/data/processed/sample_news_processed.csv"),
        ("spaCy Embeddings", "src/data/processed/sample_spacy_embeddings.npy"),
        ("BERT Embeddings", "src/data/processed/sample_bert_embeddings.npy"),
        ("F3 Features", "src/data/processed/sample_f3_features.npy"),
        ("Engagement Graphs", "src/data/graphs/retweet_networks.pkl"),
        ("Graph Features", "src/data/graphs/graph_features.pkl"),
        ("Node Features", "src/data/graphs/node_features.pkl"),
        ("Edge Features", "src/data/graphs/edge_features.pkl"),
        ("Metadata", "src/data/processed/sample_metadata.json")
    ]
    
    for name, path in files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {name}: {path} ({size:,} bytes)")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
    print()

def main():
    """Main demonstration function"""
    print("🎓 FAKE NEWS DETECTION PROJECT - PHASE 3 DEMO")
    print("For Professor Presentation")
    print()
    
    show_processed_data()
    show_embeddings()
    show_graphs()
    show_profile_features()
    show_file_locations()
    
    print("=" * 60)
    print("✅ PHASE 3 COMPLETE - READY FOR PHASE 4")
    print("=" * 60)
    print("Next: GNN Models (GCN, GAT, BiGCN) and Training")
    print("All data is processed and ready for neural network training!")

if __name__ == "__main__":
    main()
