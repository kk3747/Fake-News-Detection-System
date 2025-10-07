#!/usr/bin/env python3
"""
Simple script to view results in VS Code
Run this file directly in VS Code to see all results
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import pickle
import json

def main():
    print("🎓 FAKE NEWS DETECTION PROJECT - RESULTS VIEWER")
    print("=" * 60)
    
    # Check if files exist
    csv_file = 'src/data/processed/sample_news_processed.csv'
    if not os.path.exists(csv_file):
        print("❌ Results not found! Please run the processing first:")
        print("   python3 src/fake_news/graphs/process_sample.py")
        return
    
    # Load and display processed data
    print("📊 PROCESSED NEWS DATA")
    print("-" * 40)
    df = pd.read_csv(csv_file)
    print(f"Total articles: {len(df)}")
    print(f"Fake news: {df['label'].sum()}")
    print(f"Real news: {len(df) - df['label'].sum()}")
    print()
    
    print("Sample articles:")
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"{i+1}. {row['title']}")
        print(f"   Processed: {row['title_processed']}")
        print(f"   Label: {'FAKE' if row['label'] == 1 else 'REAL'}")
        print()
    
    # Load and display embeddings
    print("🔤 TEXT EMBEDDINGS")
    print("-" * 40)
    try:
        spacy_emb = np.load('src/data/processed/sample_spacy_embeddings.npy')
        bert_emb = np.load('src/data/processed/sample_bert_embeddings.npy')
        f3_features = np.load('src/data/processed/sample_f3_features.npy')
        
        print(f"spaCy embeddings: {spacy_emb.shape} (300D)")
        print(f"BERT embeddings: {bert_emb.shape} (768D)")
        print(f"F3 features: {f3_features.shape} (315D)")
        print()
    except FileNotFoundError as e:
        print(f"❌ Embedding file not found: {e}")
    
    # Load and display graphs
    print("🔗 ENGAGEMENT GRAPHS")
    print("-" * 40)
    try:
        with open('src/data/graphs/retweet_networks.pkl', 'rb') as f:
            graphs = pickle.load(f)
        
        print(f"Total graphs created: {len(graphs)}")
        
        # Show sample graph
        sample_id = list(graphs.keys())[0]
        sample_graph = graphs[sample_id]
        print(f"Sample graph: {sample_id}")
        print(f"  Nodes: {sample_graph.number_of_nodes()}")
        print(f"  Edges: {sample_graph.number_of_edges()}")
        print()
    except FileNotFoundError as e:
        print(f"❌ Graph file not found: {e}")
    
    # Show profile features
    print("👤 PROFILE FEATURES")
    print("-" * 40)
    profile_cols = [col for col in df.columns if col.startswith('profile_')]
    print(f"Total profile features: {len(profile_cols)}")
    print("Feature names:")
    for i, col in enumerate(profile_cols[:5]):  # Show first 5
        print(f"  {i+1}. {col}")
    print("  ... (and 10 more)")
    print()
    
    # Show file locations
    print("📁 RESULT FILES")
    print("-" * 40)
    files = [
        ("Processed Data", "src/data/processed/sample_news_processed.csv"),
        ("Graphs", "src/data/graphs/retweet_networks.pkl"),
        ("spaCy Features", "src/data/processed/sample_spacy_embeddings.npy"),
        ("BERT Features", "src/data/processed/sample_bert_embeddings.npy"),
        ("F3 Features", "src/data/processed/sample_f3_features.npy")
    ]
    
    for name, path in files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {name}: {path} ({size:,} bytes)")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
    print()
    
    print("✅ PHASE 3 COMPLETE!")
    print("Ready for Phase 4: GNN Models")

if __name__ == "__main__":
    main()
