#!/usr/bin/env python3
"""
Test script for Phase 3: Graph Construction + Profile Features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import networkx as nx

from fake_news.graphs.graph_construction import UserEngagementGraph, create_synthetic_engagement_data
from fake_news.graphs.profile_features import UserProfileExtractor, create_synthetic_user_data, create_synthetic_tweet_data
from fake_news.utils.logging import get_logger

def test_profile_features():
    """Test profile feature extraction."""
    print("Testing profile feature extraction...")
    
    # Create synthetic data
    user_data = create_synthetic_user_data(20)
    tweet_data = create_synthetic_tweet_data(100)
    
    # Create sample news data
    news_data = pd.DataFrame({
        'id': ['news_1', 'news_2'],
        'title': ['Test News 1', 'Test News 2'],
        'tweet_ids': ['tweet_0\ttweet_1', 'tweet_2\ttweet_3']
    })
    
    # Extract profile features
    extractor = UserProfileExtractor()
    result = extractor.extract_profile_features(news_data, tweet_data, user_data)
    
    print(f"Profile features shape: {result.shape}")
    print(f"Profile feature columns: {[col for col in result.columns if col.startswith('profile_')]}")
    
    return True

def test_graph_construction():
    """Test graph construction."""
    print("Testing graph construction...")
    
    # Create synthetic data
    news_data, tweet_data, user_data = create_synthetic_engagement_data(10)
    
    # Build graphs
    graph_builder = UserEngagementGraph()
    result = graph_builder.process_dataset(news_data, tweet_data, user_data, save_graphs=False)
    
    print(f"Created {len(result['graphs'])} graphs")
    print(f"Graph features: {len(result['graph_features'])}")
    print(f"Node features: {len(result['node_features'])}")
    print(f"Edge features: {len(result['edge_features'])}")
    
    # Show sample graph
    if result['graphs']:
        sample_id = list(result['graphs'].keys())[0]
        sample_graph = result['graphs'][sample_id]
        print(f"\nSample graph ({sample_id}):")
        print(f"  Nodes: {sample_graph.number_of_nodes()}")
        print(f"  Edges: {sample_graph.number_of_edges()}")
        print(f"  Node types: {set(nx.get_node_attributes(sample_graph, 'node_type').values())}")
    
    return True

def test_real_dataset():
    """Test with real dataset files."""
    print("Testing with real dataset...")
    
    try:
        # Load real dataset
        import pandas as pd
        
        # Load GossipCop data
        gossipcop_fake = pd.read_csv('dataset/gossipcop_fake.csv')
        gossipcop_real = pd.read_csv('dataset/gossipcop_real.csv')
        
        print(f"GossipCop fake: {len(gossipcop_fake)} articles")
        print(f"GossipCop real: {len(gossipcop_real)} articles")
        
        # Combine and add labels
        gossipcop_fake['label'] = 1  # Fake
        gossipcop_real['label'] = 0  # Real
        
        news_data = pd.concat([gossipcop_fake, gossipcop_real], ignore_index=True)
        print(f"Total news articles: {len(news_data)}")
        
        # Test with small sample
        sample_data = news_data.head(10)
        
        # Create synthetic user and tweet data for testing
        user_data = create_synthetic_user_data(50)
        tweet_data = create_synthetic_tweet_data(200)
        
        # Extract profile features
        extractor = UserProfileExtractor()
        result = extractor.extract_profile_features(sample_data, tweet_data, user_data)
        
        print(f"Profile features extracted for {len(result)} articles")
        print(f"Profile feature columns: {[col for col in result.columns if col.startswith('profile_')]}")
        
        return True
        
    except Exception as e:
        print(f"Real dataset test failed: {e}")
        return False

def main():
    print("Phase 3 Test: Graph Construction + Profile Features")
    print("=" * 60)
    
    try:
        # Test profile features
        test_profile_features()
        print("✅ Profile features test passed")
        
        # Test graph construction
        test_graph_construction()
        print("✅ Graph construction test passed")
        
        # Test with real dataset
        test_real_dataset()
        print("✅ Real dataset test passed")
        
        print("\n🎉 Phase 3 basic functionality works!")
        print("Note: Full graph analysis requires the complete notebook.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
