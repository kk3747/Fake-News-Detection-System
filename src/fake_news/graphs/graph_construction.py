"""
Graph construction module for user engagement networks.
Implements the hierarchical tree structure described in the paper.
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import pickle
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.paths import GRAPHS_DIR

logger = get_logger(__name__)

class UserEngagementGraph:
    """
    Construct user engagement graphs for fake news detection.
    
    Implements the hierarchical tree structure from the paper:
    - Root node: news article
    - Leaf nodes: users who retweeted
    - Edges: user-to-news (retweet) and user-to-user (retweet chain)
    """
    
    def __init__(self):
        self.graphs = {}
        self.node_features = {}
        self.edge_features = {}
        logger.info("Initialized UserEngagementGraph")
    
    def parse_tweet_ids(self, tweet_ids_str: str) -> List[str]:
        """
        Parse tweet IDs from tab-separated string.
        
        Args:
            tweet_ids_str: Tab-separated string of tweet IDs
            
        Returns:
            List of tweet IDs
        """
        if pd.isna(tweet_ids_str) or tweet_ids_str == 'nan':
            return []
        
        return str(tweet_ids_str).split('\t')
    
    def build_retweet_network(self, 
                            news_data: pd.DataFrame,
                            tweet_data: Optional[pd.DataFrame] = None,
                            user_data: Optional[pd.DataFrame] = None) -> Dict[str, nx.Graph]:
        """
        Build retweet networks for each news article.
        
        Args:
            news_data: DataFrame with news articles
            tweet_data: Optional DataFrame with tweet information
            user_data: Optional DataFrame with user profile information
            
        Returns:
            Dictionary mapping news_id to NetworkX graph
        """
        logger.info(f"Building retweet networks for {len(news_data)} news articles...")
        
        graphs = {}
        
        for idx, row in news_data.iterrows():
            news_id = row['id']
            tweet_ids = self.parse_tweet_ids(row.get('tweet_ids', ''))
            
            if not tweet_ids or tweet_ids[0] == 'nan':
                # Create empty graph if no tweets
                graphs[news_id] = nx.Graph()
                continue
            
            # Create graph for this news article
            G = nx.Graph()
            
            # Add news article as root node
            G.add_node(news_id, node_type='news', title=row.get('title', ''))
            
            if tweet_data is not None:
                # Filter tweets for this article
                article_tweets = tweet_data[tweet_data['tweet_id'].astype(str).isin(tweet_ids)]
                
                if len(article_tweets) > 0:
                    # Add user nodes and edges
                    for _, tweet in article_tweets.iterrows():
                        user_id = tweet['user_id']
                        
                        # Add user node
                        G.add_node(user_id, node_type='user')
                        
                        # Add edge from user to news (retweet)
                        G.add_edge(user_id, news_id, edge_type='retweet')
                        
                        # Add user-to-user edges if retweet chain exists
                        if 'retweet_of' in tweet and pd.notna(tweet['retweet_of']):
                            original_user = tweet['retweet_of']
                            if original_user in G.nodes():
                                G.add_edge(user_id, original_user, edge_type='retweet_chain')
            
            graphs[news_id] = G
        
        logger.info(f"Built {len(graphs)} retweet networks")
        return graphs
    
    def extract_graph_features(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Extract graph-level features for analysis.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        if len(graph.nodes()) == 0:
            return {feat: 0.0 for feat in [
                'num_nodes', 'num_edges', 'density', 'avg_degree',
                'num_users', 'num_news', 'max_degree', 'clustering_coeff'
            ]}
        
        # Basic graph metrics
        features['num_nodes'] = graph.number_of_nodes()
        features['num_edges'] = graph.number_of_edges()
        features['density'] = nx.density(graph)
        features['avg_degree'] = np.mean([d for n, d in graph.degree()])
        features['max_degree'] = max([d for n, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0
        
        # Node type counts
        node_types = nx.get_node_attributes(graph, 'node_type')
        features['num_users'] = sum(1 for nt in node_types.values() if nt == 'user')
        features['num_news'] = sum(1 for nt in node_types.values() if nt == 'news')
        
        # Clustering coefficient
        try:
            features['clustering_coeff'] = nx.average_clustering(graph)
        except:
            features['clustering_coeff'] = 0.0
        
        return features
    
    def create_node_features(self, 
                           graph: nx.Graph,
                           user_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Create node feature matrix for the graph.
        
        Args:
            graph: NetworkX graph
            user_data: Optional user profile data
            
        Returns:
            Node feature matrix
        """
        nodes = list(graph.nodes())
        if len(nodes) == 0:
            return np.array([])
        
        # Basic node features (10-dimensional as mentioned in paper)
        features = []
        
        for node in nodes:
            node_features = np.zeros(10)  # 10-dimensional profile features
            
            if user_data is not None:
                # Get user profile features
                user_info = user_data[user_data['user_id'].astype(str) == str(node)]
                if len(user_info) > 0:
                    user_row = user_info.iloc[0]
                    node_features[0] = min(1.0, user_row.get('followers', 0) / 1000000)
                    node_features[1] = min(1.0, user_row.get('friends', 0) / 5000)
                    node_features[2] = min(1.0, user_row.get('listed', 0) / 1000)
                    node_features[3] = min(1.0, user_row.get('favourites', 0) / 10000)
                    node_features[4] = min(1.0, user_row.get('statuses', 0) / 50000)
                    node_features[5] = min(1.0, user_row.get('followers', 0) / max(1, user_row.get('friends', 1)))
                    node_features[6] = min(1.0, user_row.get('statuses', 0) / max(1, 365))  # tweets per day
                    node_features[7] = 1.0 if user_row.get('followers', 0) > 10000 else 0.0  # verified
                    node_features[8] = 1.0 if user_row.get('followers', 0) > 0 else 0.0  # has description
                    node_features[9] = 1.0 if user_row.get('verified', False) else 0.0  # verified status
            
            features.append(node_features)
        
        return np.array(features)
    
    def create_edge_features(self, graph: nx.Graph) -> np.ndarray:
        """
        Create edge feature matrix for the graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Edge feature matrix
        """
        edges = list(graph.edges())
        if len(edges) == 0:
            return np.array([])
        
        # Basic edge features
        features = []
        
        for edge in edges:
            edge_features = np.zeros(3)  # 3-dimensional edge features
            
            # Edge type (retweet vs retweet_chain)
            edge_type = graph.get_edge_data(edge[0], edge[1], {}).get('edge_type', 'unknown')
            edge_features[0] = 1.0 if edge_type == 'retweet' else 0.0
            edge_features[1] = 1.0 if edge_type == 'retweet_chain' else 0.0
            
            # Edge weight (simplified)
            edge_features[2] = 1.0
            
            features.append(edge_features)
        
        return np.array(features)
    
    def process_dataset(self, 
                       news_data: pd.DataFrame,
                       tweet_data: Optional[pd.DataFrame] = None,
                       user_data: Optional[pd.DataFrame] = None,
                       save_graphs: bool = True) -> Dict[str, any]:
        """
        Process entire dataset and create graphs with features.
        
        Args:
            news_data: DataFrame with news articles
            tweet_data: Optional DataFrame with tweet information
            user_data: Optional DataFrame with user profile information
            save_graphs: Whether to save graphs to disk
            
        Returns:
            Dictionary with processed graphs and features
        """
        logger.info("Processing dataset for graph construction...")
        
        # Build retweet networks
        graphs = self.build_retweet_network(news_data, tweet_data, user_data)
        
        # Extract features for each graph
        graph_features = {}
        node_features = {}
        edge_features = {}
        
        for news_id, graph in graphs.items():
            # Graph-level features
            graph_features[news_id] = self.extract_graph_features(graph)
            
            # Node features
            node_features[news_id] = self.create_node_features(graph, user_data)
            
            # Edge features
            edge_features[news_id] = self.create_edge_features(graph)
        
        # Save graphs if requested
        if save_graphs:
            GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save graphs
            with open(GRAPHS_DIR / 'retweet_networks.pkl', 'wb') as f:
                pickle.dump(graphs, f)
            
            # Save features
            with open(GRAPHS_DIR / 'graph_features.pkl', 'wb') as f:
                pickle.dump(graph_features, f)
            
            with open(GRAPHS_DIR / 'node_features.pkl', 'wb') as f:
                pickle.dump(node_features, f)
            
            with open(GRAPHS_DIR / 'edge_features.pkl', 'wb') as f:
                pickle.dump(edge_features, f)
            
            logger.info(f"Saved graphs and features to {GRAPHS_DIR}")
        
        return {
            'graphs': graphs,
            'graph_features': graph_features,
            'node_features': node_features,
            'edge_features': edge_features
        }
    
    def load_graphs(self, graphs_dir: Optional[Path] = None) -> Dict[str, any]:
        """
        Load pre-computed graphs and features.
        
        Args:
            graphs_dir: Directory containing graph files
            
        Returns:
            Dictionary with loaded graphs and features
        """
        if graphs_dir is None:
            graphs_dir = GRAPHS_DIR
        
        logger.info(f"Loading graphs from {graphs_dir}")
        
        result = {}
        
        # Load graphs
        graphs_file = graphs_dir / 'retweet_networks.pkl'
        if graphs_file.exists():
            with open(graphs_file, 'rb') as f:
                result['graphs'] = pickle.load(f)
        
        # Load features
        features_files = {
            'graph_features': 'graph_features.pkl',
            'node_features': 'node_features.pkl',
            'edge_features': 'edge_features.pkl'
        }
        
        for key, filename in features_files.items():
            file_path = graphs_dir / filename
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result[key] = pickle.load(f)
        
        logger.info(f"Loaded {len(result.get('graphs', {}))} graphs")
        return result


def create_synthetic_engagement_data(n_articles: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic engagement data for testing.
    
    Args:
        n_articles: Number of news articles to create
        
    Returns:
        Tuple of (news_data, tweet_data, user_data)
    """
    logger.info(f"Creating synthetic engagement data for {n_articles} articles...")
    
    # Create news articles
    news_data = []
    for i in range(n_articles):
        # Create tweet IDs for this article
        n_tweets = np.random.randint(5, 50)
        tweet_ids = [f'tweet_{i}_{j}' for j in range(n_tweets)]
        
        news_data.append({
            'id': f'news_{i}',
            'title': f'News Article {i}',
            'news_url': f'https://example.com/news/{i}',
            'tweet_ids': '\t'.join(tweet_ids)
        })
    
    news_df = pd.DataFrame(news_data)
    
    # Create tweets
    tweet_data = []
    for i in range(n_articles):
        n_tweets = len(news_df.iloc[i]['tweet_ids'].split('\t'))
        for j in range(n_tweets):
            tweet_data.append({
                'tweet_id': f'tweet_{i}_{j}',
                'user_id': f'user_{np.random.randint(0, 20)}',  # 20 different users
                'created_at': '2023-01-01',
                'text': f'This is tweet {j} about news {i}',
                'retweet_of': None,
                'in_reply_to': None,
                'lang': 'en'
            })
    
    tweet_df = pd.DataFrame(tweet_data)
    
    # Create users
    user_data = []
    for i in range(20):  # 20 users
        user_data.append({
            'user_id': f'user_{i}',
            'screen_name': f'user{i}',
            'followers': np.random.randint(100, 10000),
            'friends': np.random.randint(50, 1000),
            'listed': np.random.randint(0, 100),
            'favourites': np.random.randint(100, 5000),
            'statuses': np.random.randint(100, 10000),
            'created_at': '2010-01-01',
            'verified': np.random.choice([True, False], p=[0.1, 0.9])
        })
    
    user_df = pd.DataFrame(user_data)
    
    logger.info(f"Created synthetic data: {len(news_df)} articles, {len(tweet_df)} tweets, {len(user_df)} users")
    return news_df, tweet_df, user_df


if __name__ == "__main__":
    # Test the graph construction
    graph_builder = UserEngagementGraph()
    
    # Create synthetic data
    news_data, tweet_data, user_data = create_synthetic_engagement_data(10)
    
    # Process dataset
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
