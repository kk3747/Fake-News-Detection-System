import os
import json
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from fake_news_data_loader import FakeNewsDataset
from text_feature_extractor import TextFeatureExtractor


def deterministic_vector_from_id(s: str, dim: int = 64) -> np.ndarray:
    # Deterministic pseudo-random vector from string id using SHA256 seed
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]
    seed = int(h, 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=(dim,)).astype(np.float32)
    # normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class EngagementGraphBuilder:
    def __init__(self, data_dir: str = 'dataset', out_dir: str = 'graph_data', tweet_feat_dim: int = 64):
        self.data_dir = data_dir
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tweet_feat_dim = tweet_feat_dim

    def build(self):
        ds = FakeNewsDataset(self.data_dir)
        df = ds.get_all()
        df = ds.preprocess(df)

        news_feat_path = os.path.join(self.out_dir, 'news_features.npy')
        if os.path.exists(news_feat_path):
            print('Loading existing news features from', news_feat_path)
            news_feats = np.load(news_feat_path)
        else:
            # Extract text features for news nodes
            extractor = TextFeatureExtractor()
            news_feats = extractor.extract_features(df, text_column='title')
            np.save(news_feat_path, news_feats)

        # Collect all tweet ids
        all_tweet_ids = set()
        for lst in df['tweet_ids']:
            for t in lst:
                all_tweet_ids.add(t)
        all_tweet_ids = sorted(list(all_tweet_ids))

        print(f'News items: {len(df)}, Tweet nodes: {len(all_tweet_ids)}')

        # Mapping
        news_idx_map = {i: i for i in range(len(df))}
        tweet_idx_offset = len(df)
        tweet_idx_map = {tid: (i + tweet_idx_offset) for i, tid in enumerate(all_tweet_ids)}

        # Build edge list (news_idx, tweet_idx)
        edges: List[List[int]] = []
        for i, lst in enumerate(df['tweet_ids']):
            for t in lst:
                if t in tweet_idx_map:
                    edges.append((i, tweet_idx_map[t]))
        edges_arr = np.array(edges, dtype=np.int64)

        # Build tweet features
        tweet_feats = np.zeros((len(all_tweet_ids), self.tweet_feat_dim), dtype=np.float32)
        for i, tid in enumerate(all_tweet_ids):
            tweet_feats[i] = deterministic_vector_from_id(tid, dim=self.tweet_feat_dim)

        # Save artifacts
        edges_path = os.path.join(self.out_dir, 'edges.npy')
        np.save(edges_path, edges_arr)

        tweet_feat_path = os.path.join(self.out_dir, 'tweet_features.npy')
        np.save(tweet_feat_path, tweet_feats)

        # 3) mappings
        maps = {
            'news_index_to_id': {i: df.iloc[i]['id'] for i in range(len(df))},
            'tweet_id_to_index': tweet_idx_map,
            'num_news': len(df),
            'num_tweets': len(all_tweet_ids),
            'edge_count': int(edges_arr.shape[0])
        }
        with open(os.path.join(self.out_dir, 'node_maps.json'), 'w') as f:
            json.dump(maps, f)

        print('Saved edges to', edges_path)
        print('Saved news features to', news_feat_path)
        print('Saved tweet features to', tweet_feat_path)
        print('Saved node maps to', os.path.join(self.out_dir, 'node_maps.json'))


if __name__ == '__main__':
    builder = EngagementGraphBuilder(data_dir='dataset', out_dir='graph_data')
    builder.build()
