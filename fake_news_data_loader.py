import pandas as pd
from typing import List, Dict
import os

class FakeNewsDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = {
            'gossipcop_fake': 'gossipcop_fake.csv',
            'gossipcop_real': 'gossipcop_real.csv',
            'politifact_fake': 'politifact_fake.csv',
            'politifact_real': 'politifact_real.csv',
        }
        self.data = self._load_all()

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path)
        return df

    def _load_all(self) -> Dict[str, pd.DataFrame]:
        return {k: self._load_csv(v) for k, v in self.files.items()}

    def get_all(self) -> pd.DataFrame:
        dfs = []
        for label, df in self.data.items():
            df = df.copy()
            df['label'] = 1 if 'fake' in label else 0
            df['source'] = 'gossipcop' if 'gossipcop' in label else 'politifact'
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Basic preprocessing: drop missing, clean text, parse tweet_ids
        df = df.dropna(subset=['title', 'tweet_ids']).copy()  # Create explicit copy
        df.loc[:, 'tweet_ids'] = df['tweet_ids'].apply(lambda x: str(x).split())
        df.loc[:, 'title'] = df['title'].astype(str).str.strip()
        return df

if __name__ == "__main__":
    data_dir = "dataset"
    dataset = FakeNewsDataset(data_dir)
    all_data = dataset.get_all()
    all_data = dataset.preprocess(all_data)
    print(all_data.head())
