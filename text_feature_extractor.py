import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm

class TextFeatureExtractor:
    def __init__(self, model_name: str = 'en_core_web_md'):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        return doc.vector

    def extract_features(self, df: pd.DataFrame, text_column: str = 'title') -> np.ndarray:
        embeddings = []
        for text in tqdm(df[text_column], desc='Extracting text embeddings'):
            embeddings.append(self.get_embedding(str(text)))
        return np.vstack(embeddings)

if __name__ == "__main__":
    from fake_news_data_loader import FakeNewsDataset
    data_dir = "dataset"
    dataset = FakeNewsDataset(data_dir)
    all_data = dataset.get_all()
    all_data = dataset.preprocess(all_data)
    extractor = TextFeatureExtractor()
    text_features = extractor.extract_features(all_data)
    print("Text feature shape:", text_features.shape)
