"""
Text embedding extraction module for fake news detection.
Implements spaCy and BERT embeddings as described in the paper.
"""
import os
import pickle
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm

from ..utils.logging import get_logger
from ..utils.paths import PROCESSED_DIR

logger = get_logger(__name__)

class SpacyEmbedder:
    """
    spaCy-based text embedding extractor.
    Uses pre-trained word2vec vectors (300-dimensional) as described in the paper.
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        self.model_name = model_name
        self.nlp = None
        self._load_model()
        logger.info(f"Initialized SpacyEmbedder with {model_name}")
    
    def _load_model(self):
        """Load spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(f"spaCy model {self.model_name} not found. Please install it:")
            logger.error(f"python -m spacy download {self.model_name}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Extract 300-dimensional spaCy embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            300-dimensional numpy array
        """
        if not text or pd.isna(text):
            return np.zeros(300)
        
        doc = self.nlp(str(text))
        
        # Use mean of word vectors (as mentioned in paper)
        if len(doc) > 0:
            return doc.vector
        else:
            return np.zeros(300)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Extract embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of shape (len(texts), 300)
        """
        logger.info(f"Extracting spaCy embeddings for {len(texts)} texts...")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="spaCy embedding"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class BERTEmbedder:
    """
    BERT-based text embedding extractor.
    Uses pre-trained BERT model for 768-dimensional contextual embeddings.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        logger.info(f"Initialized BERTEmbedder with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load BERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERT model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Extract 768-dimensional BERT embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            768-dimensional numpy array
        """
        if not text or pd.isna(text):
            return np.zeros(768)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            str(text),
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing (smaller for BERT)
            
        Returns:
            Array of shape (len(texts), 768)
        """
        logger.info(f"Extracting BERT embeddings for {len(texts)} texts...")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class EmbeddingExtractor:
    """
    Main embedding extraction class that combines spaCy and BERT.
    Implements the three feature combinations from the paper:
    1. spaCy only
    2. BERT only  
    3. spaCy + Profile (to be added in Phase 3)
    """
    
    def __init__(self, spacy_model: str = 'en_core_web_sm', 
                 bert_model: str = 'bert-base-uncased'):
        self.spacy_embedder = SpacyEmbedder(spacy_model)
        self.bert_embedder = BERTEmbedder(bert_model)
        logger.info("Initialized EmbeddingExtractor")
    
    def extract_spacy_embeddings(self, texts: List[str], 
                                save_path: Optional[str] = None) -> np.ndarray:
        """Extract spaCy embeddings and optionally save them."""
        embeddings = self.spacy_embedder.embed_batch(texts)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved spaCy embeddings to {save_path}")
        
        return embeddings
    
    def extract_bert_embeddings(self, texts: List[str], 
                               save_path: Optional[str] = None) -> np.ndarray:
        """Extract BERT embeddings and optionally save them."""
        embeddings = self.bert_embedder.embed_batch(texts)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved BERT embeddings to {save_path}")
        
        return embeddings
    
    def extract_all_embeddings(self, texts: List[str], 
                              base_save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract both spaCy and BERT embeddings.
        
        Args:
            texts: List of input texts
            base_save_path: Base path for saving (will add _spacy.pkl, _bert.pkl)
            
        Returns:
            Dictionary with 'spacy' and 'bert' embeddings
        """
        logger.info(f"Extracting all embeddings for {len(texts)} texts...")
        
        results = {}
        
        # Extract spaCy embeddings
        spacy_path = f"{base_save_path}_spacy.pkl" if base_save_path else None
        results['spacy'] = self.extract_spacy_embeddings(texts, spacy_path)
        
        # Extract BERT embeddings
        bert_path = f"{base_save_path}_bert.pkl" if base_save_path else None
        results['bert'] = self.extract_bert_embeddings(texts, bert_path)
        
        logger.info("Completed extracting all embeddings")
        return results


def load_embeddings(file_path: str) -> np.ndarray:
    """Load embeddings from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create synthetic data for testing the embedding pipeline.
    This replaces the need for real FakeNewsNet data during development.
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples...")
    
    # Sample news titles and content
    fake_titles = [
        "Breaking: Scientists discover miracle cure for all diseases",
        "Shocking: Government hiding alien contact for decades", 
        "Exclusive: Celebrity caught in major scandal",
        "Urgent: Stock market crash predicted by insider",
        "Revealed: Secret documents expose corruption"
    ]
    
    real_titles = [
        "New study shows benefits of regular exercise",
        "Weather forecast predicts sunny weekend ahead",
        "Local school board announces new policies",
        "Technology company reports quarterly earnings",
        "City council approves infrastructure improvements"
    ]
    
    fake_content = [
        "This groundbreaking discovery will revolutionize medicine forever. The cure works instantly and has no side effects. Share this immediately before big pharma suppresses it!",
        "Classified documents reveal that aliens have been visiting Earth for years. The government has been covering this up to maintain control over the population.",
        "Exclusive footage shows the celebrity engaging in illegal activities. This will change everything you thought you knew about them.",
        "Insider sources predict a massive stock market crash within days. Sell everything now to protect your investments!",
        "Leaked documents expose widespread corruption at the highest levels of government. This is bigger than Watergate!"
    ]
    
    real_content = [
        "A comprehensive study published in a peer-reviewed journal demonstrates the positive effects of regular physical activity on cardiovascular health and longevity.",
        "Meteorologists predict clear skies and warm temperatures for the upcoming weekend, making it perfect for outdoor activities.",
        "The school board announced new policies focused on improving student safety and academic performance based on community feedback.",
        "The technology company reported strong quarterly earnings, exceeding analyst expectations due to increased demand for their products.",
        "City council members voted unanimously to approve funding for infrastructure improvements including road repairs and public transportation upgrades."
    ]
    
    data = []
    for i in range(n_samples):
        is_fake = i < n_samples // 2
        
        if is_fake:
            title = np.random.choice(fake_titles)
            content = np.random.choice(fake_content)
            label = 1  # Fake news
        else:
            title = np.random.choice(real_titles)
            content = np.random.choice(real_content)
            label = 0  # Real news
        
        data.append({
            'id': f'synthetic_{i}',
            'title': title,
            'content': content,
            'label': label,
            'tweet_ids': f'tweet_{i}_1,tweet_{i}_2'  # Mock tweet IDs
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic dataset: {len(df)} samples, {df['label'].sum()} fake, {len(df) - df['label'].sum()} real")
    return df


if __name__ == "__main__":
    # Test the embedding extractors
    extractor = EmbeddingExtractor()
    
    # Create synthetic data
    df = create_synthetic_data(10)
    print("Sample data:")
    print(df[['title', 'label']].head())
    
    # Extract embeddings
    texts = df['title'].tolist()
    embeddings = extractor.extract_all_embeddings(texts)
    
    print(f"\nspaCy embeddings shape: {embeddings['spacy'].shape}")
    print(f"BERT embeddings shape: {embeddings['bert'].shape}")

