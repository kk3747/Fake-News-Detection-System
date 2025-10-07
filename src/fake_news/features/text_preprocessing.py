"""
Text preprocessing module for fake news detection.
Implements cleaning, tokenization, and normalization as described in the paper.
"""
import re
import string
from typing import List, Optional
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """
    Text preprocessing pipeline following the paper's methodology.
    
    Steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters
    3. Tokenize using NLTK word_tokenize
    4. Remove stop words
    5. Apply Porter stemming
    """
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        logger.info(f"Initialized TextPreprocessor for {language}")
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing non-alphabetic characters and normalizing."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove non-alphabetic characters, keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize."""
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list."""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text as single string
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming
        tokens = self.stem_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        logger.info(f"Preprocessing {len(texts)} texts...")
        processed = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            processed.append(self.preprocess(text))
        
        logger.info(f"Completed preprocessing {len(texts)} texts")
        return processed


def preprocess_news_data(df: pd.DataFrame, 
                        text_columns: List[str] = ['title', 'content'],
                        preprocessor: Optional[TextPreprocessor] = None) -> pd.DataFrame:
    """
    Preprocess news data following the paper's methodology.
    
    Args:
        df: DataFrame with news articles
        text_columns: Columns to preprocess
        preprocessor: Optional preprocessor instance
        
    Returns:
        DataFrame with preprocessed text columns
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    df_processed = df.copy()
    
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Preprocessing column: {col}")
            df_processed[f'{col}_processed'] = preprocessor.preprocess_batch(df[col].fillna('').tolist())
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    
    return df_processed


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_text = "This is a FAKE news article!!! It contains @mentions and #hashtags. Let's test it!"
    print(f"Original: {test_text}")
    print(f"Processed: {preprocessor.preprocess(test_text)}")

