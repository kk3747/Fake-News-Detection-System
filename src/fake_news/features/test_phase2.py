#!/usr/bin/env python3
"""
Test script for Phase 2: Text Preprocessing + Embeddings
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_news.features.text_preprocessing import TextPreprocessor, preprocess_news_data
from fake_news.features.embeddings import EmbeddingExtractor, create_synthetic_data
from fake_news.utils.logging import get_logger
from fake_news.utils.paths import PROCESSED_DIR

def main():
    logger = get_logger('phase2_test')
    logger.info("Starting Phase 2 test...")
    
    # Create synthetic data
    logger.info("Creating synthetic dataset...")
    df = create_synthetic_data(n_samples=50)  # Small dataset for testing
    print(f"Created dataset: {len(df)} samples")
    print(f"Fake: {df['label'].sum()}, Real: {len(df) - df['label'].sum()}")
    
    # Test text preprocessing
    logger.info("Testing text preprocessing...")
    preprocessor = TextPreprocessor()
    
    test_text = "This is a FAKE news article!!! It contains @mentions and #hashtags."
    processed = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
    
    # Preprocess dataset
    df_processed = preprocess_news_data(df, text_columns=['title', 'content'])
    print(f"Processed dataset shape: {df_processed.shape}")
    
    # Test embedding extraction
    logger.info("Testing embedding extraction...")
    try:
        extractor = EmbeddingExtractor()
        
        # Extract spaCy embeddings
        logger.info("Extracting spaCy embeddings...")
        spacy_embeddings = extractor.extract_spacy_embeddings(
            df_processed['title_processed'].tolist()
        )
        print(f"spaCy embeddings shape: {spacy_embeddings.shape}")
        
        # Extract BERT embeddings
        logger.info("Extracting BERT embeddings...")
        bert_embeddings = extractor.extract_bert_embeddings(
            df_processed['title_processed'].tolist()
        )
        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        
        # Save embeddings
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(PROCESSED_DIR / 'test_spacy_embeddings.pkl', 'wb') as f:
            pickle.dump(spacy_embeddings, f)
        with open(PROCESSED_DIR / 'test_bert_embeddings.pkl', 'wb') as f:
            pickle.dump(bert_embeddings, f)
        
        logger.info("Phase 2 test completed successfully!")
        print("✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"Phase 2 test failed: {e}")
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

