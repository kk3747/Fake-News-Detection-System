# Quick Reference - Fake News Detection Project

## 🚀 Quick Start Commands

### 1. Open Project in VS Code
```bash
cd /Users/hemantkumar/Desktop/fake_news_project
code .
```

### 2. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 3. Run Processing
```bash
export PYTHONPATH=/Users/hemantkumar/Desktop/fake_news_project/src
python3 src/fake_news/graphs/process_sample.py
```

### 4. View Results
```bash
python3 show_results.py
```

## 📁 Key File Locations

| File | Location | Description |
|------|----------|-------------|
| **Main Results** | `src/data/processed/sample_news_processed.csv` | 50 processed articles with features |
| **Graphs** | `src/data/graphs/retweet_networks.pkl` | 50 engagement graphs |
| **spaCy Features** | `src/data/processed/sample_spacy_embeddings.npy` | 300D word2vec features |
| **BERT Features** | `src/data/processed/sample_bert_embeddings.npy` | 768D contextual features |
| **F3 Features** | `src/data/processed/sample_f3_features.npy` | 315D combined features |

## 📊 Results Summary

- **50 articles processed** (25 fake, 25 real)
- **Text preprocessing**: Lowercase, tokenization, stemming
- **Feature extraction**: spaCy (300D), BERT (768D), Profile (15D)
- **Graph construction**: 50 engagement networks
- **F3 features**: 315D combined features

## 🔍 View Results in VS Code

### View CSV Data
```python
import pandas as pd
df = pd.read_csv('src/data/processed/sample_news_processed.csv')
print(df[['id', 'title', 'label', 'title_processed']].head())
```

### View Graphs
```python
import pickle
with open('src/data/graphs/retweet_networks.pkl', 'rb') as f:
    graphs = pickle.load(f)
print(f"Total graphs: {len(graphs)}")
```

### View Embeddings
```python
import numpy as np
spacy_emb = np.load('src/data/processed/sample_spacy_embeddings.npy')
print(f"spaCy shape: {spacy_emb.shape}")
```

## 🎯 What's Next

- **Phase 4**: GNN Models (GCN, GAT, BiGCN)
- **Phase 5**: Ensemble Methods
- **Phase 6**: Evaluation & Visualization

## 📞 Help

- Full guide: `VS_CODE_GUIDE.md`
- Professor demo: `DEMO_FOR_PROFESSOR.md`
- Show results: `python3 show_results.py`
