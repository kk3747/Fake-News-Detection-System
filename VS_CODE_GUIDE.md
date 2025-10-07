# Fake News Detection Project - Complete VS Code Guide

## 🎯 Project Overview
This project implements an **Ensemble Graph Neural Network (EGNN)** for fake news detection using NLP and Graph Neural Networks. We've completed **Phase 3** which includes text preprocessing, embedding extraction, and graph construction.

## 📁 Project Structure in VS Code
```
fake_news_project/
├── 📁 dataset/                          # Your real dataset
│   ├── gossipcop_fake.csv              # 8,000+ fake news articles
│   ├── gossipcop_real.csv              # 8,000+ real news articles
│   ├── politifact_fake.csv             # 300+ fake news articles
│   └── politifact_real.csv             # 300+ real news articles
├── 📁 src/
│   ├── 📁 fake_news/                   # Source code
│   │   ├── 📁 data/                    # Data processing modules
│   │   ├── 📁 features/                # Text preprocessing & embeddings
│   │   ├── 📁 graphs/                  # Graph construction & profile features
│   │   ├── 📁 models/                  # GNN models (Phase 4)
│   │   ├── 📁 training/                # Training pipeline (Phase 5)
│   │   └── 📁 utils/                   # Utilities
│   └── 📁 data/                        # 📊 PROCESSED DATA LOCATION
│       ├── 📁 processed/               # Text embeddings & processed data
│       │   ├── sample_news_processed.csv    # 🎯 MAIN RESULTS FILE
│       │   ├── sample_spacy_embeddings.npy  # spaCy features (300D)
│       │   ├── sample_bert_embeddings.npy   # BERT features (768D)
│       │   ├── sample_f3_features.npy       # Combined features (315D)
│       │   └── sample_metadata.json         # Dataset statistics
│       └── 📁 graphs/                  # 🔗 GRAPH STRUCTURES LOCATION
│           ├── retweet_networks.pkl         # 50 engagement graphs
│           ├── graph_features.pkl           # Graph-level metrics
│           ├── node_features.pkl            # Node feature matrices
│           └── edge_features.pkl            # Edge feature matrices
├── 📁 notebooks/                       # Jupyter notebooks for each phase
│   ├── 02_phase2_text_embeddings.ipynb
│   └── 03_phase3_graph_construction.ipynb
├── 📄 VS_CODE_GUIDE.md                 # This guide
├── 📄 DEMO_FOR_PROFESSOR.md            # Professor presentation guide
├── 📄 show_results.py                  # Results demonstration script
└── 📄 requirements.txt                 # Python dependencies
```

## 🚀 How to Run the Project in VS Code

### Step 1: Open Project in VS Code
1. Open VS Code
2. File → Open Folder
3. Select: `/Users/hemantkumar/Desktop/fake_news_project`

### Step 2: Set Up Python Environment
1. Open VS Code Terminal (Ctrl+` or View → Terminal)
2. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python3 -m spacy download en_core_web_sm
   ```

### Step 3: Run Phase 3 Processing
In VS Code Terminal, run:
```bash
# Set Python path and run processing
export PYTHONPATH=/Users/hemantkumar/Desktop/fake_news_project/src
python3 src/fake_news/graphs/process_sample.py
```

**Expected Output:**
```
2025-10-06 04:19:02,975 | INFO | fake_news.__main__ | Processing sample dataset with 50 articles...
2025-10-06 04:19:03,667 | INFO | fake_news.__main__ | Sample dataset: 50 articles
2025-10-06 04:19:03,668 | INFO | fake_news.__main__ | Fake: 25, Real: 25
...
Sample processing completed successfully!
Processed 50 articles
F3 features shape: (50, 111)
Created 50 engagement graphs
```

### Step 4: View Results
Run the demonstration script:
```bash
python3 show_results.py
```

## 📊 Where to Find Results in VS Code

### 🎯 Main Results File
**Location**: `src/data/processed/sample_news_processed.csv`
- **How to open**: Click on the file in VS Code Explorer
- **Content**: 50 processed news articles with all features
- **Columns**: id, title, label, title_processed, profile_* features

### 🔗 Graph Structures
**Location**: `src/data/graphs/retweet_networks.pkl`
- **How to view**: Use Python script (see below)
- **Content**: 50 NetworkX graphs representing user engagement

### 📈 Feature Files
**Location**: `src/data/processed/`
- `sample_spacy_embeddings.npy` - spaCy features (50×300)
- `sample_bert_embeddings.npy` - BERT features (50×768)
- `sample_f3_features.npy` - Combined features (50×315)

## 🔍 How to View Different Results

### 1. View Processed News Data
Create a new Python file in VS Code and run:
```python
import pandas as pd
import sys
sys.path.append('src')

# Load processed data
df = pd.read_csv('src/data/processed/sample_news_processed.csv')

# View first 5 rows
print("📊 PROCESSED NEWS DATA:")
print(df[['id', 'title', 'label', 'title_processed']].head())

# View dataset summary
print(f"\n📈 DATASET SUMMARY:")
print(f"Total articles: {len(df)}")
print(f"Fake news: {df['label'].sum()}")
print(f"Real news: {len(df) - df['label'].sum()}")
```

### 2. View Graph Structures
Create a new Python file and run:
```python
import pickle
import networkx as nx
import sys
sys.path.append('src')

# Load graphs
with open('src/data/graphs/retweet_networks.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Show graph information
print("🔗 ENGAGEMENT GRAPHS:")
print(f"Total graphs: {len(graphs)}")

# Show sample graph
sample_id = list(graphs.keys())[0]
sample_graph = graphs[sample_id]
print(f"\nSample graph ({sample_id}):")
print(f"Nodes: {sample_graph.number_of_nodes()}")
print(f"Edges: {sample_graph.number_of_edges()}")

# Show node types
node_types = nx.get_node_attributes(sample_graph, 'node_type')
news_nodes = sum(1 for nt in node_types.values() if nt == 'news')
user_nodes = sum(1 for nt in node_types.values() if nt == 'user')
print(f"News nodes: {news_nodes}")
print(f"User nodes: {user_nodes}")
```

### 3. View Embeddings
Create a new Python file and run:
```python
import numpy as np
import sys
sys.path.append('src')

# Load embeddings
spacy_emb = np.load('src/data/processed/sample_spacy_embeddings.npy')
bert_emb = np.load('src/data/processed/sample_bert_embeddings.npy')
f3_features = np.load('src/data/processed/sample_f3_features.npy')

print("🔤 TEXT EMBEDDINGS:")
print(f"spaCy shape: {spacy_emb.shape}")
print(f"BERT shape: {bert_emb.shape}")
print(f"F3 shape: {f3_features.shape}")

print(f"\nspaCy range: [{spacy_emb.min():.3f}, {spacy_emb.max():.3f}]")
print(f"BERT range: [{bert_emb.min():.3f}, {bert_emb.max():.3f}]")
print(f"F3 range: [{f3_features.min():.3f}, {f3_features.max():.3f}]")
```

## 📋 Quick Commands for VS Code Terminal

### Run Complete Demo
```bash
# Set Python path
export PYTHONPATH=/Users/hemantkumar/Desktop/fake_news_project/src

# Run processing
python3 src/fake_news/graphs/process_sample.py

# Show results
python3 show_results.py
```

### Check File Sizes
```bash
ls -la src/data/processed/
ls -la src/data/graphs/
```

### View CSV Content
```bash
head -5 src/data/processed/sample_news_processed.csv
```

## 🎓 What Each Phase Does

### ✅ Phase 1: Project Setup
- Created project structure
- Set up data directories
- Created utility modules

### ✅ Phase 2: Text Preprocessing + Embeddings
- Text cleaning and preprocessing
- spaCy embeddings (300D)
- BERT embeddings (768D)

### ✅ Phase 3: Graph Construction + Profile Features
- User profile feature extraction (15D)
- Engagement graph construction
- F3 features (spaCy + Profile = 315D)

### 🔄 Phase 4: GNN Models (Next)
- GCN, GAT, BiGCN models
- Training pipeline
- Model evaluation

### 🔄 Phase 5: Ensemble Methods (Next)
- Majority voting
- Weighted average
- Stacking ensemble

### 🔄 Phase 6: Evaluation (Next)
- Focal loss implementation
- Performance metrics
- Visualizations

## 📊 Results Summary

### Dataset Processed
- **50 articles** from GossipCop dataset
- **25 fake news**, **25 real news**
- **Text preprocessing**: Lowercase, tokenization, stemming
- **Feature extraction**: spaCy (300D), BERT (768D), Profile (15D)
- **Graph construction**: 50 engagement networks

### Key Results Files
1. **`sample_news_processed.csv`** - Main processed data (125KB)
2. **`retweet_networks.pkl`** - 50 engagement graphs (23KB)
3. **`sample_spacy_embeddings.npy`** - spaCy features (19KB)
4. **`sample_bert_embeddings.npy`** - BERT features (154KB)
5. **`sample_f3_features.npy`** - Combined features (45KB)

## 🔧 Troubleshooting

### If you get import errors:
```bash
export PYTHONPATH=/Users/hemantkumar/Desktop/fake_news_project/src
```

### If you get module not found errors:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn tqdm spacy transformers torch
python3 -m spacy download en_core_web_sm
```

### If you want to process more data:
Edit `src/fake_news/graphs/process_sample.py` and change:
```python
result = process_sample_dataset(100)  # Process 100 articles instead of 50
```

## 📞 Support
- All source code is in `src/fake_news/` directory
- Check `DEMO_FOR_PROFESSOR.md` for professor presentation
- Run `python3 show_results.py` for complete results overview

## ✅ Ready for Phase 4
All data is processed and ready for GNN model training!
The results are located in `src/data/processed/` and `src/data/graphs/` directories.
