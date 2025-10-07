# Fake News Detection with NLP + Graph Neural Networks

This project incrementally builds a system inspired by the EGNN approach in the uploaded paper: combining text embeddings (spaCy, BERT) with user engagement graphs (GCN, GAT, BiGCN) and simple profile features, then ensembling the models.

We proceed in small, testable phases. Each phase has a notebook in `notebooks/` and CLI modules in `src/`.

## Phases
1. Data notes + scaffold (you are here)
2. Text preprocessing + embeddings (spaCy, BERT)
3. User engagement graph + profile features
4. GNN models (GCN, GAT, BiGCN) and training loop
5. Ensemble methods (majority, weighted by F1, stacking)
6. Evaluation (ROC/AUC, accuracy, F1, precision, recall, MCC) with focal loss and visualizations

## Environment
- Create a Python 3.10 env and install `requirements.txt`.
- Install PyTorch matching your platform (CPU or CUDA).
- Install PyTorch Geometric following the official instructions for your Torch/CUDA version.

## Data (FakeNewsNet subset)
Place raw CSV files under `data/raw/`:
- `text.csv`: id,title,url,content,label,tweet_ids
- `tweets.csv`: tweet_id,user_id,created_at,text,retweet_of,in_reply_to,lang
- `users.csv`: user_id,screen_name,followers,friends,listed,favourites,statuses,created_at

We will also provide a tiny synthetic sample later to validate the pipeline.

## Running Phase 1
- `python -m fake_news.data.prepare_dataset` to generate schema guides.
# Fake-News-Detection-System
