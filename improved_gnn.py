import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImprovedNewsGNN(torch.nn.Module):
    def __init__(self, news_feat_dim, tweet_feat_dim, hidden_dim=128, num_heads=4, dropout=0.2, att_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node type embeddings
        self.news_type_emb = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.tweet_type_emb = torch.nn.Parameter(torch.randn(1, hidden_dim))
        
        # Feature encoders with layer normalization
        self.news_encoder = torch.nn.Sequential(
            torch.nn.Linear(news_feat_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.tweet_encoder = torch.nn.Sequential(
            torch.nn.Linear(tweet_feat_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # GAT layers with attention dropout and layer normalization
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, 
                           heads=num_heads, dropout=att_dropout,
                           add_self_loops=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, 
                           heads=num_heads, dropout=att_dropout,
                           add_self_loops=True)
        
        # Layer normalization after GAT
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        
        # MLP classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x_news, x_tweets, edge_index):
        # Add node type embeddings
        h_news = self.news_encoder(x_news) + self.news_type_emb
        h_tweets = self.tweet_encoder(x_tweets) + self.tweet_type_emb
        
        # Combine node features
        x = torch.cat([h_news, h_tweets], dim=0)
        
        # First GAT layer with residual connection and layer norm
        identity = x
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + identity  # Residual connection
        x = self.norm1(x)
        
        # Second GAT layer with residual connection and layer norm
        identity = x
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + identity  # Residual connection
        x = self.norm2(x)
        
        # Get news node embeddings for classification
        news_embeddings = x[:h_news.size(0)]
        
        # Classify
        return self.classifier(news_embeddings)

class BipartiteData(Data):
    def __init__(self, x_news=None, x_tweets=None, edge_index=None, y=None):
        super().__init__()
        self.x_news = x_news
        self.x_tweets = x_tweets
        self.edge_index = edge_index
        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_news.size(0)], 
                               [self.x_tweets.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def load_pyg_dataset(graph_data_dir: str = 'graph_data'):
    """Load the bipartite graph data into PyG format."""
    # Load features and edges
    news_features = np.load(os.path.join(graph_data_dir, 'news_features.npy'))
    tweet_features = np.load(os.path.join(graph_data_dir, 'tweet_features.npy'))
    edges = np.load(os.path.join(graph_data_dir, 'edges.npy'))
    
    with open(os.path.join(graph_data_dir, 'node_maps.json'), 'r') as f:
        node_maps = json.load(f)
    
    # Generate labels from node_maps index to ID mapping (1 for fake, 0 for real).
    # The node IDs in `node_maps` do not contain 'fake'/'real' substrings, so
    # derive the label set by reading the original dataset CSVs which are
    # already split into fake/real files.
    import csv
    import sys
    # Increase CSV field size limit to handle very long tweet_id fields
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        # Fall back to a large constant if platform does not allow maxsize
        try:
            csv.field_size_limit(10**7)
        except Exception:
            pass

    fake_ids = set()
    # Look for known dataset files that contain the fake samples
    for fname in ('dataset/gossipcop_fake.csv', 'dataset/politifact_fake.csv'):
        try:
            with open(fname, newline='', encoding='utf-8') as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    # dataset CSVs use the column 'id' for the news identifier
                    if 'id' in row and row['id']:
                        fake_ids.add(row['id'])
        except FileNotFoundError:
            # If a dataset file is missing, continue; the code will still work
            # but we should surface an informative message below.
            print(f"Warning: dataset file not found: {fname}")

    news_ids = [node_maps['news_index_to_id'][str(i)] for i in range(len(news_features))]
    labels = np.array([1 if nid in fake_ids else 0 for nid in news_ids])
    # Print distribution to help debugging (will be visible when loading dataset)
    try:
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"Label distribution (label:count): {dist}")
    except Exception:
        pass
    
    # Convert to torch tensors
    x_news = torch.FloatTensor(news_features)
    x_tweets = torch.FloatTensor(tweet_features)
    edge_index = torch.LongTensor(edges.T)
    y = torch.LongTensor(labels)
    
    return x_news, x_tweets, edge_index, y

def plot_training_curves(train_metrics, val_metrics, save_path='training_curves.png'):
    epochs = range(1, len(train_metrics['loss']) + 1)
    metrics = ['loss', 'acc', 'f1', 'auc']
    titles = ['Loss', 'Accuracy', 'F1 Score', 'AUC-ROC']
    
    plt.figure(figsize=(15, 10))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, train_metrics[metric], label='Train')
        plt.plot(epochs, val_metrics[metric], label='Validation')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_evaluation_metrics(model, val_loader, device, save_dir='evaluation'):
    """Generate and save comprehensive evaluation plots."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
    from sklearn.metrics import classification_report, roc_curve, auc
    import csv
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x_news, batch.x_tweets, batch.edge_index)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Save metrics and plots inside try/except so headless/plotting errors won't crash
    try:
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

        # Precision-Recall Curve (for positive class)
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        avg_precision = average_precision_score(all_labels, all_probs[:, 1])
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: plotting failed ({e}). Saved metrics will still be written to disk.")
    
    # Print detailed metrics
    print("\nDetailed Evaluation Metrics:")
    print(f"Average Precision Score: {avg_precision:.4f}")

    # Confusion matrix and class-wise metrics
    try:
        print("\nConfusion Matrix:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = None

    precision_pos = None
    recall_pos = None
    if tp is not None:
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\nClass-wise Metrics:")
    if tn is not None:
        print(f"True Negatives (Real classified as Real): {tn}")
        print(f"False Positives (Real classified as Fake): {fp}")
        print(f"False Negatives (Fake classified as Real): {fn}")
        print(f"True Positives (Fake classified as Fake): {tp}")
        print(f"Precision (When model says Fake, how often is it right?): {precision_pos:.4f}")
        print(f"Recall (What fraction of actual Fake news was caught?): {recall_pos:.4f}")
    else:
        print("Confusion matrix not available.")

    # Classification report
    try:
        cls_report = classification_report(all_labels, all_preds, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
    except Exception as e:
        cls_report = None
        print(f"Warning: could not compute classification report: {e}")

    # Save summary metrics to a CSV and report to a text file
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    report_path = os.path.join(save_dir, 'classification_report.json')
    try:
        # Write a compact CSV of key scalar metrics
        with open(metrics_path, 'w', newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(['metric', 'value'])
            writer.writerow(['average_precision', f"{avg_precision:.6f}"])
            writer.writerow(['roc_auc', f"{roc_auc:.6f}"])
            if precision_pos is not None:
                writer.writerow(['precision_pos', f"{precision_pos:.6f}"])
                writer.writerow(['recall_pos', f"{recall_pos:.6f}"])
            if tn is not None:
                writer.writerow(['tn', tn])
                writer.writerow(['fp', fp])
                writer.writerow(['fn', fn])
                writer.writerow(['tp', tp])

        # Save classification report JSON if available
        if cls_report is not None:
            with open(report_path, 'w') as rf:
                json.dump(cls_report, rf, indent=2)
    except Exception as e:
        print(f"Warning: saving evaluation metrics failed: {e}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'auc': 0}
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in tqdm(train_loader, desc='Training'):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x_news, batch.x_tweets, batch.edge_index)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store predictions and metrics
        probs = F.softmax(out, dim=1).detach().cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)
        metrics['loss'] += loss.item()
    
    # Calculate epoch metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics['loss'] /= len(train_loader)
    metrics['acc'] = accuracy_score(all_labels, all_preds)
    metrics['f1'] = f1_score(all_labels, all_preds)
    metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    
    return metrics

def evaluate(model, val_loader, criterion, device):
    model.eval()
    metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'auc': 0}
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            batch = batch.to(device)
            out = model(batch.x_news, batch.x_tweets, batch.edge_index)
            loss = criterion(out, batch.y)
            
            # Store predictions and metrics
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
            metrics['loss'] += loss.item()
    
    # Calculate epoch metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics['loss'] /= len(val_loader)
    metrics['acc'] = accuracy_score(all_labels, all_preds)
    metrics['f1'] = f1_score(all_labels, all_preds)
    metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=50, patience=5, model_path='models/gnn_model.pt'):
    """Train the model with early stopping and metric tracking."""
    best_val_metrics = None
    counter = 0
    train_history = {'loss': [], 'acc': [], 'f1': [], 'auc': []}
    val_history = {'loss': [], 'acc': [], 'f1': [], 'auc': []}
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        for k, v in train_metrics.items():
            train_history[k].append(v)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        for k, v in val_metrics.items():
            val_history[k].append(v)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Early stopping based on validation loss
        if best_val_metrics is None or val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            counter = 0
            # Ensure target directory exists before saving the model
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
            except Exception:
                pass
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curves
    plot_training_curves(train_history, val_history)
    return train_history, val_history, best_val_metrics

def update_edge_indices(edge_index, train_idx):
    # Create mapping from old to new indices for news nodes
    old_to_new = torch.full((edge_index[0].max().item() + 1,), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(train_idx):
        old_to_new[old_idx] = new_idx
    
    # Get mask for edges connected to training nodes
    mask = old_to_new[edge_index[0]] != -1
    
    # Update edge indices
    new_edge_index = edge_index.clone()
    new_edge_index[0, mask] = old_to_new[edge_index[0, mask]]
    
    # Create a mapping for tweet nodes (continuous indices starting from 0)
    unique_tweet_indices = torch.unique(new_edge_index[1, mask])
    tweet_old_to_new = torch.full((edge_index[1].max().item() + 1,), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(unique_tweet_indices):
        tweet_old_to_new[old_idx] = new_idx
    
    # Update tweet node indices
    new_edge_index[1, mask] = tweet_old_to_new[new_edge_index[1, mask]]
    
    return new_edge_index[:, mask], tweet_old_to_new

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining GNN on {device}")
    
    # Load data
    x_news, x_tweets, edge_index, y = load_pyg_dataset()
    
    # Print dataset stats
    print(f"\nDataset loaded:")
    print(f"- News nodes: {x_news.size(0)} (features: {x_news.size(1)})")
    print(f"- Tweet nodes: {x_tweets.size(0)} (features: {x_tweets.size(1)})")
    print(f"- Edges: {edge_index.size(1)}")
    print(f"- Labels: {y.size(0)} (fake: {y.sum().item()}, real: {(1-y).sum().item()})")
    print(f"- Edge index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
    print(f"- First row max: {edge_index[0].max().item()}")
    print(f"- Second row max: {edge_index[1].max().item()}")
    
    # Split dataset
    torch.manual_seed(42)
    indices = torch.randperm(y.size(0))
    train_size = int(0.8 * y.size(0))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    # Update edge indices for training and validation sets
    train_edge_index, train_tweet_map = update_edge_indices(edge_index, train_idx)
    val_edge_index, val_tweet_map = update_edge_indices(edge_index, val_idx)
    
    # Get used tweet features for each split
    train_tweet_indices = train_tweet_map[train_tweet_map != -1]
    val_tweet_indices = val_tweet_map[val_tweet_map != -1]
    
    train_data = BipartiteData(
        x_news=x_news[train_idx],
        x_tweets=x_tweets[train_tweet_indices],
        edge_index=train_edge_index,
        y=y[train_idx]
    )
    
    val_data = BipartiteData(
        x_news=x_news[val_idx],
        x_tweets=x_tweets[val_tweet_indices],
        edge_index=val_edge_index,
        y=y[val_idx]
    )
    
    print(f"\nSplit sizes:")
    print(f"- Training: {len(train_idx)} nodes, {train_edge_index.size(1)} edges")
    print(f"- Validation: {len(val_idx)} nodes, {val_edge_index.size(1)} edges")

    # Set num_nodes explicitly to suppress PyG warnings (total nodes = news + tweets)
    train_data.num_nodes = int(train_data.x_news.size(0) + train_data.x_tweets.size(0))
    val_data.num_nodes = int(val_data.x_news.size(0) + val_data.x_tweets.size(0))

    # Create dataloaders
    train_loader = DataLoader([train_data], batch_size=1)
    val_loader = DataLoader([val_data], batch_size=1)

    # Initialize improved model
    model = ImprovedNewsGNN(
        news_feat_dim=x_news.size(1),
        tweet_feat_dim=x_tweets.size(1),
        hidden_dim=128,
        num_heads=4,
        dropout=0.2,
        att_dropout=0.1
    ).to(device)
    
    # Training setup
    # Use class-weighted loss to handle class imbalance (fast and low-cost on CPU)
    train_labels = y[train_idx].long()
    unique, counts = torch.unique(train_labels, return_counts=True)
    counts_dict = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
    total = int(train_labels.size(0))
    # Inverse frequency weighting (symmetric): weight = total / (num_classes * count)
    weights = []
    for cls in [0, 1]:
        cnt = counts_dict.get(cls, 0)
        w = float(total) / (2 * cnt) if cnt > 0 else 1.0
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Allow overriding number of epochs and patience via environment variables for quick tests
    num_epochs = int(os.environ.get('NUM_EPOCHS', '50'))
    patience = int(os.environ.get('PATIENCE', '3'))

    # Train model (pass patience through for early stopping)
    train_history, val_history, best_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=num_epochs, patience=patience
    )
    
    # Print final results
    print("\nBest validation metrics:")
    print(f"Accuracy: {best_metrics['acc']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"AUC-ROC: {best_metrics['auc']:.4f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load('models/gnn_model.pt'))
    plot_evaluation_metrics(model, val_loader, device)

if __name__ == "__main__":
    main()