import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fake_news_data_loader import FakeNewsDataset

class NewsGNN(torch.nn.Module):
    def __init__(self, news_feat_dim, tweet_feat_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.news_encoder = torch.nn.Sequential(
            torch.nn.Linear(news_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        self.tweet_encoder = torch.nn.Sequential(
            torch.nn.Linear(tweet_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # GAT layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        # Output layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x_news, x_tweets, edge_index):
        # Encode features
        h_news = self.news_encoder(x_news)
        h_tweets = self.tweet_encoder(x_tweets)
        
        # Combine node features
        x = torch.cat([h_news, h_tweets], dim=0)
        
        # GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        # Get news node predictions only
        news_embeddings = x[:h_news.size(0)]
        
        # Classify
        out = self.classifier(news_embeddings)
        return out

def load_pyg_dataset(graph_data_dir: str = 'graph_data'):
    """Load the bipartite graph data into PyG format."""
    # Load features and edges
    news_features = np.load(os.path.join(graph_data_dir, 'news_features.npy'))
    tweet_features = np.load(os.path.join(graph_data_dir, 'tweet_features.npy'))
    edges = np.load(os.path.join(graph_data_dir, 'edges.npy'))
    
    with open(os.path.join(graph_data_dir, 'node_maps.json'), 'r') as f:
        node_maps = json.load(f)
    
    # Convert to torch tensors
    x_news = torch.FloatTensor(news_features)
    x_tweets = torch.FloatTensor(tweet_features)
    edge_index = torch.LongTensor(edges.T)  # PyG expects (2, E) shape
    
    return x_news, x_tweets, edge_index, node_maps

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(train_loader, desc='Training'):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x_news, batch.x_tweets, batch.edge_index)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = F.softmax(out, dim=1).detach().cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs)
    
    avg_loss = total_loss / len(train_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'loss': avg_loss,
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs[:, 1])
    }
    return metrics

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            batch = batch.to(device)
            out = model(batch.x_news, batch.x_tweets, batch.edge_index)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs)
    
    avg_loss = total_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'loss': avg_loss,
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs[:, 1])
    }
    return metrics

def main():
    # Load data
    data_dir = 'dataset'
    graph_data_dir = 'graph_data'
    
    # Load graph data
    x_news, x_tweets, edge_index, node_maps = load_pyg_dataset(graph_data_dir)
    
    # Get labels
    ds = FakeNewsDataset(data_dir)
    df = ds.get_all()
    df = ds.preprocess(df)
    labels = torch.LongTensor(df['label'].values)
    
    print(f"Dataset loaded:")
    print(f"- News nodes: {len(x_news)} (features: {x_news.size(1)})")
    print(f"- Tweet nodes: {len(x_tweets)} (features: {x_tweets.size(1)})")
    print(f"- Edges: {edge_index.size(1)}")
    print(f"- Labels: {len(labels)} ({sum(labels).item()} positive)")
    
    # Create PyG Data objects
    # We'll create a list of subgraphs for batching
    # Each subgraph will contain a news node and its direct tweet neighbors
    subgraphs = []
    for news_idx in range(len(x_news)):
        # Find edges containing this news node
        edge_mask = edge_index[0] == news_idx
        local_edges = edge_index[:, edge_mask]
        
        # Get tweet nodes connected to this news
        tweet_indices = local_edges[1]
        
        # Create local features
        local_x_news = x_news[news_idx].unsqueeze(0)
        local_x_tweets = x_tweets[tweet_indices - len(x_news)]  # Adjust indices
        
        # Create local edge indices (reindex tweets to start from 1)
        local_edge_index = torch.stack([
            torch.zeros_like(tweet_indices),  # News node is always 0
            torch.arange(1, len(tweet_indices) + 1)  # Tweet nodes start from 1
        ])
        
        # Create Data object
        data = Data(
            x_news=local_x_news,
            x_tweets=local_x_tweets,
            edge_index=local_edge_index,
            y=labels[news_idx]
        )
        subgraphs.append(data)
    
    # Split into train/val
    train_graphs, val_graphs = train_test_split(
        subgraphs, test_size=0.2, random_state=42,
        stratify=[g.y.item() for g in subgraphs]
    )
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsGNN(
        news_feat_dim=x_news.size(1),
        tweet_feat_dim=x_tweets.size(1)
    ).to(device)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    
    # Train
    print(f"\nTraining GNN on {device}")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        # Compute metrics
        train_acc = (np.array(train_preds) == np.array(train_labels)).mean()
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
        
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/gnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    print("Training complete!")

if __name__ == '__main__':
    main()