import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from fake_news_data_loader import FakeNewsDataset
from train_utils import NewsDataset, train

class TextOnlyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    # Load data
    data_dir = 'dataset'
    graph_data_dir = 'graph_data'
    
    # Load news features and labels
    news_features = np.load(os.path.join(graph_data_dir, 'news_features.npy'))
    with open(os.path.join(graph_data_dir, 'node_maps.json'), 'r') as f:
        node_maps = json.load(f)
    
    # Get labels in the same order as features
    ds = FakeNewsDataset(data_dir)
    df = ds.get_all()
    df = ds.preprocess(df)  # Important: apply same preprocessing
    
    # Verify we have same number of features as news items
    assert len(news_features) == len(df), f"Mismatch: {len(news_features)} features but {len(df)} labels"
    labels = df['label'].values
    
    print(f"Data loaded: {len(labels)} samples, {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        news_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextOnlyClassifier(input_dim=news_features.shape[1]).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 50
    
    # Train
    train_metrics, val_metrics = train(
        model, train_loader, val_loader, criterion, optimizer, 
        n_epochs, device
    )
    
    print('\nFinal validation metrics:')
    print(f"Accuracy: {val_metrics['acc']:.4f}")
    print(f"F1 Score: {val_metrics['f1']:.4f}")
    print(f"AUC-ROC: {val_metrics['auc']:.4f}")
    
    # Save model
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'text_only_classifier.pt'))

if __name__ == '__main__':
    main()