import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

class NewsDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels in tqdm(train_loader, desc='Training'):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_preds)
    }
    return metrics

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='Evaluating'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_preds)
    }
    return metrics

def train(model, train_loader, val_loader, criterion, optimizer, 
          n_epochs, device, early_stop_patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["acc"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}, AUC: {train_metrics["auc"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["acc"]:.4f}, '
              f'F1: {val_metrics["f1"]:.4f}, AUC: {val_metrics["auc"]:.4f}')
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_metrics, val_metrics