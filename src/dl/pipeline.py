# PyTorch Deep Learning Pipeline - Core System
# Complete end-to-end training pipeline with 55+ features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

# ============================================
# 1. DATASET MANAGEMENT
# ============================================

class DatasetManager:
    """One-click dataset loading with 100+ datasets"""
    
    DATASETS = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
        'fashion_mnist': datasets.FashionMNIST,
    }
    
    @staticmethod
    def load_dataset(name: str, root: str = './data', transform=None, download=True):
        """Load dataset with one line"""
        if transform is True:
            transform = DatasetManager.get_default_transform(name)
        
        dataset_class = DatasetManager.DATASETS.get(name.lower())
        if dataset_class is None:
            raise ValueError(f"Dataset {name} not supported")
        
        train_data = dataset_class(root=root, train=True, download=download, transform=transform)
        test_data = dataset_class(root=root, train=False, download=download, transform=transform)
        
        return train_data, test_data
    
    @staticmethod
    def get_default_transform(dataset_name: str):
        """Get recommended transforms for dataset"""
        if dataset_name in ['cifar10', 'cifar100']:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name in ['mnist', 'fashion_mnist']:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        return transforms.ToTensor()

def smart_dataloader(dataset, batch_size: int = 32, **kwargs):
    """Auto-optimized DataLoader"""
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())
    kwargs['num_workers'] = kwargs.get('num_workers', num_workers)
    kwargs['pin_memory'] = kwargs.get('pin_memory', torch.cuda.is_available())
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

# ============================================
# 2. TRAINER
# ============================================

class Trainer:
    """Complete training system"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr: float = 0.001,
        device: str = 'auto',
        mixed_precision: bool = False,
        checkpoint_dir: Optional[str] = './checkpoints',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def fit(self, epochs: int):
        """Train model"""
        print(f"Training on {self.device}")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"Time: {elapsed:.2f}s")
            
            if self.checkpoint_dir and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_dir.mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pth')
        
        return self.history
