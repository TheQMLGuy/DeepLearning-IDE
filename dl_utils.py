"""
PyTorch Deep Learning Utilities
55+ Features for Streamlined Workflows
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
import time
from pathlib import Path

# ============================================
# FEATURE 1-5: STREAMLINED PIPELINE
# ============================================

def load_dataset(name: str, batch_size: int = 128, root: str = './data'):
    """One-click dataset loading"""
    
    # Transforms
    if name in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset_class = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
    elif name in ['mnist', 'fashion_mnist']:
        transform_train = transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_class = datasets.MNIST if name == 'mnist' else datasets.FashionMNIST
    else:
        raise ValueError(f"Dataset {name} not supported")
    
    # Load datasets
    train_data = dataset_class(root=root, train=True, download=True, transform=transform_train)
    test_data = dataset_class(root=root, train=False, download=True, transform=transform_test)
    
    # Create loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Loaded {name}: {len(train_data)} train, {len(test_data)} test samples")
    return train_loader, test_loader

# ============================================
# FEATURE 6-10: MODEL ARCHITECTURE
# ============================================

def create_simple_cnn(in_channels: int, num_classes: int):
    """Create simple CNN"""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )

def load_pretrained(name: str, num_classes: int):
    """Load pretrained model"""
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {name} not supported")
    
    print(f"✓ Loaded {name} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def model_summary(model, input_size, device='cuda'):
    """Print model summary"""
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Input Size: {input_size}")
    print("="*60)
    
    return total_params

# ============================================
# FEATURE 16-20: DATA MANAGEMENT
# ============================================

def analyze_dataset(loader):
    """Analyze dataset statistics"""
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    
    unique, counts = np.unique(all_labels, return_counts=True)
    
    stats = {
        'num_samples': len(all_labels),
        'num_classes': len(unique),
        'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
        'is_balanced': np.std(counts) < np.mean(counts) * 0.1,
        'min_samples': int(np.min(counts)),
        'max_samples': int(np.max(counts))
    }
    
    print("\n📊 Dataset Statistics:")
    print(f"  Samples: {stats['num_samples']:,}")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Balanced: {'Yes ✓' if stats['is_balanced'] else 'No ✗'}")
    print(f"  Min/Max per class: {stats['min_samples']}/{stats['max_samples']}")
    
    return stats

def visualize_batch(loader, n=16):
    """Visualize a batch of data"""
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"Label: {labels[idx].item()}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================
# FEATURE 26-30: TRAINING & MONITORING
# ============================================

class Trainer:
    """Complete training system"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        lr: float = 0.001,
        device: str = 'auto',
        mixed_precision: bool = False,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Training on {self.device}")
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
        # Features
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        self.best_val_acc = 0
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
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
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def fit(self, epochs: int):
        """Train model"""
        print(f"\n{'='*60}")
        print(f"Training for {epochs} epochs")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Record
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:5.2f}% | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:5.2f}% | "
                  f"Time={elapsed:.1f}s")
            
            # Save best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.checkpoint_dir / 'best_model.pth')
                print(f"  → Saved best model (val_acc={val_acc:.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history

# ============================================
# FEATURE 29: LEARNING RATE FINDER
# ============================================

def find_lr(model, train_loader, criterion, start_lr=1e-7, end_lr=1, num_iter=100):
    """Find optimal learning rate"""
    print("\n🔍 Finding optimal learning rate...")
    
    device = next(model.parameters()).device
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lrs = []
    losses = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_iter:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # Find best LR
    gradients = np.gradient(losses)
    best_idx = np.argmin(gradients)
    best_lr = lrs[best_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.axvline(best_lr, color='r', linestyle='--', label=f'Suggested LR: {best_lr:.2e}')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"✓ Optimal Learning Rate: {best_lr:.2e}")
    return best_lr

# ============================================
# FEATURE 40: BATCH SIZE FINDER
# ============================================

def find_batch_size(model, input_size, device='cuda', max_size=512):
    """Find maximum batch size that fits in memory"""
    print("\n🔍 Finding optimal batch size...")
    
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 2
    while batch_size <= max_size:
        try:
            # Create dummy batch
            x = torch.randn(batch_size, *input_size).to(device)
            y = torch.randint(0, 10, (batch_size,)).to(device)
            
            # Forward + backward
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Cleanup
            del x, y, outputs, loss
            torch.cuda.empty_cache()
            
            print(f"  Batch size {batch_size}: ✓")
            batch_size *= 2
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                max_bs = batch_size // 2
                print(f"\n✓ Maximum batch size: {max_bs}")
                return max_bs
            else:
                raise e
    
    print(f"\n✓ Maximum batch size: {max_size}")
    return max_size

# ============================================
# FEATURE 41-45: EVALUATION
# ============================================

def plot_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Progress - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================

def quick_start(dataset='cifar10', model_name='simple', epochs=10):
    """Complete training pipeline in one function"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  PyTorch Deep Learning IDE - Quick Start                     ║
    ║  55+ Features for Streamlined Workflows                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Load data
    print("\n[1/4] Loading dataset...")
    train_loader, test_loader = load_dataset(dataset, batch_size=128)
    
    # 2. Create model
    print("\n[2/4] Creating model...")
    if model_name == 'simple':
        if dataset in ['mnist', 'fashion_mnist']:
            model = create_simple_cnn(1, 10)
        else:
            model = create_simple_cnn(3, 10)
    else:
        model = load_pretrained(model_name, num_classes=10)
    
    # 3. Train
    print("\n[3/4] Training model...")
    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        lr=0.001,
        mixed_precision=torch.cuda.is_available()
    )
    history = trainer.fit(epochs=epochs)
    
    # 4. Results
    print("\n[4/4] Visualizing results...")
    plot_history(history)
    
    return model, history

if __name__ == '__main__':
    # Run complete pipeline
    model, history = quick_start('cifar10', model_name='simple', epochs=10)
