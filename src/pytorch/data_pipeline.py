"""
PyTorch Data Pipeline - Smart Data Loading & Augmentation
Features: Auto data loaders, augmentation preview, dataset statistics, imbalanced handling
"""

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from typing import Optional, Callable, List, Tuple, Dict
import numpy as np
from collections import Counter


class SmartDataLoader:
    """Intelligent DataLoader with automatic optimization."""
    
    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create optimized DataLoader.
        
        Auto-determines:
        - num_workers based on CPU count
        - pin_memory based on CUDA availability
        """
        if num_workers is None:
            import multiprocessing
            num_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8
        
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0
        )
        
        print(f"✓ DataLoader created:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Pin memory: {pin_memory}")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Batches per epoch: {len(loader)}")
        
        return loader


class AugmentationPipeline:
    """Pre-built augmentation pipelines for common tasks."""
    
    @staticmethod
    def image_classification_train(img_size: int = 224):
        """Standard augmentation for image classification."""
        return T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def image_classification_val(img_size: int = 224):
        """Validation augmentation (minimal)."""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def strong_augmentation(img_size: int = 224):
        """Strong augmentation for difficult datasets."""
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            T.RandomGrayscale(p=0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def cutout(img_size: int = 224, n_holes: int = 1, length: int = 16):
        """Cutout augmentation."""
        class Cutout:
            def __init__(self, n_holes, length):
                self.n_holes = n_holes
                self.length = length
            
            def __call__(self, img):
                h, w = img.size(1), img.size(2)
                mask = np.ones((h, w), np.float32)
                
                for _ in range(self.n_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)
                    
                    y1 = np.clip(y - self.length // 2, 0, h)
                    y2 = np.clip(y + self.length // 2, 0, h)
                    x1 = np.clip(x - self.length // 2, 0, w)
                    x2 = np.clip(x + self.length // 2, 0, w)
                    
                    mask[y1:y2, x1:x2] = 0.
                
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask
                
                return img
        
        return Cutout(n_holes, length)
    
    @staticmethod
    def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
        """Apply MixUp augmentation to a batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class DatasetAnalyzer:
    """Analyze dataset properties and quality."""
    
    @staticmethod
    def analyze(dataset: Dataset, num_samples: int = 1000) -> Dict:
        """
        Comprehensive dataset analysis.
        
        Returns:
            Dict with statistics: class distribution, mean, std, etc.
        """
        print("🔍 Analyzing dataset...")
        
        # Sample random indices
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        # Collect labels
        labels = []
        images = []
        
        for idx in indices:
            if isinstance(dataset[idx], tuple):
                img, label = dataset[idx]
            else:
                img = dataset[idx]
                label = -1
            
            labels.append(label)
            if isinstance(img, torch.Tensor):
                images.append(img.numpy())
        
        # Class distribution
        label_counts = Counter(labels)
        
        # Image statistics
        if images:
            images = np.array(images)
            mean = np.mean(images, axis=(0, 2, 3))
            std = np.std(images, axis=(0, 2, 3))
        else:
            mean, std = None, None
        
        stats = {
            'size': len(dataset),
            'num_classes': len(label_counts),
            'class_distribution': dict(label_counts),
            'is_balanced': max(label_counts.values()) / min(label_counts.values()) < 1.5 if label_counts else True,
            'mean': mean.tolist() if mean is not None else None,
            'std': std.tolist() if std is not None else None
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print("Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total samples: {stats['size']}")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"Balanced: {'Yes' if stats['is_balanced'] else 'No'}")
        
        if stats['class_distribution']:
            print("\nClass Distribution:")
            for label, count in sorted(stats['class_distribution'].items()):
                pct = 100 * count / sum(stats['class_distribution'].values())
                print(f"  Class {label}: {count:5d} ({pct:5.1f}%)")
        
        if mean is not None:
            print(f"\nChannel means: {mean}")
            print(f"Channel stds:  {std}")
        print(f"{'='*50}\n")
        
        return stats
    
    @staticmethod
    def find_corrupted(dataset: Dataset) -> List[int]:
        """Find corrupted or invalid samples."""
        corrupted = []
        
        print("🔍 Checking for corrupted samples...")
        
        for idx in range(len(dataset)):
            try:
                item = dataset[idx]
                if isinstance(item, tuple):
                    img, label = item
                    # Check if image is valid
                    if isinstance(img, torch.Tensor):
                        if torch.isnan(img).any() or torch.isinf(img).any():
                            corrupted.append(idx)
            except Exception as e:
                corrupted.append(idx)
                print(f"  Error at index {idx}: {e}")
        
        if corrupted:
            print(f"⚠️  Found {len(corrupted)} corrupted samples")
        else:
            print("✓ No corrupted samples found")
        
        return corrupted


class ImbalancedDataHandler:
    """Handle imbalanced datasets."""
    
    @staticmethod
    def create_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
        """Create sampler for imbalanced dataset."""
        # Get all labels
        labels = []
        for item in dataset:
            if isinstance(item, tuple):
                _, label = item
            else:
                label = item
            labels.append(label)
        
        # Calculate class weights
        class_counts = Counter(labels)
        total = len(labels)
        
        # Weight inversely proportional to frequency
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        
        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print("✓ Created weighted sampler for balanced sampling")
        print(f"  Class weights: {class_weights}")
        
        return sampler
    
    @staticmethod
    def create_balanced_loader(
        dataset: Dataset,
        batch_size: int = 32,
        **kwargs
    ) -> DataLoader:
        """Create DataLoader with balanced sampling."""
        sampler = ImbalancedDataHandler.create_weighted_sampler(dataset)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # Note: can't use shuffle with sampler
            **kwargs
        )


class AugmentationPreview:
    """Visualize augmentations before training."""
    
    @staticmethod
    def preview(
        dataset: Dataset,
        transform: Callable,
        num_samples: int = 9,
        num_augmentations: int = 4
    ) -> List:
        """
        Generate preview of augmented images.
        
        Returns:
            List of augmented images for visualization
        """
        print(f"🎨 Generating augmentation preview...")
        
        # Select random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        previews = []
        
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, tuple):
                img, label = item
            else:
                img = item
                label = None
            
            # Generate multiple augmentations
            augmented = []
            for _ in range(num_augmentations):
                aug_img = transform(img)
                augmented.append(aug_img)
            
            previews.append({
                'original': img,
                'augmented': augmented,
                'label': label
            })
        
        print(f"✓ Generated {num_samples} samples × {num_augmentations} augmentations")
        
        return previews


class DataSplitter:
    """Split datasets for training/validation/test."""
    
    @staticmethod
    def split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/val/test."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(random_seed)
        
        # Calculate sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        print(f"✓ Dataset split:")
        print(f"  Train: {len(train_dataset)} ({100*train_ratio:.1f}%)")
        print(f"  Val:   {len(val_dataset)} ({100*val_ratio:.1f}%)")
        print(f"  Test:  {len(test_dataset)} ({100*test_ratio:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def stratified_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """Stratified split preserving class distribution."""
        from sklearn.model_selection import train_test_split
        
        # Get labels
        labels = []
        for item in dataset:
            if isinstance(item, tuple):
                _, label = item
            else:
                label = 0
            labels.append(label)
        
        indices = np.arange(len(dataset))
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=random_seed
        )
        
        # Second split: val vs test
        temp_labels = [labels[i] for i in temp_idx]
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=random_seed
        )
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        print(f"✓ Stratified split complete")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val:   {len(val_dataset)}")
        print(f"  Test:  {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset


# Example usage
if __name__ == "__main__":
    from torchvision import datasets
    
    # Load example dataset
    transform = AugmentationPipeline.image_classification_train()
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create smart data loader
    loader = SmartDataLoader.create(dataset, batch_size=64)
    
    # Analyze dataset
    stats = DatasetAnalyzer.analyze(dataset)
    
    # Split dataset
    train_ds, val_ds, test_ds = DataSplitter.stratified_split(dataset)
    
    # Handle imbalanced data
    balanced_loader = ImbalancedDataHandler.create_balanced_loader(dataset, batch_size=64)
