"""
PyTorch Model Builder - Visual Model Construction & Templates
Features: Pre-built architectures, model surgery, auto-summary, NAS basics
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


class ModelTemplates:
    """Pre-built neural network architectures."""
    
    @staticmethod
    def simple_cnn(num_classes=10, input_channels=3):
        """Simple CNN for image classification."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    @staticmethod
    def resnet_block(in_channels, out_channels, stride=1):
        """ResNet residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    @staticmethod
    def resnet18(num_classes=1000, input_channels=3):
        """ResNet-18 architecture."""
        class ResNet18(nn.Module):
            def __init__(self, num_classes, input_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                
                # ResNet layers
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)
            
            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = []
                layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                
                for _ in range(1, blocks):
                    layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.ReLU())
                
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return ResNet18(num_classes, input_channels)
    
    @staticmethod
    def transformer_encoder(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        """Transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    @staticmethod
    def unet(in_channels=3, out_channels=1):
        """U-Net for segmentation."""
        class UNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                # Encoder
                self.enc1 = self._conv_block(in_channels, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)
                
                # Bottleneck
                self.bottleneck = self._conv_block(512, 1024)
                
                # Decoder
                self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                self.dec4 = self._conv_block(1024, 512)
                
                self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = self._conv_block(512, 256)
                
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = self._conv_block(256, 128)
                
                self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = self._conv_block(128, 64)
                
                self.out = nn.Conv2d(64, out_channels, 1)
                
                self.pool = nn.MaxPool2d(2)
            
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                
                # Bottleneck
                b = self.bottleneck(self.pool(e4))
                
                # Decoder with skip connections
                d4 = self.upconv4(b)
                d4 = torch.cat([d4, e4], dim=1)
                d4 = self.dec4(d4)
                
                d3 = self.upconv3(d4)
                d3 = torch.cat([d3, e3], dim=1)
                d3 = self.dec3(d3)
                
                d2 = self.upconv2(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d1 = self.upconv1(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                
                return self.out(d1)
        
        return UNet(in_channels, out_channels)
    
    @staticmethod
    def lstm_classifier(input_size, hidden_size=128, num_layers=2, num_classes=10):
        """LSTM for sequence classification."""
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len, input_size)
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use last hidden state
                out = self.fc(h_n[-1])
                return out
        
        return LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    
    @staticmethod
    def vae(input_dim=784, latent_dim=20):
        """Variational Autoencoder."""
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU()
                )
                
                self.fc_mu = nn.Linear(256, latent_dim)
                self.fc_logvar = nn.Linear(256, latent_dim)
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim),
                    nn.Sigmoid()
                )
            
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, logvar = self.encode(x.view(-1, x.size(-1)))
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
        
        return VAE(input_dim, latent_dim)
    
    @staticmethod
    def gan(latent_dim=100, img_channels=1, img_size=28):
        """Generative Adversarial Network."""
        class Generator(nn.Module):
            def __init__(self, latent_dim, img_channels, img_size):
                super().__init__()
                self.img_shape = (img_channels, img_size, img_size)
                
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, int(np.prod(self.img_shape))),
                    nn.Tanh()
                )
            
            def forward(self, z):
                img = self.model(z)
                img = img.view(img.size(0), *self.img_shape)
                return img
        
        class Discriminator(nn.Module):
            def __init__(self, img_channels, img_size):
                super().__init__()
                
                self.model = nn.Sequential(
                    nn.Linear(img_channels * img_size * img_size, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, img):
                img_flat = img.view(img.size(0), -1)
                validity = self.model(img_flat)
                return validity
        
        return {
            'generator': Generator(latent_dim, img_channels, img_size),
            'discriminator': Discriminator(img_channels, img_size)
        }


class ModelSurgery:
    """Tools for modifying existing models."""
    
    @staticmethod
    def freeze_layers(model: nn.Module, until: Optional[str] = None):
        """Freeze layers up to a certain layer name."""
        freeze_all = until is None
        
        for name, param in model.named_parameters():
            if freeze_all or until in name:
                param.requires_grad = False
                if until in name:
                    break
        
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        print(f"✓ Frozen {frozen_count}/{total_count} parameter groups")
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, from_layer: Optional[str] = None):
        """Unfreeze layers from a certain layer name onwards."""
        found = from_layer is None
        
        for name, param in model.named_parameters():
            if found:
                param.requires_grad = True
            elif from_layer in name:
                found = True
                param.requires_grad = True
        
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"✓ {trainable} parameter groups trainable")
    
    @staticmethod
    def replace_head(model: nn.Module, num_classes: int, layer_name: str = 'fc'):
        """Replace the classification head."""
        # Find the last linear layer
        for name, module in model.named_children():
            if name == layer_name and isinstance(module, nn.Linear):
                in_features = module.in_features
                setattr(model, name, nn.Linear(in_features, num_classes))
                print(f"✓ Replaced {layer_name}: {in_features} → {num_classes}")
                return
        
        # Try to find any linear layer
        modules = list(model.modules())
        for i in range(len(modules) - 1, -1, -1):
            if isinstance(modules[i], nn.Linear):
                in_features = modules[i].in_features
                modules[i] = nn.Linear(in_features, num_classes)
                print(f"✓ Replaced last linear layer: {in_features} → {num_classes}")
                return
    
    @staticmethod
    def add_dropout(model: nn.Module, p: float = 0.5):
        """Add dropout layers after ReLU activations."""
        new_model = []
        for module in model.children():
            new_model.append(module)
            if isinstance(module, nn.ReLU):
                new_model.append(nn.Dropout(p))
        
        return nn.Sequential(*new_model)
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


class ModelSummary:
    """Generate comprehensive model summary."""
    
    @staticmethod
    def summary(model: nn.Module, input_size: Tuple[int, ...], device: str = 'cpu'):
        """
        Generate detailed model summary.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (e.g., (3, 224, 224))
            device: Device to run on
        """
        model = model.to(device)
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = f"{class_name}-{module_idx+1}"
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["output_shape"] = list(output.size())
                
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                    summary[m_key]["trainable"] = module.weight.requires_grad
                
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
                
                summary[m_key]["nb_params"] = params
            
            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))
        
        # Create summary dict
        summary = OrderedDict()
        hooks = []
        
        # Register hooks
        model.apply(register_hook)
        
        # Make a forward pass
        with torch.no_grad():
            x = torch.zeros(1, *input_size).to(device)
            model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Print summary
        print("=" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
        print("=" * 80)
        
        total_params = 0
        trainable_params = 0
        
        for layer in summary:
            print(f"{layer:<30} {str(summary[layer]['output_shape']):<25} "
                  f"{summary[layer]['nb_params']:,}")
            total_params += summary[layer]["nb_params"]
            if "trainable" in summary[layer] and summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 80)
        
        # Calculate model size
        param_size = total_params * 4 / (1024 ** 2)  # Assume float32
        print(f"Model size: {param_size:.2f} MB")
        print("=" * 80)
        
        return summary


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create models from templates
    print("Creating ResNet-18...")
    resnet = ModelTemplates.resnet18(num_classes=10)
    
    print("\nCreating U-Net...")
    unet = ModelTemplates.unet()
    
    print("\nCreating Simple CNN...")
    cnn = ModelTemplates.simple_cnn(num_classes=10)
    
    # Model surgery
    print("\n" + "="*50)
    print("Model Surgery Examples")
    print("="*50)
    
    ModelSurgery.freeze_layers(resnet, until='layer3')
    ModelSurgery.replace_head(resnet, num_classes=100)
    
    params = ModelSurgery.count_parameters(resnet)
    print(f"\nParameters: {params}")
    
    # Model summary
    print("\n" + "="*50)
    print("Model Summary")
    print("="*50)
    ModelSummary.summary(cnn, input_size=(3, 224, 224))
