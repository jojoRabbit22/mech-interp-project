import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from model import InterpretableTransformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class InterpretabilityMetrics:
    """Class to calculate interpretability metrics during training"""
    
    @staticmethod
    def attention_entropy(attn_weights):
        """Calculate entropy of attention weights as a measure of focus"""
        # attn_weights shape: [batch_size, n_heads, seq_len, seq_len]
        # Entropy is calculated per head, then averaged
        eps = 1e-10  # To avoid log(0)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
        return entropy.mean().item()
    
    @staticmethod
    def activation_sparsity(activations):
        """Calculate percentage of near-zero activations"""
        # Consider values with abs < 0.01 as "inactive"
        inactive = (torch.abs(activations) < 0.01).float().mean().item()
        return inactive
    
    @staticmethod
    def head_specialization(model, val_loader, device):
        """Measure how specialized attention heads are (higher = more specialized)"""
        head_patterns = []
        
        # Collect attention patterns on validation data
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                # Forward pass
                _ = model(inputs)
                # Get attention weights from all layers
                all_attns = model.get_all_attention_weights()
                
                for layer_idx, attn in enumerate(all_attns):
                    if len(head_patterns) <= layer_idx:
                        head_patterns.append([])
                    head_patterns[layer_idx].append(attn.cpu())
        
        # Calculate head diversity score
        diversity_scores = []
        for layer_attns in head_patterns:
            layer_attns = torch.cat(layer_attns, dim=0)  # Combine batches
            n_heads = layer_attns.size(1)
            
            # Calculate average pattern for each head
            head_avg_patterns = layer_attns.mean(dim=0)  # Average over batch
            
            # Calculate pairwise cosine similarity between head patterns
            similarities = []
            for i in range(n_heads):
                for j in range(i+1, n_heads):
                    h1 = head_avg_patterns[i].flatten()
                    h2 = head_avg_patterns[j].flatten()
                    sim = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))
                    similarities.append(sim.item())
            
            # Average similarity (lower = more diverse/specialized)
            avg_sim = np.mean(similarities) if similarities else 0
            # Convert to diversity score (higher = more specialized)
            diversity = 1.0 - avg_sim
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores)


def train_model(train_loader, val_loader, model, criterion, optimizer, 
                num_epochs=10, device='cuda', log_interval=100):
    """Train the model with interpretability metrics"""
    
    model = model.to(device)
    metrics = InterpretabilityMetrics()
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    attn_entropies = []
    activation_sparsities = []
    head_specializations = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for i, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Display progress
                if (i + 1) % log_interval == 0:
                    # Calculate attention entropy
                    attn_weights = model.get_attention_weights(0)  # First layer
                    entropy = metrics.attention_entropy(attn_weights)
                    attn_entropies.append(entropy)
                    
                    # Calculate activation sparsity
                    layer_output = model.get_layer_output(0)  # First layer
                    sparsity = metrics.activation_sparsity(layer_output)
                    activation_sparsities.append(sparsity)
                    
                    # Update progress bar
                    t.set_postfix(loss=train_loss/(i+1), 
                                  entropy=entropy,
                                  sparsity=sparsity)
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate head specialization metric (expensive, so only once per epoch)
        specialization = metrics.head_specialization(model, val_loader, device)
        head_specializations.append(specialization)
        
        print(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Head Specialization: {specialization:.4f}")
    
    # Create a directory to save metric plots
    os.makedirs('metrics', exist_ok=True)
    
    # Plot training metrics
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(attn_entropies)
    plt.xlabel('Steps (x{})'.format(log_interval))
    plt.ylabel('Entropy')
    plt.title('Attention Entropy')
    
    plt.subplot(2, 2, 3)
    plt.plot(activation_sparsities)
    plt.xlabel('Steps (x{})'.format(log_interval))
    plt.ylabel('Sparsity')
    plt.title('Activation Sparsity')
    
    plt.subplot(2, 2, 4)
    plt.plot(head_specializations)
    plt.xlabel('Epoch')
    plt.ylabel('Specialization')
    plt.title('Head Specialization')
    
    plt.tight_layout()
    plt.savefig('metrics/training_metrics.png')
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'attn_entropies': attn_entropies,
        'activation_sparsities': activation_sparsities,
        'head_specializations': head_specializations
    }


def create_dummy_data(vocab_size=1000, seq_len=32, n_samples=1000):
    """Create dummy data for testing the model"""
    X = torch.randint(1, vocab_size, (n_samples, seq_len))
    # Shift inputs to create targets
    Y = torch.cat([X[:, 1:], torch.ones(n_samples, 1).long()], dim=1)
    
    # Split into train and validation
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader, vocab_size


def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    train_loader, val_loader, vocab_size = create_dummy_data()
    
    # Initialize model
    model = InterpretableTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=32,
        dropout=0.1
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model, metrics = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        device=device,
        log_interval=10
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/interpretable_transformer.pt')
    print("Model saved to models/interpretable_transformer.pt")


if __name__ == "__main__":
    main() 