import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_attention_heatmap(attention_weights, head_idx=0, layer_idx=0, tokens=None):
    """
    Plot attention weights as a heatmap
    
    Args:
        attention_weights: Tensor of shape [batch_size, n_heads, seq_len, seq_len]
        head_idx: Index of attention head to visualize
        layer_idx: Only used for the title
        tokens: Optional list of tokens for axis labels
    """
    # Extract single head attention pattern from first example in batch
    attn = attention_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attn, cmap="viridis", vmin=0, vmax=1)
    
    # Add token labels if provided
    if tokens is not None:
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
    
    plt.title(f"Layer {layer_idx}, Head {head_idx} Attention")
    plt.tight_layout()
    
    return plt.gcf()


def plot_all_heads(attention_weights, layer_idx=0, tokens=None):
    """
    Plot all attention heads in a grid
    
    Args:
        attention_weights: Tensor of shape [batch_size, n_heads, seq_len, seq_len]
        layer_idx: Layer index for the title
        tokens: Optional list of tokens for axis labels
    """
    n_heads = attention_weights.size(1)
    
    # Calculate grid dimensions
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_heads):
        attn = attention_weights[0, i].cpu().numpy()
        ax = axes[i]
        sns.heatmap(attn, cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title(f"Head {i}")
        
        # Simplify x and y tick labels for readability
        if tokens is not None and len(tokens) <= 10:
            ax.set_xticklabels(tokens)
            ax.set_yticklabels(tokens)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Layer {layer_idx} Attention Heads", y=1.02)
    plt.tight_layout()
    
    return fig


def plot_layer_activations(layer_outputs, layer_idx=0):
    """
    Plot activation patterns for a layer
    
    Args:
        layer_outputs: Tensor of shape [batch_size, seq_len, d_model]
        layer_idx: Layer index for the title
    """
    # Take the first example in the batch
    activations = layer_outputs[0].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot heatmap of activations
    sns.heatmap(activations, cmap="coolwarm", center=0)
    
    plt.title(f"Layer {layer_idx} Activations")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Sequence Position")
    plt.tight_layout()
    
    return plt.gcf()


def plot_activation_histograms(model, layer_indices=None):
    """
    Plot histograms of activations for multiple layers
    
    Args:
        model: The transformer model
        layer_indices: List of layer indices to plot (default: all layers)
    """
    if layer_indices is None:
        layer_indices = range(model.n_layers)
    
    n_layers = len(layer_indices)
    
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
    if n_layers == 1:
        axes = [axes]
    
    for i, layer_idx in enumerate(layer_indices):
        # Get layer output
        layer_output = model.get_layer_output(layer_idx)
        if layer_output is None:
            continue
        
        # Flatten activations for histogram
        activations = layer_output.cpu().numpy().flatten()
        
        # Plot histogram
        axes[i].hist(activations, bins=50, alpha=0.7)
        axes[i].set_title(f"Layer {layer_idx} Activation Distribution")
        axes[i].set_xlabel("Activation Value")
        axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    return fig


def create_interactive_attention_view(attention_weights, tokens=None):
    """
    Create an interactive Plotly visualization of attention patterns
    
    Args:
        attention_weights: List of tensors, each of shape [batch_size, n_heads, seq_len, seq_len]
                          representing attention weights for each layer
        tokens: Optional list of tokens for axis labels
    """
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].size(1)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[f"Layer {l}, Head {h}" for l in range(n_layers) for h in range(n_heads)]
    )
    
    # Add heatmaps for each layer and head
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=attn,
                colorscale="Viridis",
                showscale=False,
                x=tokens if tokens else None,
                y=tokens if tokens else None
            )
            
            fig.add_trace(heatmap, row=layer_idx+1, col=head_idx+1)
    
    # Update layout
    fig.update_layout(
        title="Attention Patterns Across Layers and Heads",
        height=300 * n_layers,
        width=250 * n_heads,
        showlegend=False
    )
    
    return fig


def visualize_attention_flow(model, input_ids, tokenizer=None):
    """
    Visualize how attention flows from token to token through layers
    
    Args:
        model: The transformer model
        input_ids: Input tensor of shape [1, seq_len]
        tokenizer: Optional tokenizer for decoding tokens
    """
    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get attention weights from all layers
    all_attentions = model.get_all_attention_weights()
    
    # Get tokens if tokenizer is provided
    tokens = None
    if tokenizer is not None:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Create interactive visualization
    fig = create_interactive_attention_view(all_attentions, tokens)
    
    return fig


def compare_head_behaviors(model, dataset, n_samples=10):
    """
    Analyze and compare attention head behaviors across different inputs
    
    Args:
        model: The transformer model
        dataset: Dataset with input_ids
        n_samples: Number of samples to analyze
    """
    model.eval()
    head_patterns = []
    
    # Collect attention patterns for multiple inputs
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if i >= n_samples:
                break
                
            inputs = batch[0].unsqueeze(0)  # Add batch dimension
            _ = model(inputs)
            
            # Get attention weights from all layers
            all_attentions = model.get_all_attention_weights()
            head_patterns.append(all_attentions)
    
    # Calculate consistency metrics for each head
    n_layers = len(head_patterns[0])
    n_heads = head_patterns[0][0].size(1)
    
    consistency_scores = np.zeros((n_layers, n_heads))
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Extract patterns for this head across samples
            patterns = [p[layer_idx][0, head_idx].cpu() for p in head_patterns]
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    p1 = patterns[i].flatten()
                    p2 = patterns[j].flatten()
                    sim = torch.nn.functional.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                    similarities.append(sim.item())
            
            # Average similarity as consistency score
            consistency_scores[layer_idx, head_idx] = np.mean(similarities)
    
    # Plot head consistency heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(consistency_scores, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title("Attention Head Consistency Across Inputs")
    plt.tight_layout()
    
    return plt.gcf(), consistency_scores 