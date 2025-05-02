import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbedMLP(nn.Module):
    """MLP with probes for activation tracking"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation storage
        self.activations = {}
        
    def forward(self, x):
        # Store input
        self.activations['input'] = x.detach()
        
        # First linear layer
        x1 = self.linear1(x)
        self.activations['pre_activation'] = x1.detach()
        
        # Activation
        x2 = F.gelu(x1)
        self.activations['post_activation'] = x2.detach()
        
        # Second linear layer with dropout
        x3 = self.dropout(self.linear2(x2))
        self.activations['output'] = x3.detach()
        
        return x3
    
    def get_activation(self, name):
        return self.activations.get(name, None)


class ProbedAttention(nn.Module):
    """Multi-head attention with probes for interpretability"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation storage
        self.activations = {}
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Store inputs
        self.activations['q_input'] = q.detach()
        self.activations['k_input'] = k.detach()
        self.activations['v_input'] = v.detach()
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        self.activations['q_proj'] = q.detach()
        self.activations['k_proj'] = k.detach()
        self.activations['v_proj'] = v.detach()
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        self.activations['attention_scores'] = scores.detach()
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        self.activations['attention_weights'] = attn_weights.detach()
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        self.activations['attention_output'] = output.detach()
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        self.activations['final_output'] = output.detach()
        
        return output, attn_weights
    
    def get_activation(self, name):
        return self.activations.get(name, None)


class ProbedTransformerLayer(nn.Module):
    """Transformer layer with probes throughout for interpretability"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ProbedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ProbedMLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Layer activation storage
        self.activations = {}
        
    def forward(self, x, mask=None):
        # Store input
        self.activations['layer_input'] = x.detach()
        
        # Self-attention block
        attn_output, attn_weights = self.attention(x, x, x, mask)
        self.activations['attention_weights'] = attn_weights.detach()
        
        # Residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_output))
        self.activations['post_attention'] = x.detach()
        
        # Feed-forward block
        ffn_output = self.ffn(x)
        
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout(ffn_output))
        self.activations['layer_output'] = x.detach()
        
        return x
    
    def get_attention_weights(self):
        return self.activations.get('attention_weights', None)
    
    def get_activation(self, name):
        return self.activations.get(name, None)


class InterpretableTransformer(nn.Module):
    """Transformer model designed for interpretability with probes at each layer"""
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, d_ff=1024, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ProbedTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Layer-wise activations
        self.layer_outputs = {}
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # Xavier initialization for linear layers
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tokens [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
        """
        batch_size, seq_len = x.size()
        
        # Apply token embeddings
        x = self.token_embedding(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Store embedding output
        self.layer_outputs['embedding_output'] = x.detach()
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            self.layer_outputs[f'layer_{i}_output'] = x.detach()
        
        # Apply output layer
        logits = self.output_layer(x)
        
        return logits
    
    def get_layer_output(self, layer_idx):
        """Get the output of a specific layer"""
        return self.layer_outputs.get(f'layer_{layer_idx}_output', None)
    
    def get_attention_weights(self, layer_idx):
        """Get attention weights from a specific layer"""
        return self.layers[layer_idx].get_attention_weights()
    
    def get_all_attention_weights(self):
        """Get attention weights from all layers"""
        return [layer.get_attention_weights() for layer in self.layers] 