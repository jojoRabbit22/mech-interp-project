import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Header
header_cell = nbf.v4.new_markdown_cell("""# Mechanistic Interpretability Analysis

This notebook demonstrates how to analyze the interpretable transformer model and extract insights about its internal mechanics.""")

# Imports
imports_cell = nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from model import InterpretableTransformer
from visualization import (
    plot_attention_heatmap, 
    plot_all_heads, 
    plot_layer_activations,
    plot_activation_histograms,
    create_interactive_attention_view
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)""")

# Load model section
load_model_header = nbf.v4.new_markdown_cell("""## 1. Load a Trained Model

First we'll load a trained model (or create a new one if no trained model exists).""")

load_model_cell = nbf.v4.new_code_cell("""# Model parameters
vocab_size = 1000
d_model = 128
n_heads = 4
n_layers = 2
d_ff = 512
max_seq_len = 32

# Initialize model
model = InterpretableTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
    dropout=0.1
)

# Try to load trained model if available
try:
    model.load_state_dict(torch.load('models/interpretable_transformer.pt'))
    print("Loaded trained model")
except FileNotFoundError:
    print("No trained model found, using initialized model")

# Set model to evaluation mode
model.eval()""")

# Sample input section
sample_input_header = nbf.v4.new_markdown_cell("""## 2. Create Sample Input

Let's create some sample input data to analyze model behavior.""")

sample_input_cell = nbf.v4.new_code_cell("""# Create random input sequence
seq_len = 20
input_ids = torch.randint(1, vocab_size, (1, seq_len))
print(f"Input shape: {input_ids.shape}")

# For demonstration, let's assign some token names
# In a real application, these would come from a tokenizer
sample_tokens = [f"token_{i}" for i in input_ids[0].tolist()]
print(f"Sample tokens: {sample_tokens}")""")

# Analysis section
analysis_header = nbf.v4.new_markdown_cell("""## 3. Analyze Model Outputs and Attention Patterns

Now we'll run the model on our sample input and examine its internal states.""")

analysis_cell = nbf.v4.new_code_cell("""# Run model forward pass
with torch.no_grad():
    outputs = model(input_ids)

print(f"Output logits shape: {outputs.shape}")""")

# Attention visualization section
attn_header = nbf.v4.new_markdown_cell("""### 3.1 Visualize Attention Patterns

One of the key insights into transformer behavior comes from analyzing attention patterns.""")

attn_cell1 = nbf.v4.new_code_cell("""# Get attention weights from the first layer
attention_weights = model.get_attention_weights(0)  # Layer 0

# Plot attention heatmap for the first head
plot_attention_heatmap(attention_weights, head_idx=0, layer_idx=0, tokens=sample_tokens)""")

attn_cell2 = nbf.v4.new_code_cell("""# Visualize all attention heads in the first layer
plot_all_heads(attention_weights, layer_idx=0, tokens=sample_tokens)""")

# Activation patterns section
activation_header = nbf.v4.new_markdown_cell("""### 3.2 Activation Patterns

Let's examine the activation patterns in the transformer layers.""")

activation_cell1 = nbf.v4.new_code_cell("""# Get output from the first layer
layer_0_output = model.get_layer_output(0)

# Plot activation patterns
plot_layer_activations(layer_0_output, layer_idx=0)""")

activation_cell2 = nbf.v4.new_code_cell("""# Plot activation histograms for all layers
plot_activation_histograms(model)""")

# Interactive visualization section
interactive_header = nbf.v4.new_markdown_cell("""## 4. Interactive Attention Visualization

Create an interactive visualization to explore attention flows.""")

interactive_cell = nbf.v4.new_code_cell("""# Get attention weights from all layers
all_attentions = model.get_all_attention_weights()

# Create interactive visualization
fig = create_interactive_attention_view(all_attentions, tokens=sample_tokens)
fig.show()""")

# Circuits section
circuits_header = nbf.v4.new_markdown_cell("""## 5. Examining Model Circuits

Let's probe deeper into the model to identify potential circuits and information flow.""")

circuits_cell = nbf.v4.new_code_cell('''def trace_information_flow(model, layer_idx, head_idx, position_idx, input_ids):
    """
    Trace how information flows from inputs through a specific attention head at a specific position
    """
    # Run model
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get attention weights for the specified layer and head
    attn_weights = model.get_attention_weights(layer_idx)
    
    # Extract weights for the specified position
    # This shows which input positions this position is attending to
    position_attention = attn_weights[0, head_idx, position_idx].cpu().numpy()
    
    # Find the top-3 positions that have the highest attention weights
    top_attended_positions = np.argsort(position_attention)[-3:]
    
    # Plot the attention
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(position_attention)), position_attention)
    plt.xlabel('Input Position')
    plt.ylabel('Attention Weight')
    plt.title(f'Information Flow: Layer {layer_idx}, Head {head_idx}, Position {position_idx}')
    plt.xticks(range(len(position_attention)), sample_tokens, rotation=45)
    plt.tight_layout()
    
    print(f"Top attended positions: {top_attended_positions}")
    print(f"Top attended tokens: {[sample_tokens[i] for i in top_attended_positions]}")
    
    return position_attention, top_attended_positions

# Trace information flow for a specific position in layer 0, head 0
position_to_analyze = 5  # Middle of the sequence
position_attention, top_positions = trace_information_flow(
    model, 
    layer_idx=0, 
    head_idx=0, 
    position_idx=position_to_analyze, 
    input_ids=input_ids
)''')

# Conclusion section
conclusion_header = nbf.v4.new_markdown_cell("""## 6. Conclusion

This notebook has demonstrated several techniques for analyzing and interpreting the behavior of a transformer model:

1. Visualizing attention patterns to understand what the model is focusing on
2. Examining activation patterns and distributions
3. Tracing information flow through specific heads and positions

These techniques can be extended to more complex models and real datasets to provide deeper insights into how transformer models work.""")

# Assemble the notebook
nb.cells = [
    header_cell,
    imports_cell,
    load_model_header,
    load_model_cell,
    sample_input_header,
    sample_input_cell,
    analysis_header,
    analysis_cell,
    attn_header,
    attn_cell1,
    attn_cell2,
    activation_header,
    activation_cell1,
    activation_cell2,
    interactive_header,
    interactive_cell,
    circuits_header,
    circuits_cell,
    conclusion_header
]

# Write the notebook to a file
with open('analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!") 