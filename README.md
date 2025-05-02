# Minimal Mechanistic Interpretability Project

A project to build a small transformer model with built-in interpretability features.

## Project Structure

- `model.py` - Core transformer implementation with probes
- `visualization.py` - Basic visualization tools for attention and activations
- `train.py` - Simple training loop with interpretability metrics
- `analysis.ipynb` - Interactive notebook for analyzing model internals
- `data/` - Directory for training data
- `requirements.txt` - Project dependencies

## Key Features

- Small-scale transformer with inspection points at every layer
- Built-in activation probes for tracking information flow
- Basic visualization tools for attention patterns
- Simple analysis framework for feature attribution

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py`
3. Explore the model: `jupyter notebook analysis.ipynb` 