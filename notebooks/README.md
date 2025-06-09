# Research Notebooks Archive

This directory previously contained the original research notebooks used during development. The notebooks have been cleaned up and their functionality integrated into the main codebase.

## What was integrated:

### From `patch.ipynb`:
- **HierarchicalTransformerModel**: The main transformer architecture with hierarchical processing
- **ResidualBlock improvements**: Enhanced 3D residual blocks
- **Training utilities**: Lightning-based training loops

### From `vae.ipynb`:
- **VAE model**: Complete variational autoencoder implementation
- **Training callbacks**: Model checkpointing and early stopping
- **Evaluation metrics**: Confusion matrix and accuracy tracking

### From `exploratory.ipynb`:
- **Data analysis**: Size distribution analysis and filtering
- **Category encoding**: One-hot encoding for schematic categories
- **Visualization utilities**: Distribution plotting functions

### From `train_embeddings.ipynb`:
- **Autoencoder variants**: Multiple autoencoder architectures
- **Embedding strategies**: Block ID embedding approaches
- **Loss functions**: Combined reconstruction and classification losses

### From `temp.py`:
- **Experimental layers**: Custom attention and transformer implementations
- **Patch processing**: Advanced tensor rearrangement utilities

## Current structure:

All functionality has been properly organized into:
- `src/minegen/models/` - Model architectures
- `src/minegen/layers/` - Custom neural network layers  
- `src/minegen/data/` - Data loading and processing
- `src/minegen/cli/` - Command line interface
- `tests/` - Comprehensive test suite

## Usage:

Instead of running notebooks, use the CLI:

```bash
# Train a model
minegen train-model --model-type vae --batch-size 32 --max-epochs 100

# Generate schematics  
minegen generate --checkpoint model.ckpt --num-samples 10

# Download data
minegen download-data --criteria most-downloaded --num-pages 50
```

The codebase now follows modern Python best practices with proper type hints, error handling, and modular design.