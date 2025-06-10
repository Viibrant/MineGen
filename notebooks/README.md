# Research Notebooks Archive

This directory previously contained the original research notebooks used during development. **All functionality has been successfully integrated into the main codebase** and the notebooks have been cleaned up.

## ✅ What was integrated:

### From `patch.ipynb`:
- **HierarchicalTransformerModel**: Complete transformer architecture → `src/minegen/models/hierarchical_transformer.py`
- **ResidualBlock improvements**: Enhanced 3D residual blocks → `src/minegen/layers/residual.py`
- **Training utilities**: Lightning-based training loops → `src/minegen/cli/main.py`

### From `vae.ipynb`:
- **VAE model**: Complete variational autoencoder → `src/minegen/models/vae.py`
- **Training callbacks**: Model checkpointing and early stopping → `src/minegen/cli/main.py`
- **Evaluation metrics**: Confusion matrix and accuracy tracking → `tests/test_models.py`

### From `exploratory.ipynb`:
- **Data analysis**: Size distribution analysis → `src/minegen/data/utils.py`
- **Category encoding**: One-hot encoding for schematics → `src/minegen/data/dataset.py`
- **Visualization utilities**: Distribution plotting → `src/minegen/data/utils.py`

### From `train_embeddings.ipynb`:
- **AutoEncoder variants**: Multiple architectures → `src/minegen/models/autoencoder.py`
- **Embedding strategies**: Block ID embedding → `src/minegen/models/vae.py`
- **Loss functions**: Combined reconstruction and classification → All model files

### From `temp.py`:
- **Experimental layers**: Custom attention and transformer → `src/minegen/layers/`
- **Patch processing**: Advanced tensor rearrangement → `src/minegen/layers/blocks.py`
- **Experimental model**: Complete architecture → `src/minegen/models/experimental.py`

## 🚀 New Features Added:

### Real-time Dashboard
- **Training Monitor**: Live loss curves, metrics, progress tracking
- **Generation Viewer**: Interactive sample generation and gallery
- **Data Explorer**: Dataset statistics and quality analysis
- **System Status**: Resource monitoring and model management

### Enhanced CLI
- **Multiple Models**: VAE, Transformer, AutoEncoder, Voxel, Experimental
- **Advanced Generation**: Category-specific, batch generation, format options
- **Data Pipeline**: Automated downloading, processing, evaluation
- **Configuration**: YAML-based config system with templates

### Production Features
- **Lightning Integration**: Scalable training with callbacks
- **Experiment Tracking**: Weights & Biases integration
- **Model Management**: Checkpointing, resuming, evaluation
- **Data Quality**: Validation, filtering, preprocessing

## 🛠️ Current Usage:

Instead of running notebooks, use the modern CLI and dashboard:

```bash
# Download and prepare data
minegen download-data --criteria most-downloaded --num-pages 50

# Train models with real-time monitoring
minegen train-model --model-type vae --batch-size 32 --max-epochs 100 --use-wandb

# Launch interactive dashboard
minegen dashboard --host localhost --port 8050

# Generate schematics
minegen generate --checkpoint model.ckpt --num-samples 10 --save-schematics

# Evaluate model performance
minegen evaluate --checkpoint model.ckpt --num-samples 100
```

## 📊 Dashboard Features:

- **Real-time Training**: Watch loss curves update live
- **Interactive Generation**: Generate samples with custom parameters
- **Data Visualization**: Explore dataset distributions and quality
- **Model Comparison**: Compare different architectures and checkpoints
- **Export Tools**: Download generated schematics and results

## 🧪 Testing:

Comprehensive test suite with pytest:

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_models.py -v
pytest tests/test_dashboard.py -v
pytest tests/test_integration.py -v

# Coverage reporting
pytest --cov=src/minegen --cov-report=html
```

## 🎯 Benefits of Integration:

1. **Maintainability**: Proper module structure, type hints, error handling
2. **Scalability**: Lightning framework, distributed training support
3. **Usability**: CLI tools, interactive dashboard, configuration management
4. **Reliability**: Comprehensive testing, CI/CD ready
5. **Extensibility**: Plugin architecture, model registry, custom layers

The codebase now follows modern Python best practices and provides a complete toolkit for Minecraft schematic generation with deep learning.