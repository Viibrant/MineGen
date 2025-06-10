# 💎 MineGen: Minecraft Schematic Generator 🚧

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About

MineGen is a comprehensive deep learning toolkit for generating Minecraft schematic files. It uses state-of-the-art transformer architectures and variational autoencoders to create unique and interesting building structures for players to use in their worlds.

## ✨ Features

- 🏗️ **Multiple Model Architectures**: VAE, Hierarchical Transformers, AutoEncoders, and Experimental models
- 📊 **Real-time Dashboard**: Monitor training progress and generate schematics in real-time
- 🔄 **Automated Data Pipeline**: Download and process schematics from minecraft-schematics.com
- ⚡ **Lightning Integration**: Built on PyTorch Lightning for scalable training
- 🎯 **Category-aware Generation**: Generate schematics for specific building types
- 📈 **Experiment Tracking**: Integrated with Weights & Biases
- 🛠️ **Tool-oriented Design**: CLI and dashboard for practical workflows

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Viibrant/MineGen
cd MineGen

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"

# For dashboard features
pip install -e ".[dashboard]"

# For positional encodings (optional)
pip install -e ".[positional]"
```

### Basic Usage

```bash
# Download training data
minegen download-data --criteria most-downloaded --num-pages 10

# Train a VAE model
minegen train-model --model-type vae --batch-size 32 --max-epochs 100

# Generate schematics
minegen generate --checkpoint ./outputs/checkpoints/best.ckpt --num-samples 10

# Launch real-time dashboard
minegen dashboard --host localhost --port 8050
```

## 🏗️ Architecture

### Models Available

1. **VAE (Variational Autoencoder)**
   - Latent space generation
   - Category-aware encoding
   - Smooth interpolation

2. **Hierarchical Transformer**
   - Multi-scale attention
   - Patch-based processing
   - Skip connections

3. **AutoEncoder**
   - Basic reconstruction
   - Classification head
   - Fast training

4. **Experimental Models**
   - Research architectures
   - Advanced features
   - Cutting-edge techniques

### Dashboard Features

- 🎯 **Training Monitor**: Real-time loss curves, metrics, and progress
- 🎨 **Generation Viewer**: Interactive sample generation and gallery
- 📊 **Data Explorer**: Dataset statistics and quality analysis
- ⚙️ **Model Management**: Checkpoint handling and configuration

## 📖 Usage Examples

### Training with Custom Configuration

```bash
# Create a config file
minegen config-template

# Edit config_template.yaml, then train
minegen train-model --config config_template.yaml --model-type transformer
```

### Advanced Generation

```bash
# Generate with specific category
minegen generate \
  --checkpoint model.ckpt \
  --num-samples 20 \
  --category houses \
  --save-schematics \
  --output-dir ./my_buildings
```

### Dashboard Workflow

```bash
# Start dashboard
minegen dashboard --debug

# Navigate to http://localhost:8050
# - Monitor training in real-time
# - Generate samples interactively
# - Explore your dataset
# - Download results
```

### Evaluation and Analysis

```bash
# Evaluate model performance
minegen evaluate \
  --checkpoint model.ckpt \
  --data-dir ./schematics \
  --num-samples 100
```

## 🔧 Configuration

MineGen uses YAML configuration files for reproducible experiments:

```yaml
model:
  latent_dim: 64
  embedding_size: 512
  num_heads: 8
  num_layers: 12

solver:
  optimizer_name: "Adam"
  base_lr: 0.001
  max_epochs: 100

datasets:
  shape: [16, 16, 16]
  batch_size: 32
  
dashboard:
  host: "localhost"
  port: 8050
  auto_refresh: 5
```

See `config_examples/` for complete configuration templates.

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/minegen --cov-report=html

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_dashboard.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Project Structure

```
src/minegen/
├── cli/           # Command-line interface
├── config/        # Configuration management
├── data/          # Data loading and processing
├── dashboard/     # Real-time dashboard
├── engine/        # Training and inference
├── layers/        # Custom neural network layers
├── models/        # Model architectures
└── solver/        # Optimizers and schedulers

tests/             # Comprehensive test suite
config_examples/   # Configuration templates
notebooks/         # Research notebooks (archived)
```

## 📊 Model Performance

| Model | Parameters | Training Time | Generation Quality |
|-------|------------|---------------|-------------------|
| VAE | 2.1M | ~2 hours | ⭐⭐⭐⭐ |
| Transformer | 8.5M | ~6 hours | ⭐⭐⭐⭐⭐ |
| AutoEncoder | 1.2M | ~1 hour | ⭐⭐⭐ |

*Benchmarks on RTX 3080, 16GB dataset*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Minecraft community for schematic data
- Research papers that inspired the architectures

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/Viibrant/MineGen/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Viibrant/MineGen/discussions)

---

**Made with ❤️ for the Minecraft community**