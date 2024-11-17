# 💎 MineGen: Minecraft Schematic Generator 🚧

## About <a name = "about"></a>

MineGen is a project that uses deep learning and transformers to generate schematic files for Minecraft. The goal of this project is to create a tool that can generate unique and interesting building structures for players to use in their world.

## Getting Started <a name = "getting_started"></a>

### Prerequisites

- Python 3.9+
- PyTorch 1.x
- yacs
- ignite

### Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Viibrant/MineGen
pip install -r requirements.txt
```

### Usage

*Coming soon...*

### Project structure

```python

├── config # default configuration
│   ├── defaults.py
├── data
│   ├── build.py # script for building the dataset
│   ├── collate_batch.py # function for collating data into a batch
│   ├── datasets
│   │   ├── generate.py # script for generating data
│   │   └── mcs # folder to scrape schematic files and metadaa
│   │       └── util # helper scripts for downloading and for metadata
│   └── transforms # data augmentation for datasets 
│       ├── build.py 
│       └── transforms.py 
├── engine # scripts for training and inference
├── layers # folder to store custom layers
├── modelling # definitions of all network architectures
│   └── schem_model.py # module for the schematic generation model
├── requirements.txt # list of required libraries
├── solver # optimiser for model
│   ├── build.py 
└── tools
    ├── test_net.py # script for testing the model
    └── train_net.py # script for training the model
```

## Author

- [Viibrant](https://github.com/Viibrant) - Final year student, dissertation project
