# SimCLR: Simple Framework for Contrastive Learning of Visual Representations

This repository contains a PyTorch implementation of SimCLR, a self-supervised learning framework for visual representation learning using contrastive loss.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Details](#training-details)
- [Checkpoints & Logging](#checkpoints--logging)
- [Notes](#notes)

## Features
- Modular SimCLR model with configurable ResNet encoder and MLP projection head
- Custom LARS optimizer implementation
- Data augmentation pipeline for CIFAR-10
- PyTorch Lightning DataModule for easy data handling
- Configurable via YAML file

## Project Structure
```
├── main.py                # CLI entry point
├── train.py               # Training and validation logic
├── preprocessing.py       # Data loading and augmentation
├── model/
│   ├── model.py           # SimCLR model definition
│   ├── encoder.py         # ResNet encoder
├── utils/
│   └── lars.py            # LARS optimizer
├── config/
│   └── config.yml         # Main configuration file
├── data/                  # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/           # Model checkpoints
├── README.md
├── .gitignore
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/tuvv3ct0r/simclr_pytorch.git
   cd simclr_pytorch
   ```
2. **Install dependencies:**
   Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   Install required packages:
   ```bash
   pip install torch torchvision pytorch-lightning
   ```

## Configuration
All settings are managed via `config/config.yml`. Key sections:
- **model:** Encoder and projection head architecture
- **training:** Optimizer, learning rate, batch size, epochs, etc.
- **augmentation:** Data augmentation parameters
- **dataset:** Dataset name and path
- **checkpoint:** Checkpoint directory and frequency
- **logging:** Logging directory and interval

## Usage
Run the main CLI script:
```bash
python main.py --config config/config.yml --train   # Train the model
python main.py --config config/config.yml --eval    # Evaluate on validation set
python main.py --config config/config.yml --test    # (WIP) Test mode
```

- The CIFAR-10 dataset will be downloaded automatically to `data/`.
- Checkpoints will be saved in `checkpoints/simclr/`.

## Training Details
- **Loss:** NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
- **Optimizers:** Adam (default), LARS (for large-batch training)
- **Augmentations:** Random crop, color jitter, grayscale, Gaussian blur, normalization
- **Encoder:** Custom ResNet (configurable channels)

## Checkpoints & Logging
- Checkpoints are saved every `save_freq` epochs and the best model is tracked by validation loss.
- Logging is printed to stdout; TensorBoard support is planned (see config).

## Notes
- Only CIFAR-10 is supported out-of-the-box, but you can adapt the DataModule for other datasets.
- The `.gitignore` excludes `data/` and `__pycache__/` by default.
- For custom experiments, modify `config/config.yml`.

## Requirements
- Python 3.8+
- torch
- torchvision
- pytorch-lightning

---

For questions or contributions, please open an issue or pull request.
