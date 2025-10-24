# Neural Networks for Self-Mixing Interferometry

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)

## Overview

Self-mixing interferometry (SMI) is a powerful optical sensing technique that leverages the interference between a laser beam and its reflection from a target within the laser cavity itself. The interference inside the laser gain medium results in nonlinear self-mixing which can be read out as a drop in luminescence of the gain medium or a transient voltage drop across the laser diode. The nonlinear self-mixing results in signals that are difficult to interpret and analyze. This repository contains our PyTorch framework for training and deploying neural networks applied to this signal processing task, addressing the fundamental challenges that have limited industrial applications for decades.

## The Challenge

Traditional self-mixing interferometry faces significant practical limitations:
- **Signal interpretation complexity**: SMI signals depend sensitively on alignment conditions, target reflectivity, and feedback parameters
- **Speckle effects**: Diffusive targets cause signal quality variations that can completely invalidate measurements
- **Limited signal availability**: Systems often fail with non-cooperative targets or changing environmental conditions

This project demonstrates how **convolutional neural networks** can revolutionize self-mixing interferometry by:

### 🧠 **Intelligent Signal Processing**
- **Universal signal interpretation**: Neural networks trained on diverse signal conditions can extract displacement information from arbitrary SMI waveforms
- **Robust to noise and distortion**: Networks maintain performance even with severely degraded signals
- **Multi-regime operation**: Single model works across different feedback regimes without parameter tuning

### 🔄 **Multi-Channel High-Availability Sensing**
- **Redundant measurement channels**: Multiple independent SMI sensors provide fault tolerance
- **Graceful degradation**: System maintains accuracy even when individual channels fail completely
- **Enhanced reliability**: Achieves high-availability displacement sensing robust to speckle and alignment variations

### ⚡ **Real-Time Performance**
- **Embeddable networks**: Lightweight architectures suitable for embedded systems
- **Fast inference**: Process thousands of measurements in milliseconds
- **No parameter estimation**: Direct displacement inference without complex physics models of the laser self-mixing process

## Quick Start for Contributors

1. Clone the repository:
   ```bash
   git clone https://github.com/username/your-project.git
   cd your-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pip install pre-commit nbstripout
   pre-commit install
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

6. Make your changes and run tasks:
   ```bash
   # Run tests
   task test

   # Lint your code
   task lint

   # Format your code
   task format

   # Check spelling
   task spell

   # Run pre-commit hooks manually
   task precommit

   # Run format, lint and test in sequence
   task all
   ```

7. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

8. Open a Pull Request on GitHub

## Running the Main Script

The `main.py` script provides three different execution modes for working with the self-mixing interferometry system:

### Mode 1: Training a New Model

Train a neural network model using a YAML configuration file:

```bash
python main.py --config path/to/config.yaml
```

**Arguments:**
- `--config`: Path to YAML configuration file (default: `./analysis/configs/tcn-config.yaml`)
- `--verbosity`: Set logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO)

**Example:**
```bash
python main.py --config ./analysis/configs/tcn-config.yaml --verbosity DEBUG
```

### Mode 2: Evaluating from Checkpoint

Evaluate a trained model from a checkpoint file on a specific dataset:

```bash
python main.py --checkpoint path/to/checkpoint.ckpt --dataset path/to/dataset.h5
```

**Arguments:**
- `--checkpoint`: Path to the model checkpoint file (`.ckpt`)
- `--dataset`: Path to the dataset file (`.h5`) for evaluation

**Example:**
```bash
python main.py --checkpoint ./checkpoints/model-epoch=50.ckpt --dataset ./analysis/data/test-data.h5
```

### Mode 3: Acquiring Real Data from Red Pitaya

Acquire real experimental data from Red Pitaya hardware for training or testing:

```bash
python main.py --acquire_dataset --num_samples 5000 --dataset_name my-data.h5
```

**Arguments:**
- `--acquire_dataset`: Flag to enable dataset acquisition mode
- `--num_samples`: Number of samples to acquire (required with `--acquire_dataset`)
- `--dataset_name`: Filename for the acquired dataset (required with `--acquire_dataset`)

**Example:**
```bash
python main.py --acquire_dataset --num_samples 10000 --dataset_name experimental-data.h5 --verbosity INFO
```

**Note:** The Red Pitaya connection uses default settings configured in the `RedPitayaManager`. The acquired data will be saved to `./analysis/data/` directory.
