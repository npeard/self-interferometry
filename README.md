# Neural Networks for Self-Mixing Interferometry

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)

## Overview

Self-mixing interferometry (SMI) offers a cost-effective and optically simple alternative to Michelson interferometry. However, adapting SMI for standard vibrometry remains challenging because analytical signal processing is brittle and sensitive to laser feedback conditions. Furthermore, achieving true portability for field deployment is limited by severe phase noise introduced in optical fibers.

This repository implements an optical and computational framework that overcomes these limitations, using a multispectral SMI array constructed from standard laboratory components. By leveraging convolutional neural networks, our signal processing pipeline reconstructs mechanical displacement with an RMSE below 200 nm directly through multimode fiber. This synergy of frugal optics and deep learning yields an accessible and precise vibrometry tool uniquely suited for hidden or physically constrained environments.

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

5. Install the [Marimo](https://marimo.io/) VSCode extension for interactive notebook support. Notebooks in `notebooks/` are Marimo notebooks (`.py` files) and can be opened with `marimo edit notebooks/<name>.py` or directly in VSCode with the extension.

6. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

7. Make your changes and run tasks:
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

8. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

9. Open a Pull Request on GitHub

## Running the Main Script

The `main.py` script provides two execution modes:

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

### Mode 2: Acquiring Real Data from Red Pitaya

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

### Evaluating a Trained Model

To visualize predictions, residuals, and input gradient attributions from a trained checkpoint, use the interactive Marimo notebook:

```bash
marimo edit notebooks/predictions.py
```

Set the checkpoint path and dataset path (or `"synthetic"`) in the UI controls at the top of the notebook.
