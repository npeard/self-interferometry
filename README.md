# Self-Mixing Interferometry for Rapid Vibration/Acoustic Spectrograms

## Goal
Develop a fiber-coupled self-mixing interferometer which can rapidly 
detect sub-micron displacements of a surface. Using displacement 
time-series data, analyze vibrational transfer functions of mechanical designs. 

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