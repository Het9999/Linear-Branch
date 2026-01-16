Linear Regression example

Files
- src/linear_regression.py : Minimal example using scikit-learn
- requirements.txt : dependencies
- config.yaml : configuration for data generation, model params, evaluation

Quick start

1. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the example:

```bash
python src/linear_regression.py
```

The script loads config from config.yaml, prints learned coefficient, intercept, MSE and R2, and will plot
the data if `matplotlib` is available.
