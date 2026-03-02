# Environment Setup

## Prerequisites

- **Python** 3.8 or higher (3.10+ recommended)
- **pip** or **conda**
- **Git**

---

## Option 1 — pip (virtual environment)

```bash
# Clone the repository
git clone https://github.com/Toddni8022/Capstone.git
cd Capstone

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install the capstone package in editable mode
pip install -e .
```

---

## Option 2 — conda

```bash
# Clone the repository
git clone https://github.com/Toddni8022/Capstone.git
cd Capstone

# Create the conda environment
conda env create -f environment.yml

# Activate it
conda activate capstone-ml

# Install the capstone package in editable mode
pip install -e .
```

---

## Download the Data

```bash
make data
```

Or in Python:

```python
from capstone.data import fetch_housing_data
fetch_housing_data()
```

This downloads `housing.tgz` and extracts `housing.csv` into `data/raw/housing/`.

---

## Run the Notebook

```bash
jupyter notebook notebooks/02_end_to_end_machine_learning_project.ipynb
```

---

## Run Tests

```bash
make test
# or
pytest tests/
```

---

## Lint the Code

```bash
make lint
# or
flake8 src/ tests/
```

---

## Verify Installation

```python
import capstone
print(capstone.__version__)  # should print 0.1.0
```
