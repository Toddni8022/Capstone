# Contributing to Capstone

Thank you for considering contributing to this project! Here are a few guidelines to help you get started.

---

## How to Contribute

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```

2. **Install** the development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install flake8 pytest pytest-cov black
   ```

3. **Make your changes** — keep them focused and minimal.

4. **Run tests** to make sure nothing is broken:
   ```bash
   make test
   ```

5. **Lint** your code:
   ```bash
   make lint
   ```

6. **Commit** your changes with a descriptive message:
   ```bash
   git commit -m "feat: add XGBoost model comparison"
   ```

7. **Push** and open a Pull Request against `main`.

---

## Code Style

- Follow [PEP 8](https://pep8.org/) with a maximum line length of **100 characters**.
- Use `black` for formatting (configured in `pyproject.toml`).
- Add docstrings to public functions and classes.

---

## Reporting Bugs

Open a [GitHub Issue](https://github.com/Toddni8022/Capstone/issues) with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
