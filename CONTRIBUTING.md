# Contributing to CORC-NAH

Thank you for your interest in contributing to the CORC-NAH project! This document provides guidelines for contributing to this Náhuatl/Maya linguistic data pipeline.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project is dedicated to preserving and promoting indigenous languages. We expect all contributors to:
- Respect cultural sensitivity around indigenous language content
- Maintain professional and inclusive communication
- Give proper attribution to language sources and communities

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- (Optional) Scala 2.12 + SBT for Spark jobs
- (Optional) Docker for local Airflow testing

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/saidmoreno808/nahuatl-data-pipeline.git
cd nahuatl-data-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
make test
```

---

## Development Process

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Write tests** for new functionality
5. **Run quality checks**:
   ```bash
   make check  # Runs formatting, linting, type checking, and tests
   ```
6. **Commit** with descriptive messages:
   ```bash
   git commit -m "feat: add fuzzy deduplication for dialect variants"
   ```
7. **Push** to your fork and **create a Pull Request**

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

---

## Coding Standards

### Python

- **Style**: Follow PEP 8 (enforced by `black` and `flake8`)
- **Line length**: 100 characters
- **Type hints**: Required for all functions and methods
- **Docstrings**: Google style for all public APIs

Example:

```python
def normalize_unicode(text: str, form: str = "NFC") -> str:
    \"\"\"
    Normalize Unicode text to preserve macrons in Náhuatl.
    
    Args:
        text: Input text to normalize
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
    
    Returns:
        Normalized text string
    
    Raises:
        ValueError: If form is not a valid normalization form
    \"\"\"
    if form not in ["NFC", "NFD", "NFKC", "NFKD"]:
        raise ValueError(f"Invalid normalization form: {form}")
    
    return unicodedata.normalize(form, text)
```

### Scala

- **Style**: Follow [Scala Style Guide](https://docs.scala-lang.org/style/)
- **Formatting**: Use `scalafmt` (configured in `.scalafmt.conf`)
- **Immutability**: Prefer `val` over `var`

### SQL

- **Keywords**: UPPERCASE (`SELECT`, `FROM`, `WHERE`)
- **Identifiers**: snake_case (`customer_id`, `created_at`)
- **Formatting**: Use `sqlfluff` for consistent style

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests (no I/O)
├── integration/    # Tests with external dependencies
└── fixtures/       # Test data
```

### Writing Tests

- **Coverage target**: >90% for new code
- Use `pytest` fixtures for setup/teardown
- Mock external dependencies (APIs, databases)
- Test edge cases and error handling

Example:

```python
def test_incremental_load_empty_table(mock_oracle_connector):
    \"\"\"Test incremental load handles empty tables gracefully.\"\"\"
    connector = OracleConnector(...)
    
    results = list(connector.incremental_load(
        table="EMPTY_TABLE",
        watermark_column="updated_at",
        last_sync="2026-01-01T00:00:00"
    ))
    
    assert len(results) == 0
```

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_connectors.py -v

# With coverage
make coverage
```

---

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add/update tests** to maintain >90% coverage
3. **Run all quality checks**:
   ```bash
   make check  # Must pass before PR submission
   ```
4. **Update CHANGELOG.md** under "Unreleased" section
5. **Create PR** with:
   - Descriptive title: `feat: add Teradata connector for CDC loads`
   - Detailed description of changes
   - Link to related issues (if any)
6. **Respond to review feedback** promptly
7. **Squash commits** before merge (if requested)

### PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No secrets/credentials in code
- [ ] Pre-commit hooks run successfully

---

## Project Structure

```
corc-nah-enterprise/
├── src/
│   ├── pipeline/         # ETL pipeline modules
│   ├── connectors/       # Database connectors
│   └── scala_examples/   # Scala/Spark implementations
├── tests/
│   ├── unit/
│   └── integration/
├── airflow_dags/         # Airflow orchestration
├── terraform/            # Infrastructure as Code
├── sql/                  # SQL schemas and queries
├── docs/                 # Documentation
│   └── adr/              # Architecture Decision Records
└── great_expectations/   # Data quality definitions
```

---

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating duplicates
- For security vulnerabilities, email: said.moreno@email.com

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

