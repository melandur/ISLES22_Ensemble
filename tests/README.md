# Testing Guide

This directory contains comprehensive unit tests for the ISLES22_Ensemble project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and pytest configuration
├── test_utils.py            # Tests for src/utils.py functions
├── test_isles22_ensemble.py # Tests for IslesEnsemble class
├── test_majority_voting.py  # Tests for majority voting logic
├── test_main.py             # Tests for main.py argument parsing
└── fixtures/                # Test data fixtures
    ├── __init__.py
    └── sample_images.py     # Helper functions to create test NIfTI images
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_utils.py

# Run specific test class
pytest tests/test_utils.py::TestConvertToNii

# Run specific test function
pytest tests/test_utils.py::TestConvertToNii::test_convert_nifti_file
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser

# Generate terminal coverage report
pytest --cov=src --cov-report=term-missing
```

### Test Markers

Tests can be filtered using markers:

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only GPU-required tests (if GPU available)
pytest -m requires_gpu
```

## Test Fixtures

### Available Fixtures

The `conftest.py` file provides several reusable fixtures:

- **`tmp_dir`**: Temporary directory for test files
- **`sample_dwi_path`**: Path to a sample DWI NIfTI file
- **`sample_adc_path`**: Path to a sample ADC NIfTI file
- **`sample_flair_path`**: Path to a sample FLAIR NIfTI file
- **`sample_mask_path`**: Path to a sample binary mask file
- **`sample_4d_dwi_path`**: Path to a sample 4D DWI NIfTI file
- **`ensemble_path`**: Mock ensemble directory structure
- **`output_dir`**: Output directory for tests

### Mock Fixtures

- **`mock_subprocess_run`**: Mocks `subprocess.run` calls
- **`mock_subprocess_call`**: Mocks `subprocess.call` calls
- **`mock_gpu_available`**: Mocks GPU availability check (sufficient memory)
- **`mock_gpu_unavailable`**: Mocks GPU unavailability
- **`mock_gpu_insufficient_memory`**: Mocks GPU with insufficient memory
- **`mock_requests_get`**: Mocks `requests.get` for downloading atlas
- **`mock_sitk_elastix`**: Mocks SimpleITK Elastix operations
- **`mock_hd_bet`**: Mocks hd-bet command
- **`mock_dcm2niix`**: Mocks dcm2niix command

### Using Fixtures

Fixtures are automatically injected into test functions:

```python
def test_example(sample_dwi_path, tmp_dir):
    """Example test using fixtures."""
    assert os.path.exists(sample_dwi_path)
    # Use tmp_dir for temporary files
    test_file = os.path.join(tmp_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
```

## Test Data Generation

The `fixtures/sample_images.py` module provides helper functions to generate synthetic NIfTI images:

- `create_test_nifti_image()`: Create a generic NIfTI image
- `create_test_dwi_image()`: Create a test DWI image
- `create_test_adc_image()`: Create a test ADC image
- `create_test_flair_image()`: Create a test FLAIR image
- `create_test_mask()`: Create a test binary mask
- `create_test_4d_dwi_image()`: Create a test 4D DWI image
- `create_test_prediction_array()`: Create test prediction arrays

Example usage:

```python
from tests.fixtures.sample_images import create_test_dwi_image

def test_example(tmp_dir):
    dwi_path = os.path.join(tmp_dir, 'dwi.nii.gz')
    create_test_dwi_image(output_path=dwi_path)
    assert os.path.exists(dwi_path)
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test Structure

```python
import pytest
import os
from src.utils import some_function

class TestSomeFunction:
    """Tests for some_function."""
    
    def test_some_function_basic(self, tmp_dir):
        """Test basic functionality."""
        result = some_function('input')
        assert result is not None
    
    def test_some_function_error_case(self, tmp_dir):
        """Test error handling."""
        with pytest.raises(ValueError):
            some_function('invalid_input')
```

### Best Practices

1. **Use fixtures** for common setup/teardown
2. **Mock external dependencies** (subprocess, network calls, GPU checks)
3. **Test edge cases** and error conditions
4. **Keep tests isolated** - each test should be independent
5. **Use descriptive test names** that explain what is being tested
6. **Group related tests** in test classes

## Test Coverage Goals

- Aim for >80% code coverage
- Test all public functions and methods
- Test error handling and edge cases
- Test input validation

## Continuous Integration

Tests should be run automatically in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the project is installed in development mode:

```bash
pip install -e .
```

### Fixture Not Found

If a fixture is not found, check that:
1. The fixture is defined in `conftest.py`
2. The fixture name matches exactly
3. The fixture is in the correct scope

### Mock Not Working

If mocks are not working as expected:
1. Check the patch path matches the import path in the source code
2. Ensure the mock is applied before the function is called
3. Verify the mock is in the correct scope

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest mocking](https://docs.pytest.org/en/stable/monkeypatch.html)

