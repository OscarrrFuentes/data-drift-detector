# Data Drift Detector

**Online detection of data drift and adaptive normalization for ML datasets**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Data Drift Detector is a Python package for detecting distribution shifts in machine learning datasets using latent space representations. Born from research on ATLAS Higgs boson data, it provides:

- **Drift Detection**: PCA and autoencoder-based anomaly detection
- **Adaptive Normalization**: Custom transformations based on skewness/kurtosis
- **Visualization**: Distribution comparison tools

## Installation
```bash
pip install data-drift-detector
```

For development:
```bash
git clone https://github.com/OscarrrFuentes/data-drift-detector
cd data-drift-detector
pip install -e ".[dev]"
```

## Quick Start
```python
from data_drift_detector import DriftDetector

# Load your datasets
reference_data = ...  # Your baseline dataset
new_data = ...        # New data to check

# Detect drift
detector = DriftDetector(method='pca', n_components=10)
detector.fit(reference_data)
drift_score = detector.detect(new_data)

print(f"Drift detected: {drift_score > threshold}")
```

## Project Status

🚧 **Alpha** - Under active development

## Features (Planned)

- [x] Basic data loading utilities
- [x] Visualization tools
- [ ] PCA-based drift detection
- [ ] Autoencoder-based drift detection
- [ ] Clustering comparison (k-means)
- [ ] Statistical tests (KS, Wasserstein)
- [ ] Adaptive normalization strategies
- [ ] CLI tool

## Examples

See `examples/` directory for Jupyter notebooks demonstrating:
- Synthetic data generation
- Drift detection on real datasets
- Custom normalization strategies

## Contributing

Contributions welcome! This is a learning project, so feedback on code structure and best practices is especially appreciated.

## License

MIT License - see LICENSE file
