# Electrical Disturbance Classification with Compressive Sensing

## Overview

This project implements a machine learning pipeline for **identifying and classifying electrical disturbances in power transmission networks** using **Compressive Sensing (CS)** techniques. The system uses signal processing and sparse recovery methods to reconstruct electrical signals from compressed measurements, then applies neural networks and Support Vector Machines for classification.

The project combines:
- **Compressive Sensing**: OMP and Lasso-based sparse signal recovery
- **Signal Processing**: Wavelet decomposition and feature extraction
- **Machine Learning**: MLP, SVM, and Quadratic SVC classifiers
- **Optimization**: Hyperparameter tuning with GridSearchCV

---

## Key Features

✅ **Advanced Signal Processing**
- Wavelet-based signal decomposition
- Compressive measurement sampling (Phi matrices)
- Sparse coefficient recovery using OMP and Lasso algorithms

✅ **Feature Extraction**
- XPQRS feature extraction (15 features from signal derivatives)
- Energy and mobility-based features
- Top-K sparse coefficient selection
- PCA-based dimensionality reduction

✅ **Multiple Classification Models**
- Multi-Layer Perceptron (MLP) neural networks
- Support Vector Machine with RBF kernel
- Quadratic SVM (QSVC) classifiers

✅ **Performance Optimization**
- GPU acceleration support (CuPy, CuML)
- Parallel processing with Joblib
- Carbon emission tracking with CodeCarbon

✅ **Comprehensive Evaluation**
- Confusion matrix visualization
- Classification reports and metrics
- Model persistence and JSON logging
- Telegram notifications for training progress

---

## Installation

### Requirements
- Python >= 3.11
- See `pyproject.toml` for full dependencies

### Setup with Poetry

```bash
# Clone the repository
git clone <repository-url>
cd classify_electrical_disturbances_with_compressive_sensing

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Key Dependencies
```
numpy >= 1.24
scipy >= 1.16.1
scikit-learn >= 1.7.1
pandas >= 2.3.1
matplotlib >= 3.10.3
pywavelets >= 1.9.0
cvxpy >= 1.7.1
codecarbon >= 3.0.4
```

---

## Project Structure

```
classify_electrical_disturbances_with_compressive_sensing/
│
├── compressed_data_classification/          # Main classification module
│   ├── src/
│   │   ├── training/                        # ML model training
│   │   │   ├── main.py                      # Training orchestration
│   │   │   ├── mlp_training.py              # MLP classifier training
│   │   │   ├── svm_with_rbf_training.py    # SVM RBF training
│   │   │   ├── qsvc_training.py             # Quadratic SVC training
│   │   │   └── qsvc_training_without_gridsearch.py
│   │   │
│   │   ├── pipelines/                       # ML pipeline components
│   │   │   ├── CS_transformer.py            # Compressive sensing transformation
│   │   │   ├── FE_transformer.py            # XPQRS feature extraction
│   │   │
│   │   ├── cs/                              # Compressive sensing algorithms
│   │   ├── cs_omp/                          # OMP-based CS
│   │   ├── models/                          # Saved trained models
│   │   └── cs_constants/                    # Pre-computed CS matrices
│   │
│   ├── analysis/                            # Data analysis tools
│   │   ├── analyse_signals.py               # Signal visualization
│   │   ├── covert_report.py                 # Report parsing utilities
│   │
│   ├── data/
│   │   ├── raw/                             # Original datasets
│   │   └── processed/                       # Preprocessed data
│   │
│   └── utils/
│       └── utils.py                         # Utility functions (Telegram notifications)
│
├── compressed_data_localization/            # Localization-specific implementations
│   ├── learning_cs/                         # Learning resources and examples
│   │   ├── sample_and_reconstruct.py       # CS reconstruction examples
│   │   └── compressive_sensing/            # CS utilities
│   │
│   └── emissions_jose/                      # Carbon emission analysis
│
├── old/                                     # Legacy code and old versions
│
├── pyproject.toml                           # Project configuration and dependencies
├── poetry.lock                              # Dependency lock file
└── README.md                                # This file
```

---

## Quick Start

### 1. Data Preparation

```python
# Load dataset from CSV
from compressed_data_classification.src.training.main import import_and_split_dataset

X_train, X_test, y_train, y_test, label_encoder, X = import_and_split_dataset(
    'compressed_data_classification/data/raw/data.csv'
)
```

### 2. Train a Model

```bash
# Run the main training script
cd compressed_data_classification/src/training
python main.py
```

The script will:
- Load training data
- Apply compressive sensing transformation
- Extract features using XPQRS
- Train MLP, SVM, and QSVC models
- Save models and evaluation metrics
- Send notifications via Telegram

### 3. Use Trained Models for Prediction

```python
import joblib
from compressed_data_classification.src.pipelines.CS_transformer import CompressiveSensingTransformer
from compressed_data_classification.src.pipelines.FE_transformer import XPQRSFeatureExtractor

# Load saved model
model = joblib.load('compressed_data_classification/src/models/best_models_result/mlp/model_reconstructed_2_dot_5.pkl')

# Preprocess new data
cs_transformer = CompressiveSensingTransformer(technique='reconstructed_2_dot_5')
fe_extractor = XPQRSFeatureExtractor()

# Transform and predict
X_transformed = cs_transformer.transform(X_new)
X_features = fe_extractor.transform(X_transformed)
predictions = model.predict(X_features)
```

---

## Module Documentation

### Core Modules

#### 1. **training/main.py**
Main entry point orchestrating the entire training pipeline.

**Key Functions:**
- `import_and_split_dataset(data_path)`: Loads and splits data into train/test sets
- Manages multiple compressive sensing techniques
- Coordinates training of different classifiers

**Features:**
- Supports multiple data compression techniques
- Automatic model persistence
- Telegram notifications on completion

#### 2. **pipelines/CS_transformer.py**
Scikit-learn compatible transformer for compressive sensing-based feature extraction.

**Key Class:** `CompressiveSensingTransformer`

**Parameters:**
- `technique`: Feature extraction method (energy, topk, pca, pure_alpha)
- `lasso_alpha`: Regularization parameter for sparse recovery
- `K_topk`: Number of top coefficients to keep
- `n_jobs`: Parallel processing configuration
- GPU acceleration support via CuPy

**Methods:**
- `load_cs_structures()`: Load pre-computed CS matrices
- `save_cs_structures()`: Save CS structures for later use
- `transform()`: Apply CS-based feature extraction
- `fit()`: Fit transformer (no-op for stateless transformer)

#### 3. **pipelines/FE_transformer.py**
Feature extractor implementing XPQRS method.

**Key Class:** `XPQRSFeatureExtractor`

**Features Extracted (15 total):**
- Log Energy (LE)
- Shannon Energy (SE)
- Mobility (Mob)

Extracted from:
- Original signal
- 1st derivative
- 2nd derivative
- 3rd derivative
- 4th derivative

**Methods:**
- `_get_approximated_derivatives(x)`: Compute signal derivatives
- `_extract_features(u)`: Extract LE, SE, Mob from signal
- `transform(X)`: Process multiple signals
- `fit(X, y)`: Sklearn-compatible fit (stateless)

#### 4. **training/mlp_training.py**
Multi-Layer Perceptron classifier training with full pipeline.

**Key Function:** `mlp_training()`

**Features:**
- Integrates CS transformation and feature extraction
- GridSearchCV-based hyperparameter tuning (10-fold CV)
- Confusion matrix visualization
- Detailed metrics reporting (accuracy, ROC-AUC, various scores)
- Model and results persistence

#### 5. **analysis/analyse_signals.py**
Signal visualization tools for exploratory data analysis.

**Features:**
- Loads CSV data with electrical signals
- Groups by disturbance class
- Plots representative signals
- Saves visualizations

#### 6. **analysis/covert_report.py**
Utilities for parsing and consolidating classification reports.

**Functions:**
- `parse_classification_report()`: Convert sklearn report to DataFrame
- `process_reports()`: Extract reports from JSON files and consolidate to CSV

#### 7. **utils/utils.py**
Utility functions for the project.

**Key Function:** `send_telegram_msg(message)`
- Send training progress/completion notifications via Telegram Bot API
- Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env file

---

## Compressive Sensing Explanation

### What is Compressive Sensing?

Compressive Sensing (CS) is a signal processing technique that recovers sparse signals from far fewer measurements than traditional sampling requires.

**Process:**
1. **Measurement**: Original signal x is multiplied by measurement matrix Φ to get y = Φx
2. **Sparse Basis**: Signal is sparse in some basis (e.g., wavelets)
3. **Recovery**: Use Lasso or OMP algorithms to recover sparse coefficients α from Φα = y
4. **Reconstruction**: Reconstruct signal from sparse coefficients

**Matrices:**
- **Φ (Phi)**: Measurement matrix (M × N), where M << N
- **Ψ (Psi)**: Sparsifying basis (wavelet dictionary)
- **A**: Combined dictionary (Φ × Ψ)

### Implemented Algorithms

#### OMP (Orthogonal Matching Pursuit)
- Iterative algorithm for sparse recovery
- Selects atoms greedily from dictionary
- Parameter: Number of non-zero coefficients (sparsity)

#### Lasso (L1-regularization)
- Convex optimization for sparse recovery
- Minimizes: ||y - Φα||₂² + λ||α||₁
- Parameter: λ (regularization strength)

### Feature Extraction Techniques

1. **Energy**: Features from energy in frequency bands
2. **Top-K**: Select K largest sparse coefficients
3. **PCA**: PCA reduction of sparse coefficients
4. **Direct**: All sparse coefficients

---

## Data Format

### Input Data CSV Format
```
signal_1, signal_2, ..., signal_N, target
0.5,      0.3,      ..., 0.7,      Class_A
0.6,      0.4,      ..., 0.8,      Class_B
...
```

**Expected:** 
- Columns: N signal samples + 1 target column
- Target: Class label (categorical)
- Rows: Individual signals (samples)

---

## Configuration

### Environment Variables (.env file)

Create `.env` in project root for Telegram notifications:
```
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Compressive Sensing Configuration

Located in CS_transformer and training modules:
- `SAMPLE_M`: Number of measurements
- `WAVELET`: Wavelet function (db8, db4, etc.)
- `WAVELET_LEVEL`: Wavelet decomposition level
- `PARAM_VAL`: Lasso alpha or OMP sparsity

---

## Performance Metrics

The pipeline evaluates models using:

- **Accuracy**: Overall classification correctness
- **ROC-AUC**: Area under ROC curve (multi-class: ovr strategy)
- **MSE**: Mean squared error on test set
- **Classification Report**: Per-class precision, recall, F1-score
- **Confusion Matrix**: Visual representation of classification results

Results saved to: `compressed_data_classification/src/models/best_models_result/`

---

## GPU Acceleration

Optional GPU support via CuPy and CuML:

```bash
# Install GPU libraries (requires CUDA)
pip install cupy-cuda11x cuml
```

GPU features (automatic fallback if not available):
- Lasso solver acceleration
- Batch matrix operations
- Significant speedup for large datasets

---

## Troubleshooting

### Issue: CS structures not found
**Solution**: Load pre-computed CS matrices or run CS tuning first
```python
cs_transformer.load_cs_structures(
    'compressed_data_classification/src/cs_omp/cs_constants'
)
```

### Issue: Telegram notifications failing
**Solution**: Check .env file has correct TELEGRAM_TOKEN and TELEGRAM_CHAT_ID

### Issue: Out of memory with GPU
**Solution**: Reduce batch size or disable GPU acceleration
```python
cs_transformer = CompressiveSensingTransformer(use_gpu=False)
```

---

## Results and Outputs

### Generated Files

**Models:**
- `compressed_data_classification/src/models/best_models_result/{classifier}/model_{technique}.pkl`

**Metrics:**
- `compressed_data_classification/src/models/best_models_result/{classifier}/results_{technique}.json`
- `compressed_data_classification/src/models/best_models_result/{classifier}/report_result_{technique}.json`

**Visualizations:**
- `compressed_data_classification/src/models/best_models_result/{classifier}/plots/confusion_matrix_{technique}.png`

**Excel Reports:**
- `compressed_data_classification/src/models/best_models_result/resultados_modelos_{technique}.xlsx`

---

## References

**Signal Processing:**
- Wavelet decomposition techniques
- Sparse signal recovery theory
- Compressive sensing fundamentals

**Machine Learning:**
- Sklearn documentation for classifiers and pipelines
- Hyperparameter optimization strategies
- Cross-validation techniques

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Author

**Maxnasc7** - Project Creator and Maintainer

---

## Acknowledgments

- Scikit-learn for ML algorithms and pipeline framework
- CuPy/CuML for GPU acceleration
- CodeCarbon for carbon footprint tracking
- Wavelet computing via PyWavelets
