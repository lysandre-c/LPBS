# Laboratory of the Physics of Biological Systems

## Overview

This project aims to build a comprehensive machine learning pipeline for analyzing movement patterns of *Caenorhabditis elegans* worms tracked on plates and classify worms that have been administered drugs versus control worms.

## Project Structure

```
LPBS/
├── README.md
├── requirements.txt
├── preprocessing.py
├── feature_extraction.py
├── feature_segment_model.py
├── ts_segment_model.py
├── ts_segment_cnn_analysis.py
└── ts_segment_rf_analysis.py
│
├── data/
│   ├── Lifespan/
│   │   ├── COMPANY DRUG/
│   │   ├── TERBINAFINE/
│   │   └── TERBINAFINE_OLD/
│   ├── Optogenetics/
│   │   ├── ATR-/
│   │   └── ATR+/
│   └── Skeletonization code/
│
├── preprocessed_data/
│   ├── full/
│   └── segments/
│
├── feature_data/
│   ├── full_features.csv
│   └── segments_features.csv
│
├── results/
│   ├── feature_segment_model/
│   └── ts_segment_model/
│
└── EDA/
    ├── correlation_analysis.py
    ├── eda_extracted_features.py
    ├── feature_comparison_analysis.py
    ├── run_all_analyses.py
    ├── full_data/
    └── segments_data/
```

## Usage

### 0. Setup
Clone the repository and install dependencies.
```bash
git clone <repository-url>
cd LPBS
pip install -r requirements.txt
```

### 1. Data Preprocessing
Process raw experimental data and create segment-level data. Output: `preprocessed_data/full` and `preprocessed_data/segments` folders.
```bash
python preprocessing.py
```

### 2. Feature Extraction
Extract hand-crafted features from preprocessed time series data. Output: `feature_data` folder.
```bash
python feature_extraction.py
```

### 3. Exploratory Data Analysis (optional)
Analysis of the hand-crafted features. Output: `EDA/full_data` and `EDA/segments_data` folders.
```bash
cd EDA
python run_all_analyses.py
```

### 4. Model Training and Evaluation

#### Feature-Based Models
Train segment-level models using extracted features. Output: `results/feature_segment_model` folder.
```bash
python feature_segment_model.py
```

#### Time Series Models
Train segment-level models on raw time series. Output: `results/ts_segment_model` folder.
```bash
python ts_segment_model.py
```

### 5. Model Analysis and Interpretation

(Only available for some time series models)
#### CNN Analysis
Analyze what patterns the CNN learned.
```bash
python ts_segment_cnn_analysis.py
```

#### Random Forest Analysis
Understand Random Forest decision patterns.
```bash
python ts_segment_rf_analysis.py
```

## Core Scripts Documentation

### `preprocessing.py`
- Cleans and standardizes raw *C. elegans* time series data
- Splits data into full series and segments
- Filters out files in `Optogenetics`

### `feature_extraction.py`
- Extracts statistical and temporal features from worm movement
- Creates both segment-level and full-trajectory features

### `ts_segment_model.py`
- Trains a segment-level classifier and voting classifiers for segment-to-timeseries prediction
- Segment-level models available : Random Forest, 1D CNN, LSTM
- Apply cross-validation with proper file-level splitting

### `feature_segment_model.py`
- Trains a segment-level classifier and voting classifiers for segment-to-timeseries prediction
- Segment-level models available : Random Forest, Gradient Boosting, KNN, neural networks (MLP)
- Apply cross-validation with proper file-level splitting

### Analysis Scripts
- `ts_segment_cnn_analysis.py`: CNN interpretation (gradients, temporal patterns)
- `ts_segment_rf_analysis.py`: Random Forest analysis (feature importance, decision trees)

## Configuration

### Model Settings
Key parameters can be modified in `ts_segment_model.py`:
```python
# Model selection
MODELS_TO_TEST = {
    'CNN_1D': True,           # 1D CNN for time series
    'LSTM': False,            # LSTM for sequences
    'Random_Forest_TS': True, # Random Forest on flattened data
    'TS_KNN_Euclidean': False # KNN classifier
}

# Time series processing
MAX_SEGMENT_LENGTH = 300      # Sequence length
USE_SUBSAMPLING = True        # vs truncation
TIME_SERIES_FEATURES = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']

# Training
FAST_MODE = True              # Quick training for testing
N_CV_SPLITS = 5               # Cross-validation folds
USE_TIME_SERIES_SCALING = True # Data normalization
```

### Cross-Validation Strategy
- **File-level splitting**: Prevents data leakage between train/test
- **Stratified folds**: Maintains class balance across splits

## Results and Outputs

### Model Performance
Results saved in `results/ts_segment_model/`:
- `cv_summary.csv`: Cross-validation performance summary
- `best_*_model.joblib`: Trained models for each algorithm
- `best_*_config.json`: Model configurations and hyperparameters
- Performance visualization plots