import warnings
import numpy as np

# Simple warning suppression
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

import os
import pandas as pd
import random
from time import time
import json
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Import deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

# Import time series specific libraries
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Device configuration (set to None for auto-detection, or specify 'cpu', 'cuda', 'mps')
FORCE_DEVICE = None  # None = auto-detect, 'cpu' = force CPU, 'mps' = force MPS, 'cuda' = force CUDA

# Detect best available device or use forced device
if FORCE_DEVICE is not None:
    DEVICE = FORCE_DEVICE
    if FORCE_DEVICE == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("  MPS forced but not available, falling back to CPU")
        DEVICE = 'cpu'
    elif FORCE_DEVICE == 'cuda' and not torch.cuda.is_available():
        print("  CUDA forced but not available, falling back to CPU")
        DEVICE = 'cpu'
    device_info = f"{DEVICE.upper()} (forced)"
else:
    # Auto-detect best device
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        device_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = 'mps'
        device_info = "MPS (Apple Silicon GPU)"
    else:
        DEVICE = 'cpu'
        device_info = "CPU"

# Configuration
MODELS_TO_TEST = {
    'TS_KNN_Euclidean': False,  # Standard KNN on flattened time series
    'Random_Forest_TS': True,  # Random Forest on flattened time series
    'CNN_1D': True,  # 1D CNN for time series using PyTorch
    'LSTM': False,  # LSTM for time series using PyTorch
}

# Time series processing configuration
MAX_SEGMENT_LENGTH = 300  # Maximum length for time series (subsampling/padding)
USE_SUBSAMPLING = True  # True: evenly spaced subsampling, False: truncation (for sequences > max_length)
TIME_SERIES_FEATURES = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']  # Features to use from time series
 
# Model configuration
FAST_MODE = True
USE_PROBABILITY_CALIBRATION = True
CALIBRATION_METHOD = 'sigmoid'
USE_TIME_SERIES_SCALING = True  # Scale time series data



# Parallelization configuration
N_PARALLEL_JOBS = min(os.cpu_count() // 2, 4)

# Cross-validation configuration
N_CV_SPLITS = 5

# Voting strategies for timeseries evaluation
VOTING_STRATEGIES = {
    'equal_weight': 'Equal weight voting (every vote counts as 1)',
    'confidence_weighted': 'Confidence weighted voting (votes weighted by model confidence)'
}

# Output configuration
OUTPUT_DIR = 'results/ts_segment_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Time Series Segment-to-Timeseries Cross-Validation Pipeline ===")
print(f"Available models: {[name for name, enabled in MODELS_TO_TEST.items() if enabled]}")
print(f"Time series features ({len(TIME_SERIES_FEATURES)}): {TIME_SERIES_FEATURES}")
print(f"Max segment length: {MAX_SEGMENT_LENGTH}")
print(f"Length standardization: {'SUBSAMPLING' if USE_SUBSAMPLING else 'TRUNCATION'} for longer sequences")
print(f"Scaling: {'ENABLED' if USE_TIME_SERIES_SCALING else 'DISABLED'}")
print(f"Calibration: {CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else 'DISABLED'}")
deep_learning_models = [name for name, enabled in MODELS_TO_TEST.items() if enabled and name in ['CNN_1D', 'LSTM']]
if deep_learning_models:
    print(f" Deep Learning Device: {DEVICE.upper()} (for {', '.join(deep_learning_models)})")
print(f"Cross-validation: {N_CV_SPLITS}-fold CV")
print(f"Parallelization: {N_PARALLEL_JOBS} parallel jobs")

def load_segment_timeseries_data():
    """Load raw time series data from preprocessed segments"""
    print("Loading time series segment data...")
    
    segments_dir = Path('preprocessed_data/segments')
    if not segments_dir.exists():
        raise FileNotFoundError(f"Segments directory not found: {segments_dir}")
    
    # Load metadata to get labels
    metadata_file = segments_dir / 'labels_and_metadata.csv'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = pd.read_csv(metadata_file)
    print(f"Found metadata for {len(metadata)} segments")
    
    # Load time series data
    time_series_data = []
    labels = []
    segment_ids = []
    original_files = []
    
    failed_loads = 0
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading segments"):
        # Build segment file path from relative_path and file name
        segment_file = segments_dir / row['relative_path'] / row['file']
        
        if not segment_file.exists():
            failed_loads += 1
            continue
            
        try:
            # Load segment time series
            segment_df = pd.read_csv(segment_file)
            
            # Map column names to our expected names
            column_mapping = {
                'x': 'x_coordinate',
                'y': 'y_coordinate', 
                'speed': 'speed',
                'turning_angle': 'turning_angle'
            }
            
            # Rename columns to match expected names
            segment_df = segment_df.rename(columns=column_mapping)
            
            # Check if required columns exist
            missing_cols = [col for col in TIME_SERIES_FEATURES if col not in segment_df.columns]
            if missing_cols:
                print(f"  Missing columns in {segment_file}: {missing_cols}")
                failed_loads += 1
                continue
            
            # Extract time series features (always multivariate)
            ts_data = segment_df[TIME_SERIES_FEATURES].values
            
            # Handle NaN values
            if np.isnan(ts_data).any():
                ts_data = np.nan_to_num(ts_data, nan=0.0)
            
            time_series_data.append(ts_data)
            labels.append(row['label'])
            segment_ids.append(row['segment_index'])  # Use segment_index instead of segment_id
            original_files.append(row['original_file'])
            
        except Exception as e:
            print(f"  Error loading {segment_file}: {e}")
            failed_loads += 1
            continue
    
    if failed_loads > 0:
        print(f"  Failed to load {failed_loads} segments")
    
    print(f"Successfully loaded {len(time_series_data)} time series segments")
    
    if len(time_series_data) == 0:
        raise ValueError("No time series data loaded!")
    
    # Analyze time series lengths
    lengths = [len(ts) for ts in time_series_data]
    print(f"Time series lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    return time_series_data, np.array(labels), np.array(segment_ids), np.array(original_files)

def subsample_or_pad_timeseries(time_series_list, target_length=MAX_SEGMENT_LENGTH):
    """Subsample or pad time series to fixed length using evenly spaced sampling"""
    print(f"Standardizing time series length to {target_length} using subsampling/padding...")
    
    standardized_ts = []
    n_features = time_series_list[0].shape[1] if len(time_series_list[0].shape) > 1 else 1
    
    # Track statistics
    original_lengths = [len(ts) for ts in time_series_list]
    n_subsampled = 0
    n_padded = 0
    n_exact = 0
    
    for ts in time_series_list:
        current_length = len(ts)
        
        if current_length > target_length:
            if USE_SUBSAMPLING:
                # Subsample using evenly spaced indices
                # This preserves temporal structure better than truncation
                indices = np.linspace(0, current_length - 1, target_length, dtype=int)
                subsampled_ts = ts[indices]
                standardized_ts.append(subsampled_ts)
                n_subsampled += 1
            else:
                # Truncate (keep first target_length points)
                truncated_ts = ts[:target_length]
                standardized_ts.append(truncated_ts)
                n_subsampled += 1  # Count as subsampled for stats
            
        elif current_length < target_length:
            # Pad with zeros (same as before for shorter sequences)
            if len(ts.shape) == 1:
                ts = ts.reshape(-1, 1)
            padding = np.zeros((target_length - current_length, ts.shape[1]))
            padded_ts = np.vstack([ts, padding])
            standardized_ts.append(padded_ts)
            n_padded += 1
            
        else:
            # Exact length - no change needed
            standardized_ts.append(ts)
            n_exact += 1
    
    # Print processing statistics
    processing_method = "Subsampled" if USE_SUBSAMPLING else "Truncated"
    print(f"    Processing stats:")
    print(f"      {processing_method} (longer): {n_subsampled} sequences")
    print(f"      Padded (shorter): {n_padded} sequences") 
    print(f"      Exact length: {n_exact} sequences")
    print(f"      Original length range: {min(original_lengths)}-{max(original_lengths)} (mean: {np.mean(original_lengths):.1f})")
    
    if n_subsampled > 0:
        # Calculate compression ratio for information
        longer_sequences = [l for l in original_lengths if l > target_length]
        avg_original = np.mean(longer_sequences)
        compression_ratio = target_length / avg_original
        if USE_SUBSAMPLING:
            print(f"      Subsampling ratio: {compression_ratio:.3f} (keeping {compression_ratio*100:.1f}% of time points)")
        else:
            print(f"      Truncation ratio: {compression_ratio:.3f} (keeping first {compression_ratio*100:.1f}% of time points)")
    
    return np.array(standardized_ts)





def scale_timeseries_data(X_train, X_test=None):
    """Scale time series data"""
    if not USE_TIME_SERIES_SCALING:
        return X_train, X_test, None
    
    # Use tslearn scaler for time series
    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    return X_train_scaled, X_test_scaled, scaler

class CNN1D(nn.Module):
    """Lightweight 1D CNN model for time series classification using PyTorch"""
    def __init__(self, input_size, n_classes=2):
        super(CNN1D, self).__init__()
        
        # First convolutional block (reduced filters: 64 -> 32)
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second convolutional block (reduced filters: 128 -> 64)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Global pooling instead of third conv block
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Simplified dense layers (reduced from 256->128->64 to 64->32)
        self.fc1 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc2 = nn.Linear(32, n_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        
        # Output
        x = self.fc2(x)
        
        return x

class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PyTorch CNN classifier"""
    def __init__(self, input_size=4, n_classes=2, epochs=50, batch_size=32, learning_rate=0.001, 
                 device=None, early_stopping_patience=10, random_state=42):
        self.input_size = input_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Use global device detection or provided device
        if device is not None:
            self.device = device
        else:
            self.device = DEVICE
            
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Print device info for debugging
        print(f"          CNN training on device: {self.device}")
        
        # Convert to PyTorch tensors and transpose for Conv1d (batch, features, sequence)
        X_tensor = torch.FloatTensor(X).transpose(1, 2).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Create model
        self.model = CNN1D(self.input_size, self.n_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # Clear cache for MPS to prevent memory issues
                if self.device == 'mps':
                    torch.mps.empty_cache()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"           Early stopping at epoch {epoch + 1}")
                    break
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).transpose(1, 2).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Clear cache for MPS
            if self.device == 'mps':
                torch.mps.empty_cache()
        
        return predictions
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).transpose(1, 2).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Clear cache for MPS
            if self.device == 'mps':
                torch.mps.empty_cache()
        
        return probabilities

class LSTM(nn.Module):
    """Lightweight LSTM model for time series classification using PyTorch"""
    def __init__(self, input_size, hidden_size=32, num_layers=2, n_classes=2, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        # LSTM expects (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        output = self.dropout(last_output)
        
        # Classification
        output = self.fc(output)
        
        return output



class PyTorchLSTMClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PyTorch LSTM classifier"""
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, n_classes=2, 
                 epochs=15, batch_size=32, learning_rate=0.001, dropout=0.2,
                 device=None, early_stopping_patience=5, random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        # Use global device detection or provided device
        if device is not None:
            self.device = device
        else:
            self.device = DEVICE
            
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Print device info for debugging
        print(f"          LSTM training on device: {self.device}")
        
        # Convert to PyTorch tensors (keep original shape for LSTM: batch, sequence, features)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Create model
        self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, 
                         self.n_classes, self.dropout).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # Clear cache for MPS to prevent memory issues
                if self.device == 'mps':
                    torch.mps.empty_cache()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"           Early stopping at epoch {epoch + 1}")
                    break
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Clear cache for MPS
            if self.device == 'mps':
                torch.mps.empty_cache()
        
        return predictions
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Clear cache for MPS
            if self.device == 'mps':
                torch.mps.empty_cache()
        
        return probabilities



def get_models(input_shape):
    """Get all available time series models"""
    models = {}
    
    if MODELS_TO_TEST.get('TS_KNN_Euclidean', False):
        # Flatten time series for standard KNN
        models['TS_KNN_Euclidean'] = Pipeline([
            ('flatten', FlattenTransformer()),
            ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=1))
        ])
    
    if MODELS_TO_TEST.get('Random_Forest_TS', False):
        models['Random_Forest_TS'] = Pipeline([
            ('flatten', FlattenTransformer()),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=1,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ])
    
    if MODELS_TO_TEST.get('CNN_1D', False):
        # input_shape is (samples, sequence_length, features)
        # For PyTorch Conv1d, we need (features,) as input_size
        input_size = input_shape[2] if len(input_shape) == 3 else input_shape[1]
        
        models['CNN_1D'] = PyTorchCNNClassifier(
            input_size=input_size,
            n_classes=2,
            epochs=15,
            batch_size=64,  # Increased batch size for faster training
            learning_rate=0.001,
            early_stopping_patience=5,  # Reduced patience for faster convergence
            random_state=RANDOM_SEED
        )
    
    if MODELS_TO_TEST.get('LSTM', False):
        # input_shape is (samples, sequence_length, features)
        # For LSTM, we need features as input_size
        input_size = input_shape[2] if len(input_shape) == 3 else input_shape[1]
        
        models['LSTM'] = PyTorchLSTMClassifier(
            input_size=input_size,
            hidden_size=32,
            num_layers=2,
            n_classes=2,
            epochs=15,
            batch_size=64,  # Increased batch size for faster training
            learning_rate=0.001,
            dropout=0.2,
            early_stopping_patience=5,
            random_state=RANDOM_SEED
        )
    

    
    return models

class FlattenTransformer:
    """Transformer to flatten time series data for non-time-series models"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(X.shape[0], -1)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

def get_model_params(model_name):
    """Get parameter grid for specific model"""
    if model_name == 'TS_KNN_Euclidean':
        return {
            'knn__n_neighbors': [3, 5, 7] if not FAST_MODE else [5]
        }
    elif model_name == 'Random_Forest_TS':
        return {
            'rf__n_estimators': [10] if FAST_MODE else [50, 100, 200],
            'rf__max_depth': [3] if FAST_MODE else [5, 10, 15]
        }
    elif model_name == 'CNN_1D':
        return {
            'epochs': [15] if FAST_MODE else [20, 25],
            'batch_size': [64] if FAST_MODE else [32, 64]
        }
    elif model_name == 'LSTM':
        return {
            'hidden_size': [32] if FAST_MODE else [16, 32, 64],
            'num_layers': [2] if FAST_MODE else [1, 2],
            'dropout': [0.2] if FAST_MODE else [0.1, 0.2, 0.3],
            'batch_size': [64] if FAST_MODE else [32, 64]
        }

    return {}



def create_kfold_splits(X, y, groups, n_splits=5):
    """Create proper k-fold splits at the file level to prevent data leakage"""
    
    # Get file-level labels
    file_labels = pd.DataFrame({'file': groups, 'label': y}).drop_duplicates('file').groupby('file')['label'].first()
    
    # Create stratified k-fold splits on files
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Generate all fold splits
    fold_splits = []
    files = file_labels.index.values
    labels = file_labels.values
    
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(files, labels)):
        train_files = files[train_idx]
        test_files = files[test_idx]
        
        # Create masks and split data
        train_mask = np.isin(groups, train_files)
        test_mask = np.isin(groups, test_files)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        groups_train = groups[train_mask]
        groups_test = groups[test_mask]
        
        fold_splits.append({
            'fold_id': fold_id,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_test': groups_test,
            'train_files': list(train_files),
            'test_files': list(test_files)
        })
        
        print(f"    Fold {fold_id+1}: {len(train_files)} train files ({len(X_train)} segments), {len(test_files)} test files ({len(X_test)} segments)")
    
    return fold_splits

def train_timeseries_model(X_train, y_train, groups_train, model_name, model, cv_split_id):
    """Train a time series model for one CV split"""
    print(f"       Training {model_name}...")
    
    # Scale data if needed
    X_train_scaled, _, scaler = scale_timeseries_data(X_train)
    
    # Get parameter grid
    param_grid = get_model_params(model_name)
    
    if param_grid:
        print(f"         Parameter grid: {param_grid}")
        # Use GroupKFold for hyperparameter tuning
        cv = GroupKFold(n_splits=3)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0
        )
        
        try:
            grid_search.fit(X_train_scaled, y_train, groups=groups_train)
            model = grid_search.best_estimator_
            print(f"          Best params: {grid_search.best_params_}")
            print(f"          Best score: {grid_search.best_score_:.4f}")
            
            # Verify the parameters were actually applied
            if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
                rf_model = model.named_steps['rf']
                print(f"          Actual RF params: n_estimators={rf_model.n_estimators}, max_depth={rf_model.max_depth}")
                
        except Exception as e:
            print(f"           Grid search failed: {e}, using default parameters")
            model.fit(X_train_scaled, y_train)
    else:
        print(f"         No parameter grid available, using default parameters")
        model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_segments(model, X_segments, scaler=None):
    """Predict class probabilities for time series segments"""
    # Scale test data if scaler is provided
    if scaler is not None:
        X_segments_scaled = scaler.transform(X_segments)
    else:
        X_segments_scaled = X_segments
    
    predictions = model.predict(X_segments_scaled)
    
    # Handle probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_segments_scaled)
        if probabilities.shape[1] == 2:
            prob_class_0 = probabilities[:, 0]
            prob_class_1 = probabilities[:, 1]
        else:
            # Multi-class case - use max probability as confidence
            prob_class_1 = np.max(probabilities, axis=1)
            prob_class_0 = 1 - prob_class_1
    else:
        # For models without probability prediction
        prob_class_1 = predictions.astype(float)
        prob_class_0 = 1 - prob_class_1
    
    confidence = np.maximum(prob_class_0, prob_class_1)
    
    return {
        'predicted_class': predictions,
        'prob_class_0': prob_class_0,
        'prob_class_1': prob_class_1,
        'confidence': confidence
    }

def apply_majority_voting(segment_predictions, groups, strategy='equal_weight'):
    """Apply majority voting to get full time series predictions"""
    results = []
    
    # Create DataFrame for easier grouping
    pred_df = pd.DataFrame(segment_predictions)
    pred_df['original_file'] = groups
    
    # Group by original file (time series)
    for file_name, group in pred_df.groupby('original_file'):
        if strategy == 'equal_weight':
            # Equal weight voting: every segment vote counts as 1
            votes = group['predicted_class'].value_counts()
            predicted_class = votes.index[0]
            confidence = votes.iloc[0] / len(group)
            
        elif strategy == 'confidence_weighted':
            # Confidence weighted voting: votes weighted by model prediction confidence
            weighted_votes = {}
            for _, row in group.iterrows():
                pred_class = row['predicted_class']
                conf = row['confidence']
                weighted_votes[pred_class] = weighted_votes.get(pred_class, 0) + conf
            
            predicted_class = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weighted_votes.values())
            confidence = weighted_votes[predicted_class] / total_weight
            
        else:
            raise ValueError(f"Unknown voting strategy: {strategy}")
        
        # Calculate average probabilities
        avg_prob_0 = group['prob_class_0'].mean()
        avg_prob_1 = group['prob_class_1'].mean()
        
        results.append({
            'file_name': file_name,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'avg_prob_0': avg_prob_0,
            'avg_prob_1': avg_prob_1,
            'num_segments': len(group),
            'segment_agreement': group['predicted_class'].value_counts().iloc[0] / len(group)
        })
    
    return pd.DataFrame(results)



def evaluate_timeseries_predictions(voting_results, true_labels, strategy_name):
    """Evaluate full time series predictions"""
    # Get true labels for each file
    file_labels = {}
    for file_name in voting_results['file_name']:
        file_labels[file_name] = true_labels[file_name]
    
    # Add true labels to results
    voting_results['true_label'] = voting_results['file_name'].map(file_labels)
    
    y_true = voting_results['true_label']
    y_pred = voting_results['predicted_class']
    y_prob = voting_results['avg_prob_1']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'strategy': strategy_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'num_samples': len(voting_results),
        'voting_results': voting_results
    }

def run_single_cv_split(cv_split_id, fold_split, input_shape):
    """Run a single cross-validation split using pre-computed fold data"""
    print(f"\n CV Fold {cv_split_id + 1}/{N_CV_SPLITS}")
    
    # Extract fold data
    X_train = fold_split['X_train']
    X_test = fold_split['X_test']
    y_train = fold_split['y_train']
    y_test = fold_split['y_test']
    groups_train = fold_split['groups_train']
    groups_test = fold_split['groups_test']
    train_files = fold_split['train_files']
    test_files = fold_split['test_files']
    
    # Create file-level true labels for test set
    test_file_labels = {}
    for file_name in test_files:
        # Get label from any segment of this file (all segments have same label)
        file_segments_mask = groups_test == file_name
        if np.any(file_segments_mask):
            segment_idx = np.where(file_segments_mask)[0][0]
            test_file_labels[file_name] = y_test[segment_idx]
    
    # Create file-level true labels for training set
    train_file_labels = {}
    for file_name in train_files:
        # Get label from any segment of this file (all segments have same label)
        file_segments_mask = groups_train == file_name
        if np.any(file_segments_mask):
            segment_idx = np.where(file_segments_mask)[0][0]
            train_file_labels[file_name] = y_train[segment_idx]
    
    cv_results = {}
    
    # Test each model
    models = get_models(input_shape)
    for model_name, model in models.items():
        print(f"\n    Testing {model_name}...")
        
        # Train model
        trained_model, scaler = train_timeseries_model(X_train, y_train, groups_train, model_name, model, cv_split_id)
        
        # Predict on training segments (for training metrics)
        train_segment_predictions = predict_segments(trained_model, X_train, scaler)
        
        # Evaluate training segment-level performance
        train_segment_accuracy = accuracy_score(y_train, train_segment_predictions['predicted_class'])
        train_segment_f1 = f1_score(y_train, train_segment_predictions['predicted_class'])
        train_segment_roc_auc = roc_auc_score(y_train, train_segment_predictions['prob_class_1'])
        
        # Predict on test segments
        segment_predictions = predict_segments(trained_model, X_test, scaler)
        
        # Evaluate test segment-level performance
        segment_accuracy = accuracy_score(y_test, segment_predictions['predicted_class'])
        segment_f1 = f1_score(y_test, segment_predictions['predicted_class'])
        segment_roc_auc = roc_auc_score(y_test, segment_predictions['prob_class_1'])
        
        print(f"       Train Segment: Acc={train_segment_accuracy:.3f}, F1={train_segment_f1:.3f}, AUC={train_segment_roc_auc:.3f}")
        print(f"       Test Segment: Acc={segment_accuracy:.3f}, F1={segment_f1:.3f}, AUC={segment_roc_auc:.3f}")
        
        # Apply different voting strategies for timeseries evaluation
        model_results = {
            'train_segment_performance': {
                'accuracy': train_segment_accuracy,
                'f1_score': train_segment_f1,
                'roc_auc': train_segment_roc_auc
            },
            'segment_performance': {
                'accuracy': segment_accuracy,
                'f1_score': segment_f1,
                'roc_auc': segment_roc_auc
            },
            'train_timeseries_performance': {},
            'timeseries_performance': {}
        }
        
        # Default threshold evaluation
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            # Training timeseries evaluation
            train_voting_results = apply_majority_voting(train_segment_predictions, groups_train, strategy_key)
            train_timeseries_eval = evaluate_timeseries_predictions(train_voting_results, train_file_labels, strategy_name)
            
            model_results['train_timeseries_performance'][strategy_key] = train_timeseries_eval
            
            # Test timeseries evaluation
            voting_results = apply_majority_voting(segment_predictions, groups_test, strategy_key)
            timeseries_eval = evaluate_timeseries_predictions(voting_results, test_file_labels, strategy_name)
            
            model_results['timeseries_performance'][strategy_key] = timeseries_eval
            
            strategy_short = strategy_name.split('(')[0].strip()
            print(f"       {strategy_short} Train (0.5): Acc={train_timeseries_eval['accuracy']:.3f}, "
                  f"F1={train_timeseries_eval['f1_score']:.3f}, AUC={train_timeseries_eval['roc_auc']:.3f}")
            print(f"       {strategy_short} Test (0.5): Acc={timeseries_eval['accuracy']:.3f}, "
                  f"F1={timeseries_eval['f1_score']:.3f}, AUC={timeseries_eval['roc_auc']:.3f}")
        

        
        cv_results[model_name] = model_results
    
    return {
        'cv_split_id': cv_split_id,
        'fold_id': cv_split_id,
        'train_files': train_files,
        'test_files': test_files,
        'results': cv_results
    }

def aggregate_cv_results(all_cv_results):
    """Aggregate results across all CV splits"""
    print(f"\n Aggregating {len(all_cv_results)} CV splits...")
    
    # Initialize aggregation structures
    model_names = list(all_cv_results[0]['results'].keys())
    strategy_keys = list(VOTING_STRATEGIES.keys())
    
    aggregated = {}
    
    for model_name in model_names:
        aggregated[model_name] = {
            'train_segment_performance': {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            },
            'segment_performance': {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            },
            'train_timeseries_performance': {},
            'timeseries_performance': {}
        }
        
        for strategy_key in strategy_keys:
            aggregated[model_name]['train_timeseries_performance'][strategy_key] = {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            }
            aggregated[model_name]['timeseries_performance'][strategy_key] = {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            }
            

    
    # Collect all results
    for cv_result in all_cv_results:
        for model_name in model_names:
            model_result = cv_result['results'][model_name]
            
            # Training segment performance
            train_seg_perf = model_result['train_segment_performance']
            aggregated[model_name]['train_segment_performance']['accuracy'].append(train_seg_perf['accuracy'])
            aggregated[model_name]['train_segment_performance']['f1_score'].append(train_seg_perf['f1_score'])
            aggregated[model_name]['train_segment_performance']['roc_auc'].append(train_seg_perf['roc_auc'])
            
            # Test segment performance
            seg_perf = model_result['segment_performance']
            aggregated[model_name]['segment_performance']['accuracy'].append(seg_perf['accuracy'])
            aggregated[model_name]['segment_performance']['f1_score'].append(seg_perf['f1_score'])
            aggregated[model_name]['segment_performance']['roc_auc'].append(seg_perf['roc_auc'])
            
            # Training timeseries performance
            for strategy_key in strategy_keys:
                train_ts_perf = model_result['train_timeseries_performance'][strategy_key]
                aggregated[model_name]['train_timeseries_performance'][strategy_key]['accuracy'].append(train_ts_perf['accuracy'])
                aggregated[model_name]['train_timeseries_performance'][strategy_key]['f1_score'].append(train_ts_perf['f1_score'])
                aggregated[model_name]['train_timeseries_performance'][strategy_key]['roc_auc'].append(train_ts_perf['roc_auc'])
                
                # Test timeseries performance
                ts_perf = model_result['timeseries_performance'][strategy_key]
                aggregated[model_name]['timeseries_performance'][strategy_key]['accuracy'].append(ts_perf['accuracy'])
                aggregated[model_name]['timeseries_performance'][strategy_key]['f1_score'].append(ts_perf['f1_score'])
                aggregated[model_name]['timeseries_performance'][strategy_key]['roc_auc'].append(ts_perf['roc_auc'])
                

    
    # Calculate statistics
    summary_stats = {}
    for model_name in model_names:
        summary_stats[model_name] = {
            'train_segment_performance': {},
            'segment_performance': {},
            'train_timeseries_performance': {},
            'timeseries_performance': {}
        }
        
        # Training segment performance statistics
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            values = aggregated[model_name]['train_segment_performance'][metric]
            summary_stats[model_name]['train_segment_performance'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Test segment performance statistics
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            values = aggregated[model_name]['segment_performance'][metric]
            summary_stats[model_name]['segment_performance'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Training timeseries performance statistics
        for strategy_key in strategy_keys:
            summary_stats[model_name]['train_timeseries_performance'][strategy_key] = {}
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                values = aggregated[model_name]['train_timeseries_performance'][strategy_key][metric]
                summary_stats[model_name]['train_timeseries_performance'][strategy_key][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Test timeseries performance statistics
        for strategy_key in strategy_keys:
            summary_stats[model_name]['timeseries_performance'][strategy_key] = {}
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                values = aggregated[model_name]['timeseries_performance'][strategy_key][metric]
                summary_stats[model_name]['timeseries_performance'][strategy_key][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            

    
    return summary_stats

def print_summary_results(summary_stats):
    """Print formatted summary of cross-validation results"""
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY RESULTS")
    print(f"{'='*80}")
    
    for model_name, model_stats in summary_stats.items():
        print(f"\n {model_name.upper()}")
        print("-" * 50)
        
        # Training segment-level performance
        print(" Training Segment-Level Performance:")
        train_seg_stats = model_stats['train_segment_performance']
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            stats = train_seg_stats[metric]
            print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        # Test segment-level performance
        print("\n Test Segment-Level Performance:")
        seg_stats = model_stats['segment_performance']
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            stats = seg_stats[metric]
            print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        # Training timeseries-level performance
        print("\n Training Timeseries-Level Performance (Default 0.5 Threshold):")
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            print(f"\n  {strategy_name}:")
            train_ts_stats = model_stats['train_timeseries_performance'][strategy_key]
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                stats = train_ts_stats[metric]
                print(f"    {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        # Test timeseries-level performance
        print("\n Test Timeseries-Level Performance (Default 0.5 Threshold):")
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            print(f"\n  {strategy_name}:")
            ts_stats = model_stats['timeseries_performance'][strategy_key]
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                stats = ts_stats[metric]
                print(f"    {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        


def train_and_save_all_models(X, y, original_files, summary_stats):
    """Train and save the best version of each model type on full dataset"""
    print(f"\n Training and saving best version of each model type...")
    
    # Get all available models
    models = get_models(X.shape)
    
    # Scale data once
    X_scaled, _, scaler = scale_timeseries_data(X)
    
    saved_models = {}
    
    for model_name in models.keys():
        print(f"\n    Training {model_name}...")
        
        # Get fresh model instance
        model = models[model_name]
        
        # Get parameter grid and perform hyperparameter optimization
        param_grid = get_model_params(model_name)
        if param_grid:
            print(f"      Parameter grid: {param_grid}")
            
            # Use GroupKFold for hyperparameter tuning
            cv = GroupKFold(n_splits=3)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0
            )
            
            try:
                grid_search.fit(X_scaled, y, groups=original_files)
                trained_model = grid_search.best_estimator_
                print(f"       Best hyperparameters: {grid_search.best_params_}")
                print(f"       Best CV score: {grid_search.best_score_:.4f}")
            except Exception as e:
                print(f"        Grid search failed: {e}, using default parameters")
                trained_model = model
                trained_model.fit(X_scaled, y)
        else:
            print(f"      No parameter grid, using default parameters")
            trained_model = model
            trained_model.fit(X_scaled, y)
        
        # Get best performance for this model
        model_stats = summary_stats[model_name]
        best_strategy = None
        best_score = 0
        
        for strategy_key in VOTING_STRATEGIES.keys():
            score = model_stats['timeseries_performance'][strategy_key]['accuracy']['mean']
            if score > best_score:
                best_score = score
                best_strategy = strategy_key
        
        # Save the trained model
        model_path = os.path.join(OUTPUT_DIR, f'best_{model_name.lower()}_model.joblib')
        dump(trained_model, model_path)
        
        # Save model configuration
        config = {
            'model_name': model_name,
            'model_type': type(trained_model).__name__,
            'best_voting_strategy': best_strategy,
            'best_accuracy': best_score,
            'time_series_features': TIME_SERIES_FEATURES,
            'max_segment_length': MAX_SEGMENT_LENGTH,
            'use_subsampling': USE_SUBSAMPLING,
            'scaling_enabled': USE_TIME_SERIES_SCALING,
            'calibration_enabled': USE_PROBABILITY_CALIBRATION,
            'calibration_method': CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else None,
            'training_samples': len(X),
            'n_features': len(TIME_SERIES_FEATURES),
            'n_files': len(np.unique(original_files)),
            'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            'random_seed': RANDOM_SEED,
            'fast_mode': FAST_MODE,
            'cv_performance': {
                strategy: {
                    'accuracy': model_stats['timeseries_performance'][strategy]['accuracy']['mean'],
                    'f1_score': model_stats['timeseries_performance'][strategy]['f1_score']['mean'],
                    'roc_auc': model_stats['timeseries_performance'][strategy]['roc_auc']['mean']
                }
                for strategy in VOTING_STRATEGIES.keys()
            }
        }
        
        # Add model-specific parameters
        if hasattr(trained_model, 'get_params'):
            try:
                model_params = trained_model.get_params()
                config['model_parameters'] = {k: str(v) for k, v in model_params.items()}
            except:
                config['model_parameters'] = "Could not extract parameters"
        
        config_path = os.path.join(OUTPUT_DIR, f'best_{model_name.lower()}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        saved_models[model_name] = {
            'model': trained_model,
            'config': config,
            'model_path': model_path,
            'config_path': config_path
        }
        
        print(f"       {model_name} saved: {model_path}")
    
    # Save the scaler (shared by all models)
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
    dump(scaler, scaler_path)
    
    return saved_models, scaler

def train_and_save_best_model(X, y, original_files, best_model_name, best_strategy):
    """Train and save the best model on full dataset for inference (legacy function)"""
    print(f"\n Training best overall model on full dataset...")
    print(f"   Model: {best_model_name}")
    print(f"   Strategy: {best_strategy}")
    
    # Get the best model
    models = get_models(X.shape)
    if best_model_name not in models:
        print(f" Error: Best model {best_model_name} not found in available models")
        return None
    
    model = models[best_model_name]
    
    # Train on full dataset
    X_scaled, _, scaler = scale_timeseries_data(X)
    
    # Get parameter grid and perform hyperparameter optimization
    param_grid = get_model_params(best_model_name)
    if param_grid:
        print(f"    Performing hyperparameter optimization with grid search...")
        print(f"   Parameter grid: {param_grid}")
        
        # Use GroupKFold for hyperparameter tuning on full dataset
        cv = GroupKFold(n_splits=3)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0
        )
        
        try:
            grid_search.fit(X_scaled, y, groups=original_files)
            model = grid_search.best_estimator_
            print(f"    Best hyperparameters: {grid_search.best_params_}")
            print(f"    Best CV score: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"     Grid search failed: {e}, using default parameters")
            model.fit(X_scaled, y)
    else:
        print(f"   No parameter grid available, using default parameters")
        model.fit(X_scaled, y)
    
    print(f"    Training {best_model_name} completed on {len(X)} segments")
    
    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, 'best_model.joblib')
    dump(model, model_path)
    
    # Save the scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'best_scaler.joblib')
    dump(scaler, scaler_path)
    
    # Save model configuration
    config = {
        'model_name': best_model_name,
        'model_type': type(model).__name__,
        'voting_strategy': best_strategy,
        'time_series_features': TIME_SERIES_FEATURES,
        'max_segment_length': MAX_SEGMENT_LENGTH,
        'use_subsampling': USE_SUBSAMPLING,
        'scaling_enabled': USE_TIME_SERIES_SCALING,
        'calibration_enabled': USE_PROBABILITY_CALIBRATION,
        'calibration_method': CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else None,
        'training_samples': len(X),
        'n_features': len(TIME_SERIES_FEATURES),
        'n_files': len(np.unique(original_files)),
        'class_balance': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'random_seed': RANDOM_SEED,
        'fast_mode': FAST_MODE
    }
    
    # Add model-specific parameters
    if hasattr(model, 'get_params'):
        try:
            model_params = model.get_params()
            config['model_parameters'] = {k: str(v) for k, v in model_params.items()}
        except:
            config['model_parameters'] = "Could not extract parameters"
    
    config_path = os.path.join(OUTPUT_DIR, 'best_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create inference example
    inference_example_path = os.path.join(OUTPUT_DIR, 'inference_example.py')
    inference_code = f'''"""
Example script for using the trained time series model for inference
"""
import numpy as np
import pandas as pd
import json

# Note: To use this inference example, you need to install joblib:
# pip install joblib

def load_trained_model():
    """Load the trained model and scaler"""
    from joblib import load  # Import only when needed for inference
    
    model = load('best_model.joblib')
    scaler = load('best_scaler.joblib')
    
    with open('best_model_config.json', 'r') as f:
        config = json.load(f)
    
    return model, scaler, config

def preprocess_segment(segment_df, config):
    """Preprocess a single segment for prediction"""
    # Extract time series features
    ts_data = segment_df[config['time_series_features']].values
    
    # Handle NaN values
    if np.isnan(ts_data).any():
        ts_data = np.nan_to_num(ts_data, nan=0.0)
    
    # Standardize length to max_length
    max_length = config['max_segment_length']
    current_length = len(ts_data)
    use_subsampling = config.get('use_subsampling', True)
    
    if current_length > max_length:
        if use_subsampling:
            # Subsample using evenly spaced indices (preserves temporal structure)
            indices = np.linspace(0, current_length - 1, max_length, dtype=int)
            ts_data = ts_data[indices]
        else:
            # Truncate (keep first max_length points)
            ts_data = ts_data[:max_length]
    elif current_length < max_length:
        # Pad with zeros
        padding = np.zeros((max_length - current_length, ts_data.shape[1]))
        ts_data = np.vstack([ts_data, padding])
    
    return ts_data

def predict_segments(model, scaler, segments_data):
    """Predict on multiple segments"""
    # Scale data
    if scaler is not None:
        segments_scaled = scaler.transform(segments_data)
    else:
        segments_scaled = segments_data
    
    # Predict
    predictions = model.predict(segments_scaled)
    probabilities = model.predict_proba(segments_scaled)
    
    return predictions, probabilities

def apply_voting_strategy(predictions, probabilities, strategy='{best_strategy}', threshold=0.5):
    """Apply voting strategy to get final prediction with default threshold"""
    if strategy == 'equal_weight':
        # Simple majority voting with default threshold
        avg_prob = np.mean(probabilities[:, 1])  # Average probability of class 1
        final_prediction = 1 if avg_prob > threshold else 0
        confidence = max(avg_prob, 1 - avg_prob)
    elif strategy == 'confidence_weighted':
        # Weight by confidence with default threshold
        confidences = np.max(probabilities, axis=1)
        weighted_prob = np.average(probabilities[:, 1], weights=confidences)
        final_prediction = 1 if weighted_prob > threshold else 0
        confidence = max(weighted_prob, 1 - weighted_prob)
    else:
        raise ValueError(f"Unknown strategy: {{strategy}}")
    
    return final_prediction, confidence

# Example usage:
if __name__ == "__main__":
    # Load model
    model, scaler, config = load_trained_model()
    print(f"Loaded model: {{config['model_name']}}")
    print(f"Features: {{config['time_series_features']}}")
    print(f"Strategy: {{config['voting_strategy']}}")
'''
    
    with open(inference_example_path, 'w') as f:
        f.write(inference_code)
    
    print(f" Best model saved:")
    print(f"    Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"     Config: {config_path}")
    print(f"   Inference example: {inference_example_path}")
    
    return model, scaler, config

def save_results(summary_stats):
    """Save results to CSV and JSON"""
    print(f"\nSaving results to {OUTPUT_DIR}...")
    
    # Save summary CSV
    csv_data = []
    for model_name, model_stats in summary_stats.items():
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            ts_stats = model_stats['timeseries_performance'][strategy_key]
            csv_data.append({
                'Model': model_name,
                'Strategy': strategy_key,
                'Accuracy_Mean': ts_stats['accuracy']['mean'],
                'Accuracy_Std': ts_stats['accuracy']['std'],
                'F1_Mean': ts_stats['f1_score']['mean'],
                'F1_Std': ts_stats['f1_score']['std'],
                'ROC_AUC_Mean': ts_stats['roc_auc']['mean'],
                'ROC_AUC_Std': ts_stats['roc_auc']['std']
            })
    
    summary_csv_path = os.path.join(OUTPUT_DIR, 'cv_summary.csv')
    pd.DataFrame(csv_data).to_csv(summary_csv_path, index=False)
    
    # Save detailed results
    results_json_path = os.path.join(OUTPUT_DIR, 'detailed_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f" Results saved:")
    print(f"    Summary CSV: {summary_csv_path}")
    print(f"    Detailed JSON: {results_json_path}")

def create_aggregated_confusion_matrix(all_cv_results):
    """Create aggregated confusion matrices across all CV folds"""
    print("Creating confusion matrix...")
    
    model_names = list(all_cv_results[0]['results'].keys())
    strategy_keys = list(VOTING_STRATEGIES.keys())
    
    # Create figure for confusion matrices
    n_models = len(model_names)
    n_strategies = len(strategy_keys)
    
    fig, axes = plt.subplots(n_models, n_strategies, figsize=(6*n_strategies, 6*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_strategies == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Confusion Matrices (Aggregated Across CV Folds)', fontsize=16, fontweight='bold')
    
    for model_idx, model_name in enumerate(model_names):
        for strategy_idx, strategy_key in enumerate(strategy_keys):
            ax = axes[model_idx, strategy_idx]
            strategy_name = VOTING_STRATEGIES[strategy_key].split('(')[0].strip()
            
            # Aggregate confusion matrices across all folds
            aggregated_cm = np.zeros((2, 2), dtype=int)
            total_accuracy = 0
            total_f1 = 0
            valid_folds = 0
            
            for cv_result in all_cv_results:
                try:
                    timeseries_perf = cv_result['results'][model_name]['timeseries_performance'][strategy_key]
                    cm = timeseries_perf['confusion_matrix']
                    aggregated_cm += cm
                    total_accuracy += timeseries_perf['accuracy']
                    total_f1 += timeseries_perf['f1_score']
                    valid_folds += 1
                except Exception as e:
                    continue
            
            if valid_folds > 0:
                # Calculate average metrics
                avg_accuracy = total_accuracy / valid_folds
                avg_f1 = total_f1 / valid_folds
                
                # Create heatmap
                sns.heatmap(aggregated_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                          xticklabels=['Class 0', 'Class 1'],
                          yticklabels=['Class 0', 'Class 1'])
                
                ax.set_title(f'{model_name.replace("_", " ")}\n{strategy_name}\n'
                           f'Avg Acc: {avg_accuracy:.3f}, Avg F1: {avg_f1:.3f}', fontsize=12)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name.replace("_", " ")}\n{strategy_name}\nNo data')
    
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Confusion matrices saved: {cm_plot_path}")
    
    plt.show()
    
    return fig

def create_visualization(summary_stats):
    """Create individual visualizations for each model showing both segment-level and timeseries-level metrics"""
    print("Creating individual model visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    model_names = list(summary_stats.keys())
    strategy_keys = list(VOTING_STRATEGIES.keys())
    strategy_names = [VOTING_STRATEGIES[k].split('(')[0].strip() for k in strategy_keys]
    
    created_figures = []
    
    # Create one figure per model
    for model_idx, model_name in enumerate(model_names):
        print(f"    Creating visualization for {model_name}...")
        
        # Create Figure for this model (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Performance Analysis', fontsize=16, fontweight='bold')
        
        model_stats = summary_stats[model_name]
        
        # Plot 1: Segment-Level Performance (Train vs Test)
        ax1 = axes[0, 0]
        
        # Get segment-level metrics for this model
        train_seg = model_stats['train_segment_performance']
        test_seg = model_stats['segment_performance']
        
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        metric_labels = ['Accuracy', 'F1 Score', 'ROC AUC']
        train_values = [train_seg[m]['mean'] for m in metrics]
        test_values = [test_seg[m]['mean'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
        
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Segment-Level Performance (Train vs Test)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars1, train_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, value in zip(bars2, test_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
        
        # Plot 2: Timeseries-Level Performance (Different Strategies)
        ax2 = axes[0, 1]
        
        # Get timeseries-level metrics across strategies for this model
        strategy_metrics = {}
        for strategy_key in strategy_keys:
            ts_stats = model_stats['timeseries_performance'][strategy_key]
            strategy_metrics[strategy_key] = {
                'accuracy': ts_stats['accuracy']['mean'],
                'f1_score': ts_stats['f1_score']['mean'],
                'roc_auc': ts_stats['roc_auc']['mean']
            }
        
        x_strat = np.arange(len(strategy_names))
        width_strat = 0.25
        
        accuracy_values = [strategy_metrics[strategy_keys[i]]['accuracy'] for i in range(len(strategy_keys))]
        f1_values = [strategy_metrics[strategy_keys[i]]['f1_score'] for i in range(len(strategy_keys))]
        auc_values = [strategy_metrics[strategy_keys[i]]['roc_auc'] for i in range(len(strategy_keys))]
        
        bars1 = ax2.bar(x_strat - width_strat, accuracy_values, width_strat, label='Accuracy', alpha=0.8, color='lightgreen')
        bars2 = ax2.bar(x_strat, f1_values, width_strat, label='F1 Score', alpha=0.8, color='orange')
        bars3 = ax2.bar(x_strat + width_strat, auc_values, width_strat, label='ROC AUC', alpha=0.8, color='lightblue')
        
        ax2.set_ylabel('Performance Score')
        ax2.set_title('Timeseries-Level Performance (Different Strategies)')
        ax2.set_xticks(x_strat)
        ax2.set_xticklabels(strategy_names)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bars, values in [(bars1, accuracy_values), (bars2, f1_values), (bars3, auc_values)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 3: Overfitting Analysis (Segment & Timeseries Level)
        ax3 = axes[1, 0]
        
        # Calculate overfitting gaps for this model
        segment_gaps = []
        timeseries_gaps = []
        level_labels = ['Accuracy', 'F1 Score', 'ROC AUC']
        
        # Segment-level gaps
        for metric in metrics:
            train_val = model_stats['train_segment_performance'][metric]['mean']
            test_val = model_stats['segment_performance'][metric]['mean']
            segment_gaps.append(train_val - test_val)
        
        # Timeseries-level gaps (using best strategy)
        best_strategy = max(strategy_keys, key=lambda s: model_stats['timeseries_performance'][s]['roc_auc']['mean'])
        for metric in metrics:
            train_val = model_stats['train_timeseries_performance'][best_strategy][metric]['mean']
            test_val = model_stats['timeseries_performance'][best_strategy][metric]['mean']
            timeseries_gaps.append(train_val - test_val)
        
        x_gap = np.arange(len(level_labels))
        width_gap = 0.35
        
        bars1 = ax3.bar(x_gap - width_gap/2, segment_gaps, width_gap, label='Segment Level', alpha=0.8, color='red')
        bars2 = ax3.bar(x_gap + width_gap/2, timeseries_gaps, width_gap, label='Timeseries Level', alpha=0.8, color='purple')
        
        ax3.set_ylabel('Overfitting Gap (Train - Test)')
        ax3.set_title('Overfitting Analysis: Segment vs Timeseries\n(Higher = More Overfitting)')
        ax3.set_xticks(x_gap)
        ax3.set_xticklabels(level_labels)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add threshold lines for gap severity
        ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good (<0.05)')
        ax3.axhline(y=0.1, color='yellow', linestyle='--', alpha=0.5, label='Moderate (<0.1)')
        ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High (>0.2)')
        
        # Add value labels
        for bars, values in [(bars1, segment_gaps), (bars2, timeseries_gaps)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 4: Model Configuration and Performance Summary
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create performance summary table for this model
        config_data = []
        config_data.append(['Metric', 'Value'])
        
        # Best strategy performance
        best_strategy_name = VOTING_STRATEGIES[best_strategy].split('(')[0].strip()
        best_auc = model_stats['timeseries_performance'][best_strategy]['roc_auc']['mean']
        best_acc = model_stats['timeseries_performance'][best_strategy]['accuracy']['mean']
        best_f1 = model_stats['timeseries_performance'][best_strategy]['f1_score']['mean']
        
        config_data.append(['Best Strategy', best_strategy_name])
        config_data.append(['Best ROC AUC', f'{best_auc:.3f}'])
        config_data.append(['Best Accuracy', f'{best_acc:.3f}'])
        config_data.append(['Best F1 Score', f'{best_f1:.3f}'])
        
        # Segment-level performance
        seg_auc = model_stats['segment_performance']['roc_auc']['mean']
        seg_acc = model_stats['segment_performance']['accuracy']['mean']
        seg_f1 = model_stats['segment_performance']['f1_score']['mean']
        
        config_data.append(['Segment ROC AUC', f'{seg_auc:.3f}'])
        config_data.append(['Segment Accuracy', f'{seg_acc:.3f}'])
        config_data.append(['Segment F1 Score', f'{seg_f1:.3f}'])
        
        # Overfitting analysis
        worst_gap_metric = metrics[np.argmax(segment_gaps)]
        worst_gap_value = max(segment_gaps)
        
        config_data.append(['Worst Gap Metric', worst_gap_metric.replace('_', ' ').title()])
        config_data.append(['Worst Gap Value', f'{worst_gap_value:.3f}'])
        
        # Performance variability
        auc_std = model_stats['timeseries_performance'][best_strategy]['roc_auc']['std']
        config_data.append(['Performance Std', f'{auc_std:.3f}'])
        
        table = ax4.table(cellText=config_data[1:],  # Skip header row
                         colLabels=config_data[0],
                         cellLoc='left', loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.3)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best performance rows
        table[(2, 0)].set_facecolor('#E8F5E8')  # Best ROC AUC
        table[(2, 1)].set_facecolor('#E8F5E8')
        
        ax4.set_title(f'{model_name.replace("_", " ").title()} Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the figure for this model
        fig_path = os.path.join(OUTPUT_DIR, f'{model_name.lower()}_performance_analysis.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"       {model_name} analysis saved: {fig_path}")
        
        created_figures.append(fig)
        
        # Show the figure
        plt.show()
    
    print(f" Created {len(created_figures)} individual model visualizations")
    return created_figures

def check_mps_availability():
    """Check MPS availability and provide helpful information"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(" MPS (Apple Silicon GPU) is available and will be used for CNN training")
        return True
    else:
        print("MPS (Apple Silicon GPU) is not available")
        if hasattr(torch.backends, 'mps'):
            print("   This might be because:")
            print("   - You're not on an Apple Silicon Mac (M1/M2/M3)")
            print("   - PyTorch version doesn't support MPS")
            print("   - macOS version is too old (requires macOS 12.3+)")
        else:
            print("   PyTorch version doesn't include MPS support")
        return False

def main():
    """Main pipeline"""
    start_time = time()
    
    print(f"Starting time series segment-to-timeseries cross-validation pipeline...")
    
    # Check device availability if CNN is enabled
    if MODELS_TO_TEST.get('CNN_1D', False):
        check_mps_availability()
    
    try:
        # Load time series data
        time_series_data, labels, segment_ids, original_files = load_segment_timeseries_data()
        
        # Analyze class balance
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f" Class balance: {dict(zip(unique_labels, counts))}")
        
        # Check if we have enough data
        if len(time_series_data) < 10:
            raise ValueError(f"Not enough data: only {len(time_series_data)} segments loaded")
        
        # Standardize time series length
        X = subsample_or_pad_timeseries(time_series_data, MAX_SEGMENT_LENGTH)
        print(f"Standardized time series shape: {X.shape}")
        
        # Check available models
        available_models = [name for name, enabled in MODELS_TO_TEST.items() if enabled]
        if not available_models:
            raise ValueError("No models available! Install required dependencies.")
        print(f"Available models: {available_models}")
        
        # Create k-fold splits
        print(f"Creating {N_CV_SPLITS}-fold CV splits...")
        fold_splits = create_kfold_splits(X, labels, original_files, n_splits=N_CV_SPLITS)
        
    except Exception as e:
        print(f" Error in data loading/preparation: {e}")
        return
    
    try:
        # Run cross-validation
        print(f"\nRunning {N_CV_SPLITS} CV folds...")
        all_cv_results = []
        
        for cv_split_id in range(N_CV_SPLITS):
            try:
                cv_result = run_single_cv_split(cv_split_id, fold_splits[cv_split_id], X.shape)
                all_cv_results.append(cv_result)
            except Exception as e:
                print(f" Error in CV fold {cv_split_id + 1}: {e}")
                continue
        
        if not all_cv_results:
            raise ValueError("No CV folds completed successfully")
        
        # Aggregate results
        summary_stats = aggregate_cv_results(all_cv_results)
        

        
        # Print summary
        print_summary_results(summary_stats)
        
        # Find best configuration
        best_model = None
        best_strategy = None
        best_score = 0
        
        for model_name, model_stats in summary_stats.items():
            for strategy_key in VOTING_STRATEGIES.keys():
                mean_acc = model_stats['timeseries_performance'][strategy_key]['accuracy']['mean']
                if mean_acc > best_score:
                    best_score = mean_acc
                    best_model = model_name
                    best_strategy = strategy_key
        
        # Train and save all models (each model type gets saved)
        saved_models, scaler = train_and_save_all_models(X, labels, original_files, summary_stats)
        
        # Also train and save the best overall model (for backward compatibility)
        train_and_save_best_model(X, labels, original_files, best_model, best_strategy)
        
        # Save results
        save_results(summary_stats)
        
        # Create visualizations
        create_visualization(summary_stats)
        
        # Create aggregated confusion matrix
        create_aggregated_confusion_matrix(all_cv_results)
        
    except Exception as e:
        print(f" Error in cross-validation execution: {e}")
        return
    
    print(f"\n BEST OVERALL CONFIGURATION:")
    print(f"   Model: {best_model}")
    print(f"   Strategy: {VOTING_STRATEGIES[best_strategy]}")
    print(f"   Mean Accuracy: {best_score:.4f}")
    
    best_stats = summary_stats[best_model]['timeseries_performance'][best_strategy]
    print(f"   Mean Accuracy: {best_stats['accuracy']['mean']:.4f} ± {best_stats['accuracy']['std']:.4f}")
    print(f"   Mean F1 Score: {best_stats['f1_score']['mean']:.4f} ± {best_stats['f1_score']['std']:.4f}")
    print(f"   Mean ROC AUC: {best_stats['roc_auc']['mean']:.4f} ± {best_stats['roc_auc']['std']:.4f}")
    
    print(f"\nSAVED MODELS:")
    for model_name, model_info in saved_models.items():
        config = model_info['config']
        print(f"   • {model_name}:")
        print(f"     Model file: {model_info['model_path']}")
        print(f"     Config file: {model_info['config_path']}")
        print(f"     Best strategy: {config['best_voting_strategy']}")
        print(f"     Best accuracy: {config['best_accuracy']:.4f}")
    
    print(f"\n MODEL COMPARISON SUMMARY:")
    for model_name, model_stats in summary_stats.items():
        best_strategy_for_model = max(VOTING_STRATEGIES.keys(), 
                                    key=lambda s: model_stats['timeseries_performance'][s]['accuracy']['mean'])
        best_acc = model_stats['timeseries_performance'][best_strategy_for_model]['accuracy']['mean']
        best_f1 = model_stats['timeseries_performance'][best_strategy_for_model]['f1_score']['mean']
        best_auc = model_stats['timeseries_performance'][best_strategy_for_model]['roc_auc']['mean']
        
        print(f"   • {model_name}: Acc={best_acc:.3f}, F1={best_f1:.3f}, AUC={best_auc:.3f} ({VOTING_STRATEGIES[best_strategy_for_model].split('(')[0].strip()})")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {time() - start_time:.1f}s")
    print(f"Tested {len([m for m, enabled in MODELS_TO_TEST.items() if enabled])} time series models across {N_CV_SPLITS} CV folds")
    print(f"Evaluated {len(VOTING_STRATEGIES)} voting strategies")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Individual model files: best_*_model.joblib")
    print(f"Model configurations: best_*_config.json")

if __name__ == "__main__":
    main() 