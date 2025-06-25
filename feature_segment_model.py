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
import threading

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

from joblib import dump, load, Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Wide MLP config giving 0.661 accuracy
#FAST_MODE = True
#USE_FEATURE_SELECTION = True
#N_FEATURES_TO_SELECT = 10
#USE_PROBABILITY_CALIBRATION = True
#CALIBRATION_METHOD = 'sigmoid'
#USE_SMOTE_AUGMENTATION = True
#SMOTE_RATIO = 2.0
#model__alpha': [0.0001]
#model__learning_rate_init': [0.01]
#model__hidden_layer_sizes': [(128, 64)]

# MLP config giving 0.675 accuracy
#FAST_MODE = True
#USE_FEATURE_SELECTION = True
#N_FEATURES_TO_SELECT = 56
#USE_PROBABILITY_CALIBRATION = True
#CALIBRATION_METHOD = 'sigmoid'
#USE_SMOTE_AUGMENTATION = True
#SMOTE_RATIO = 2.0
#model__alpha': [0.01]
#model__learning_rate_init': [0.01]
#model__hidden_layer_sizes': [(256, 128, 64)]


# Configuration
MODELS_TO_TEST = {
    'Random Forest': False,
    'Gradient Boosting': False,
    'K-Neighbors': False,
    'Wide MLP': True
}

# Parallelization configuration
N_PARALLEL_JOBS = min(os.cpu_count() // 2, 4)  # Use half of CPU cores, max 4 to avoid memory issues

# Thread-safe lock for global variable access
_global_lock = threading.Lock()

# Segment training configuration
FAST_MODE = True
USE_FEATURE_SELECTION = True
N_FEATURES_TO_SELECT = 56
USE_PROBABILITY_CALIBRATION = True
CALIBRATION_METHOD = 'sigmoid' #isotonic, sigmoid,
USE_SMOTE_AUGMENTATION = True
SMOTE_RATIO = 2.0

# Threshold optimization configuration
USE_THRESHOLD_OPTIMIZATION = False
THRESHOLD_OPTIMIZATION_METRIC = 'accuracy'  # 'f1' or 'accuracy'

# Feature selection range testing configuration
TEST_FEATURE_RANGE = False  # Set to True to test different numbers of features
FEATURE_RANGE_VALUES = list(range(19, 58, 1))  # Different numbers of features to test
FEATURE_RANGE_METRIC = 'accuracy'  # Metric to optimize for feature selection ('accuracy', 'f1_score', 'roc_auc')

# Cross-validation configuration
N_CV_SPLITS = 5  # Number of folds for proper k-fold cross-validation

# Voting strategies for timeseries evaluation
VOTING_STRATEGIES = {
    'equal_weight': 'Equal weight voting (every vote counts as 1)',
    'confidence_weighted': 'Confidence weighted voting (votes weighted by model confidence)'
}

# Output configuration
OUTPUT_DIR = 'results/feature_segment_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model features (same as segment_model_pipeline.py)
MODEL_FEATURES = [
    'mean_speed', 'std_speed', 'max_speed', 'min_speed', 'total_distance', 'time_paused', 'fraction_paused',
    'mean_turning_angle', 'std_turning_angle', 'max_turning_angle', 'turning_frequency',
    'mean_meandering_ratio', 'std_meandering_ratio', 'min_meandering_ratio', 'max_meandering_ratio', 
    'median_meandering_ratio', 'fraction_efficient_movement', 'turning_entropy', 'turning_fractal_dim', 
    'turning_dominant_freq', 'turning_spectral_centroid', 'speed_entropy', 'speed_fractal_dim',
    'wavelet_speed_level0', 'wavelet_speed_level1', 'wavelet_speed_level2', 'wavelet_speed_level3',
    'wavelet_turning_level0', 'wavelet_turning_level1', 'wavelet_turning_level2', 'wavelet_turning_level3',
    'mean_roaming_score', 'std_roaming_score', 'fraction_roaming',
    'mean_frenetic_score', 'max_frenetic_score', 'std_frenetic_score', 'pct_high_frenetic', 'mean_jerk', 'max_jerk',
    'kinetic_energy_proxy', 'movement_efficiency', 'speed_persistence', 'exploration_ratio',
    'speed_skewness', 'speed_kurtosis', 'speed_iqr', 'speed_cv',
    'speed_trend', 'speed_acceleration_mean', 'speed_acceleration_std',
    'speed_dominant_freq', 'speed_spectral_centroid',
    'activity_level', 'high_activity_fraction', 'low_activity_fraction', 'mixed_activity_fraction'
]

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find optimal threshold for binary classification through grid search.
    This addresses probability calibration issues where the default 0.5 threshold
    may not be optimal for the model's actual probability distribution.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        metric: 'f1' or 'accuracy' - metric to optimize
    
    Returns:
        best_threshold: Optimal threshold value
        best_score: Best metric score achieved
        thresholds: All tested threshold values
        scores: Metric scores for all thresholds
    """
    thresholds = np.arange(0.1, 0.9, 0.01)  # Test 80 thresholds
    best_threshold = 0.5
    best_score = 0
    
    scores = []
    for threshold in thresholds:
        # Convert probabilities to binary predictions using current threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate target metric
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Metric must be 'f1' or 'accuracy'")
        
        scores.append(score)
        
        # Track best threshold
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, thresholds, scores

def analyze_probability_calibration(y_true, y_prob, strategy_name, cv_split_id=None):
    """Analyze probability calibration of predictions"""
    # Class-specific statistics (always calculate)
    prob_class_0 = y_prob[y_true == 0]
    prob_class_1 = y_prob[y_true == 1]
    
    # Find optimal thresholds (always calculate, even if not printing)
    opt_threshold_f1, best_f1, thresholds, f1_scores = find_optimal_threshold(y_true, y_prob, 'f1')
    opt_threshold_acc, best_acc, _, acc_scores = find_optimal_threshold(y_true, y_prob, 'accuracy')
    
    # Performance at default threshold (0.5) (always calculate)
    y_pred_default = (y_prob >= 0.5).astype(int)
    default_f1 = f1_score(y_true, y_pred_default)
    default_acc = accuracy_score(y_true, y_pred_default)
    
    # Only show detailed calibration analysis in non-parallel mode
    if N_PARALLEL_JOBS == 1:
        split_info = f"CV Split {cv_split_id+1}" if cv_split_id is not None else ""
        strategy_short = strategy_name.split('(')[0].strip()
        print(f"\n          Calibration Analysis: {strategy_short}")
        
        # Basic statistics
        print(f"            Mean prob: {y_prob.mean():.3f} ± {y_prob.std():.3f}")
        print(f"            Class 0: {prob_class_0.mean():.3f}, Class 1: {prob_class_1.mean():.3f}")
        print(f"            Optimal F1 threshold: {opt_threshold_f1:.3f} (F1 = {best_f1:.3f})")
        print(f"            Optimal Acc threshold: {opt_threshold_acc:.3f} (Acc = {best_acc:.3f})")
        
        # Improvement potential
        f1_improvement = ((best_f1 - default_f1) / default_f1) * 100 if default_f1 > 0 else 0
        acc_improvement = ((best_acc - default_acc) / default_acc) * 100 if default_acc > 0 else 0
        
        print(f"            Improvement potential: F1 +{f1_improvement:.1f}%, Acc +{acc_improvement:.1f}%")
    
    return {
        'optimal_threshold_f1': opt_threshold_f1,
        'optimal_threshold_acc': opt_threshold_acc,
        'optimal_threshold_accuracy': opt_threshold_acc,  # Add alias for accuracy
        'best_f1': best_f1,
        'best_acc': best_acc,
        'default_f1': default_f1,
        'default_acc': default_acc,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'acc_scores': acc_scores,
        'prob_stats': {
            'mean': y_prob.mean(),
            'std': y_prob.std(),
            'class_0_mean': prob_class_0.mean(),
            'class_1_mean': prob_class_1.mean()
        }
    }

print("=== Segment-to-Timeseries Cross-Validation Pipeline ===")

print(f"Models: {', '.join(name for name, enabled in MODELS_TO_TEST.items() if enabled)}")

if TEST_FEATURE_RANGE:
    print(f"FEATURE SELECTION EXPERIMENT MODE:")
    print(f"  Feature counts to test: {FEATURE_RANGE_VALUES}")
    print(f"  Optimization metric: {FEATURE_RANGE_METRIC}")
    print(f"  Total experiments: {len(FEATURE_RANGE_VALUES)} × {N_CV_SPLITS} CV splits = {len(FEATURE_RANGE_VALUES) * N_CV_SPLITS}")
else:
    print(f"Feature selection: {N_FEATURES_TO_SELECT if USE_FEATURE_SELECTION else 'DISABLED'}")

print(f"Calibration: {CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else 'DISABLED'}")
print(f"SMOTE: {SMOTE_RATIO}x" if USE_SMOTE_AUGMENTATION else "SMOTE: DISABLED")

print(f"Threshold optimization: {'ENABLED' if USE_THRESHOLD_OPTIMIZATION else 'DISABLED'}")
if USE_THRESHOLD_OPTIMIZATION:
    print(f"  Optimization metric: {THRESHOLD_OPTIMIZATION_METRIC}")
    print(f"  Grid search: 0.1 to 0.9 (step=0.01, 80 thresholds)")
print(f"Cross-validation: {N_CV_SPLITS}-fold CV (proper k-fold, 100% coverage)")
print(f"Voting strategies: {len(VOTING_STRATEGIES)}")
print(f"Parallelization: {N_PARALLEL_JOBS} parallel jobs")

def load_segment_data():
    """Load and clean segment data"""
    print("Loading segment data...")
    df = pd.read_csv('feature_data/segments_features.csv')
    X, y, groups = df[MODEL_FEATURES].copy(), df['label'].copy(), df['original_file'].copy()
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X = X.drop(X.columns[X.var() == 0], axis=1)  # Remove constant columns
    
    # Outlier clipping and noise injection
    for col in X.columns:
        q05, q95 = X[col].quantile([0.05, 0.95])
        X[col] = X[col].clip(q05, q95)
        X[col] += np.random.normal(0, X[col].std() * 0.01, len(X))  # 1% noise
    
    # Analyze class balance at segment and file level
    segment_balance = y.value_counts(normalize=True)
    file_labels = df.groupby('original_file')['label'].first()
    file_balance = file_labels.value_counts(normalize=True)
    
    print(f"Loaded {X.shape[0]} segments, {X.shape[1]} features, {groups.nunique()} files")
    print(f"Segment balance: Class 0: {segment_balance[0]:.1%}, Class 1: {segment_balance[1]:.1%}")
    print(f"File balance: Class 0: {file_balance[0]:.1%}, Class 1: {file_balance[1]:.1%}")
    
    return X.astype(np.float64), y, groups

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
        train_mask = groups.isin(train_files)
        test_mask = groups.isin(test_files)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        groups_train = groups[train_mask].reset_index(drop=True)
        groups_test = groups[test_mask].reset_index(drop=True)
        
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
        
        # Only show detailed split info in non-parallel mode
        if N_PARALLEL_JOBS == 1:
            print(f"   Fold {fold_id+1}: {len(train_files)} train files ({len(X_train)} segments), {len(test_files)} test files ({len(X_test)} segments)")
    
    return fold_splits

def apply_smote_augmentation(X, y, ratio=2.0):
    """Apply SMOTE augmentation"""
    if not USE_SMOTE_AUGMENTATION:
        return X, y
    
    # Only show SMOTE details in non-parallel mode
    if N_PARALLEL_JOBS == 1:
        print(f"         SMOTE: {len(X)} → {int(len(X) * ratio)} samples")
    
    target_size = int(len(X) * ratio / 2)  # Balanced classes
    smote = SMOTE(
        sampling_strategy={label: target_size for label in pd.Series(y).value_counts().index}, 
        random_state=RANDOM_SEED, 
        k_neighbors=3
    )
    
    try:
        X_res, y_res = smote.fit_resample(X, y)
        indices = np.random.permutation(len(X_res))
        return (
            pd.DataFrame(X_res, columns=X.columns).iloc[indices].reset_index(drop=True), 
            pd.Series(y_res).iloc[indices].reset_index(drop=True)
        )
    except Exception as e:
        if N_PARALLEL_JOBS == 1:
            print(f"         SMOTE failed: {e}")
        return X, y



def get_models():
    """Get all available models"""
    # Adjust n_jobs for sklearn models to work with our parallel jobs
    sklearn_n_jobs = max(1, os.cpu_count() // N_PARALLEL_JOBS)
    
    all_models = {
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=sklearn_n_jobs, min_samples_split=10, 
            min_samples_leaf=5, max_features='sqrt'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=RANDOM_SEED, subsample=0.8, min_samples_split=10
        ),
        'K-Neighbors': KNeighborsClassifier(n_jobs=sklearn_n_jobs, n_neighbors=15),
        'Wide MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.0001,
            learning_rate_init=0.001, max_iter=300, random_state=RANDOM_SEED, 
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
        )
    }
    return {name: model for name, model in all_models.items() if MODELS_TO_TEST.get(name, False)}

def get_model_params(model_name):
    """Get parameter grid for specific model"""
    if model_name == 'Random Forest':
        return {
            'model__n_estimators': [50, 100],
            'model__max_depth': [3, 5, 8],
            'model__min_samples_split': [10, 20],
            'model__min_samples_leaf': [5, 10],
            'model__max_features': ['sqrt', 0.5]
        } if FAST_MODE else {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [3, 5, 8, 12],
            'model__min_samples_split': [10, 20, 50],
            'model__min_samples_leaf': [5, 10, 15],
            'model__max_features': ['sqrt', 'log2', 0.5, 0.3]
        }
    elif model_name == 'Gradient Boosting':
        return {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [2, 3],
            'model__subsample': [0.8],
            'model__min_samples_split': [10, 20]
        } if FAST_MODE else {
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [2, 3, 4],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__min_samples_split': [10, 20, 50]
        }
    elif model_name == 'K-Neighbors':
        return {'model__n_neighbors': [10, 20, 30]}
    elif model_name == 'Wide MLP':
        #return {
        #    'model__alpha': [0.0001, 0.001],
        #    'model__learning_rate_init': [0.001, 0.01]
        return {
            'model__alpha': [0.01],
            'model__learning_rate_init': [0.01],
            'model__hidden_layer_sizes': [(256, 128, 64)]
        } if FAST_MODE else {
            'model__alpha': [0.001, 0.01],
            'model__learning_rate_init': [0.01],
            'model__hidden_layer_sizes': [(128, 64), (64, 32), (64, 64, 32)]
        }
    return {}

def train_segment_model(X_train, y_train, groups_train, model_name, cv_split_id):
    """Train a single segment model for one CV split"""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available")
    
    model = models[model_name]
    
    # Only show detailed training info in non-parallel mode
    if N_PARALLEL_JOBS == 1:
        print(f"      Training {model_name}...")
    
    # Create pipeline
    steps = [('scaler', StandardScaler())]
    if USE_FEATURE_SELECTION:
        steps.append(('selector', SelectKBest(score_func=f_classif, k=N_FEATURES_TO_SELECT)))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps)
    param_grid = get_model_params(model_name)
    
    # Adjust n_jobs for GridSearch and Calibration to work with parallel jobs
    sklearn_n_jobs = max(1, os.cpu_count() // N_PARALLEL_JOBS)
    
    # Hyperparameter tuning if parameters available
    if param_grid:
        cv = GroupKFold(n_splits=3)  # Use 3 folds for speed
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=sklearn_n_jobs, verbose=0
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        pipeline = grid_search.best_estimator_
        if N_PARALLEL_JOBS == 1:
            print(f"         Best params: {grid_search.best_params_}")
    else:
        pipeline.fit(X_train, y_train)
    
    # Apply calibration if enabled
    if USE_PROBABILITY_CALIBRATION:
        if N_PARALLEL_JOBS == 1:
            print(f"         Applying {CALIBRATION_METHOD} calibration...")
        cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        calibrated_pipeline = CalibratedClassifierCV(
            pipeline, cv=cv_cal, method=CALIBRATION_METHOD, n_jobs=sklearn_n_jobs
        )
        calibrated_pipeline.fit(X_train, y_train)
        pipeline = calibrated_pipeline
    
    # Apply SMOTE and retrain if needed
    if USE_SMOTE_AUGMENTATION:
        X_train_smote, y_train_smote = apply_smote_augmentation(X_train, y_train, SMOTE_RATIO)
        if not USE_PROBABILITY_CALIBRATION:  # Only retrain if not calibrated
            pipeline.fit(X_train_smote, y_train_smote)
    
    return pipeline

def predict_segments(model, X_segments):
    """Predict class probabilities for segments"""
    predictions = model.predict(X_segments)
    probabilities = model.predict_proba(X_segments)
    
    return {
        'predicted_class': predictions,
        'prob_class_0': probabilities[:, 0],
        'prob_class_1': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1)
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

def apply_majority_voting_with_optimal_threshold(segment_predictions, groups, strategy='equal_weight', optimal_threshold=0.5):
    """Apply majority voting with optimized threshold"""
    results = []
    
    # Create DataFrame for easier grouping
    pred_df = pd.DataFrame(segment_predictions)
    pred_df['original_file'] = groups
    
    # Group by original file (time series)
    for file_name, group in pred_df.groupby('original_file'):        
        # Apply optimal threshold to segment predictions
        optimized_predictions = (group['prob_class_1'] >= optimal_threshold).astype(int)
        
        if strategy == 'equal_weight':
            # Equal weight voting with optimized threshold
            votes = optimized_predictions.value_counts()
            predicted_class = votes.index[0]
            confidence = votes.iloc[0] / len(group)
            
        elif strategy == 'confidence_weighted':
            # Weight votes by prediction confidence with optimized threshold
            weighted_votes = {}
            for idx, (_, row) in enumerate(group.iterrows()):
                pred_class = optimized_predictions.iloc[idx]
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
            'segment_agreement': optimized_predictions.value_counts().iloc[0] / len(group),
            'threshold_used': optimal_threshold
        })
    
    return pd.DataFrame(results)

def evaluate_timeseries_predictions(voting_results, true_labels, strategy_name):
    """Evaluate full time series predictions"""
    # Get true labels for each file
    file_labels = {}
    for file_name in voting_results['file_name']:
        # Find true label from segments data (all segments from same file have same label)
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

def run_single_cv_split_parallel(args):
    """Wrapper for parallel execution of CV splits"""
    cv_split_id, fold_split, pbar = args
    result = run_single_cv_split(cv_split_id, fold_split)
    
    # Update progress bar
    if pbar:
        pbar.set_postfix({
            'Fold': f"{cv_split_id + 1}/{N_CV_SPLITS}",
            'Files': f"{len(fold_split['test_files'])} test"
        })
        pbar.update(1)
    
    return result

def run_single_cv_split(cv_split_id, fold_split):
    """Run a single cross-validation split using pre-computed fold data"""
    # Simplified header - only show when not in parallel mode
    if N_PARALLEL_JOBS == 1:
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
        file_segments = groups_test[groups_test == file_name]
        if len(file_segments) > 0:
            segment_idx = file_segments.index[0]
            test_file_labels[file_name] = y_test.iloc[segment_idx]
    
    cv_results = {}
    
    # Test each model
    models = get_models()
    for model_name in models.keys():
        if N_PARALLEL_JOBS == 1:
            print(f"\n    Testing {model_name}...")
        
        # Train segment model
        segment_model = train_segment_model(X_train, y_train, groups_train, model_name, cv_split_id)
        
        # Predict on test segments
        segment_predictions = predict_segments(segment_model, X_test)
        
        # Evaluate segment-level performance
        segment_accuracy = accuracy_score(y_test, segment_predictions['predicted_class'])
        segment_f1 = f1_score(y_test, segment_predictions['predicted_class'])
        segment_roc_auc = roc_auc_score(y_test, segment_predictions['prob_class_1'])
        
        if N_PARALLEL_JOBS == 1:
            print(f"       Segment: Acc={segment_accuracy:.3f}, F1={segment_f1:.3f}, AUC={segment_roc_auc:.3f}")
            # Show probability distribution
            prob_mean = segment_predictions['prob_class_1'].mean()
            prob_std = segment_predictions['prob_class_1'].std()
            print(f"       Prob stats: Mean={prob_mean:.3f}, Std={prob_std:.3f}")
        
        # Apply different voting strategies for timeseries evaluation
        model_results = {
            'segment_performance': {
                'accuracy': segment_accuracy,
                'f1_score': segment_f1,
                'roc_auc': segment_roc_auc
            },
            'timeseries_performance': {},
            'calibration_analyses': {},
            'optimized_performance': {}
        }
        
        # Default threshold evaluation
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            voting_results = apply_majority_voting(segment_predictions, groups_test, strategy_key)
            timeseries_eval = evaluate_timeseries_predictions(voting_results, test_file_labels, strategy_name)
            
            model_results['timeseries_performance'][strategy_key] = timeseries_eval
            
            if N_PARALLEL_JOBS == 1:
                strategy_short = strategy_name.split('(')[0].strip()
                print(f"       {strategy_short}: Acc={timeseries_eval['accuracy']:.3f}, "
                      f"F1={timeseries_eval['f1_score']:.3f}, AUC={timeseries_eval['roc_auc']:.3f}")
        
        # Threshold optimization if enabled (using proper nested validation)
        if USE_THRESHOLD_OPTIMIZATION:
            if N_PARALLEL_JOBS == 1:
                print(f"\n       Threshold Optimization (Nested CV):")
            
            # Create validation split from training data for threshold optimization
            from sklearn.model_selection import train_test_split
            train_files_array = np.array(train_files)
            
            # Get labels for train files to ensure stratified split
            train_file_labels = {}
            for file_name in train_files:
                file_segments = groups_train[groups_train == file_name]
                if len(file_segments) > 0:
                    segment_idx = file_segments.index[0]
                    train_file_labels[file_name] = y_train.iloc[segment_idx]
            
            train_labels_array = np.array([train_file_labels[f] for f in train_files])
            
            # Split training files into train/val (80/20 split of training data)
            try:
                train_files_thresh, val_files_thresh = train_test_split(
                    train_files_array, test_size=0.2, random_state=RANDOM_SEED + cv_split_id,
                    stratify=train_labels_array
                )
            except ValueError:  # If stratification fails (too few samples)
                train_files_thresh, val_files_thresh = train_test_split(
                    train_files_array, test_size=0.2, random_state=RANDOM_SEED + cv_split_id
                )
            
            # Create validation data masks
            val_mask = groups_test.isin(val_files_thresh)  # Use test data structure but different files
            # Actually, we need to use training data for validation
            train_val_mask = groups_train.isin(val_files_thresh)
            
            # Get validation predictions from the already trained model
            if train_val_mask.sum() > 0:  # If we have validation data
                X_val = X_train[train_val_mask]
                y_val = y_train[train_val_mask]
                groups_val = groups_train[train_val_mask]
                
                val_predictions = predict_segments(segment_model, X_val)
                
                for strategy_key, strategy_name in VOTING_STRATEGIES.items():
                    # Apply voting on validation data
                    val_voting_results = apply_majority_voting(val_predictions, groups_val, strategy_key)
                    
                    # Create validation true labels
                    val_file_labels = {}
                    for file_name in val_files_thresh:
                        if file_name in train_file_labels:
                            val_file_labels[file_name] = train_file_labels[file_name]
                    
                    # Only proceed if we have validation results
                    if len(val_voting_results) > 0 and len(val_file_labels) > 0:
                        val_voting_results['true_label'] = val_voting_results['file_name'].map(val_file_labels)
                        val_voting_results = val_voting_results.dropna(subset=['true_label'])
                        
                        if len(val_voting_results) > 1:  # Need at least 2 samples
                            y_val_true = val_voting_results['true_label']
                            y_val_prob = val_voting_results['avg_prob_1']
                            
                            # Analyze probability calibration on validation data
                            calibration_analysis = analyze_probability_calibration(
                                y_val_true, y_val_prob, strategy_name, cv_split_id
                            )
                            model_results['calibration_analyses'][strategy_key] = calibration_analysis
                            
                            # Apply optimized threshold to TEST data
                            optimal_threshold = calibration_analysis[f'optimal_threshold_{THRESHOLD_OPTIMIZATION_METRIC}']
                            
                            if N_PARALLEL_JOBS == 1:
                                strategy_short = strategy_name.split('(')[0].strip()
                                print(f"          Testing {strategy_short} with threshold {optimal_threshold:.3f}...")
                            
                            optimized_voting_results = apply_majority_voting_with_optimal_threshold(
                                segment_predictions, groups_test, strategy_key, optimal_threshold
                            )
                            optimized_eval = evaluate_timeseries_predictions(
                                optimized_voting_results, test_file_labels, f"{strategy_name} (Optimized)"
                            )
                            
                            model_results['optimized_performance'][strategy_key] = optimized_eval
                            
                            # Calculate improvement
                            original_f1 = model_results['timeseries_performance'][strategy_key]['f1_score']
                            original_acc = model_results['timeseries_performance'][strategy_key]['accuracy']
                            
                            f1_improvement = ((optimized_eval['f1_score'] - original_f1) / original_f1) * 100 if original_f1 > 0 else 0
                            acc_improvement = ((optimized_eval['accuracy'] - original_acc) / original_acc) * 100 if original_acc > 0 else 0
                            
                            if N_PARALLEL_JOBS == 1:
                                print(f"             F1: {original_f1:.3f} → {optimized_eval['f1_score']:.3f} (+{f1_improvement:.1f}%)")
                                print(f"             Acc: {original_acc:.3f} → {optimized_eval['accuracy']:.3f} (+{acc_improvement:.1f}%)")
                        else:
                            # Skip threshold optimization for this strategy if not enough validation data
                            if N_PARALLEL_JOBS == 1:
                                print(f"         ⚠️  Skipping {strategy_name}: Not enough validation data")
                    else:
                        # Skip threshold optimization for this strategy if no validation results
                        if N_PARALLEL_JOBS == 1:
                            print(f"         ⚠️  Skipping {strategy_name}: No validation results")
            else:
                # Skip threshold optimization if no validation data
                if N_PARALLEL_JOBS == 1:
                    print(f"         ⚠️  Skipping threshold optimization: No validation data")
        
        cv_results[model_name] = model_results
    
    return {
        'cv_split_id': cv_split_id,
        'fold_id': cv_split_id,  # Same as cv_split_id for k-fold CV
        'train_files': train_files,
        'test_files': test_files,
        'results': cv_results
    }

def aggregate_cv_results(all_cv_results):
    """Aggregate results across all CV splits"""
    # Only show aggregation header in non-parallel mode
    if N_PARALLEL_JOBS == 1:
        print(f"\n Aggregating {len(all_cv_results)} CV splits...")
    
    # Initialize aggregation structures
    model_names = list(all_cv_results[0]['results'].keys())
    strategy_keys = list(VOTING_STRATEGIES.keys())
    
    aggregated = {}
    
    for model_name in model_names:
        aggregated[model_name] = {
            'segment_performance': {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            },
            'timeseries_performance': {},
            'optimized_performance': {},
            'calibration_analyses': {}
        }
        
        for strategy_key in strategy_keys:
            aggregated[model_name]['timeseries_performance'][strategy_key] = {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            }
            aggregated[model_name]['optimized_performance'][strategy_key] = {
                'accuracy': [],
                'f1_score': [],
                'roc_auc': []
            }
            aggregated[model_name]['calibration_analyses'][strategy_key] = {
                'optimal_threshold_f1': [],
                'optimal_threshold_acc': [],
                'best_f1': [],
                'best_acc': [],
                'default_f1': [],
                'default_acc': []
            }
    
    # Collect all results
    for cv_result in all_cv_results:
        for model_name in model_names:
            model_result = cv_result['results'][model_name]
            
            # Segment performance
            seg_perf = model_result['segment_performance']
            aggregated[model_name]['segment_performance']['accuracy'].append(seg_perf['accuracy'])
            aggregated[model_name]['segment_performance']['f1_score'].append(seg_perf['f1_score'])
            aggregated[model_name]['segment_performance']['roc_auc'].append(seg_perf['roc_auc'])
            
            # Timeseries performance
            for strategy_key in strategy_keys:
                ts_perf = model_result['timeseries_performance'][strategy_key]
                aggregated[model_name]['timeseries_performance'][strategy_key]['accuracy'].append(ts_perf['accuracy'])
                aggregated[model_name]['timeseries_performance'][strategy_key]['f1_score'].append(ts_perf['f1_score'])
                aggregated[model_name]['timeseries_performance'][strategy_key]['roc_auc'].append(ts_perf['roc_auc'])
                
                # Optimized performance (if available)
                if 'optimized_performance' in model_result and strategy_key in model_result['optimized_performance']:
                    opt_perf = model_result['optimized_performance'][strategy_key]
                    aggregated[model_name]['optimized_performance'][strategy_key]['accuracy'].append(opt_perf['accuracy'])
                    aggregated[model_name]['optimized_performance'][strategy_key]['f1_score'].append(opt_perf['f1_score'])
                    aggregated[model_name]['optimized_performance'][strategy_key]['roc_auc'].append(opt_perf['roc_auc'])
                
                # Calibration analyses (if available)
                if 'calibration_analyses' in model_result and strategy_key in model_result['calibration_analyses']:
                    cal_analysis = model_result['calibration_analyses'][strategy_key]
                    aggregated[model_name]['calibration_analyses'][strategy_key]['optimal_threshold_f1'].append(cal_analysis['optimal_threshold_f1'])
                    aggregated[model_name]['calibration_analyses'][strategy_key]['optimal_threshold_acc'].append(cal_analysis['optimal_threshold_acc'])
                    aggregated[model_name]['calibration_analyses'][strategy_key]['best_f1'].append(cal_analysis['best_f1'])
                    aggregated[model_name]['calibration_analyses'][strategy_key]['best_acc'].append(cal_analysis['best_acc'])
                    aggregated[model_name]['calibration_analyses'][strategy_key]['default_f1'].append(cal_analysis['default_f1'])
                    aggregated[model_name]['calibration_analyses'][strategy_key]['default_acc'].append(cal_analysis['default_acc'])
    
    # Calculate statistics
    summary_stats = {}
    for model_name in model_names:
        summary_stats[model_name] = {
            'segment_performance': {},
            'timeseries_performance': {},
            'optimized_performance': {},
            'calibration_stats': {}
        }
        
        # Segment performance statistics
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            values = aggregated[model_name]['segment_performance'][metric]
            summary_stats[model_name]['segment_performance'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Timeseries performance statistics
        for strategy_key in strategy_keys:
            # Default timeseries performance
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
            
            # Optimized timeseries performance
            if aggregated[model_name]['optimized_performance'][strategy_key]['accuracy']:
                summary_stats[model_name]['optimized_performance'][strategy_key] = {}
                for metric in ['accuracy', 'f1_score', 'roc_auc']:
                    values = aggregated[model_name]['optimized_performance'][strategy_key][metric]
                    summary_stats[model_name]['optimized_performance'][strategy_key][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
            
            # Calibration statistics
            if aggregated[model_name]['calibration_analyses'][strategy_key]['optimal_threshold_f1']:
                summary_stats[model_name]['calibration_stats'][strategy_key] = {}
                for metric in ['optimal_threshold_f1', 'optimal_threshold_acc', 'best_f1', 'best_acc', 'default_f1', 'default_acc']:
                    values = aggregated[model_name]['calibration_analyses'][strategy_key][metric]
                    summary_stats[model_name]['calibration_stats'][strategy_key][metric] = {
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
        
        # Segment-level performance
        print(" Segment-Level Performance:")
        seg_stats = model_stats['segment_performance']
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            stats = seg_stats[metric]
            print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        # Timeseries-level performance
        print("\n Timeseries-Level Performance (Default Threshold):")
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            print(f"\n  {strategy_name}:")
            ts_stats = model_stats['timeseries_performance'][strategy_key]
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                stats = ts_stats[metric]
                print(f"    {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        # Optimized threshold performance
        if USE_THRESHOLD_OPTIMIZATION and 'optimized_performance' in model_stats:
            print("\n Timeseries-Level Performance (Optimized Threshold):")
            for strategy_key, strategy_name in VOTING_STRATEGIES.items():
                if strategy_key in model_stats['optimized_performance']:
                    print(f"\n  {strategy_name} (Optimized):")
                    opt_stats = model_stats['optimized_performance'][strategy_key]
                    for metric in ['accuracy', 'f1_score', 'roc_auc']:
                        if metric in opt_stats:
                            stats = opt_stats[metric]
                            print(f"    {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                                  f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
                    
                    # Show improvement
                    if strategy_key in model_stats['timeseries_performance']:
                        default_f1 = model_stats['timeseries_performance'][strategy_key]['f1_score']['mean']
                        optimized_f1 = opt_stats['f1_score']['mean']
                        improvement = ((optimized_f1 - default_f1) / default_f1) * 100 if default_f1 > 0 else 0
                        print(f"     F1 IMPROVEMENT: +{improvement:.1f}%")
            
            # Calibration summary
            if 'calibration_stats' in model_stats:
                print("\n Threshold Optimization Summary:")
                for strategy_key, strategy_name in VOTING_STRATEGIES.items():
                    if strategy_key in model_stats['calibration_stats']:
                        cal_stats = model_stats['calibration_stats'][strategy_key]
                        opt_thresh_f1 = cal_stats['optimal_threshold_f1']['mean']
                        opt_thresh_acc = cal_stats['optimal_threshold_acc']['mean']
                        print(f"  {strategy_name}:")
                        print(f"    Optimal F1 threshold: {opt_thresh_f1:.3f} ± {cal_stats['optimal_threshold_f1']['std']:.3f}")
                        print(f"    Optimal Acc threshold: {opt_thresh_acc:.3f} ± {cal_stats['optimal_threshold_acc']['std']:.3f}")

def train_and_save_best_pipeline(X, y, groups, best_config):
    """Train and save the best pipeline on all data"""
    print(f"\nTraining best pipeline on full dataset...")
    
    # Get the best model configuration
    model_name = best_config['model']
    
    # Train the best model on all data
    models = get_models()
    model = models[model_name]
    
    # Create pipeline with same configuration as CV
    steps = [('scaler', StandardScaler())]
    if USE_FEATURE_SELECTION:
        steps.append(('selector', SelectKBest(score_func=f_classif, k=N_FEATURES_TO_SELECT)))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps)
    param_grid = get_model_params(model_name)
    
    # Adjust n_jobs for GridSearch and Calibration
    sklearn_n_jobs = max(1, os.cpu_count() // N_PARALLEL_JOBS)
    
    # Apply hyperparameter tuning if available
    if param_grid:
        cv = GroupKFold(n_splits=3)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=sklearn_n_jobs, verbose=0
        )
        grid_search.fit(X, y, groups=groups)
        pipeline = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")
    else:
        pipeline.fit(X, y)
    
    # Apply calibration if enabled
    if USE_PROBABILITY_CALIBRATION:
        print(f"  Applying {CALIBRATION_METHOD} calibration...")
        cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        calibrated_pipeline = CalibratedClassifierCV(
            pipeline, cv=cv_cal, method=CALIBRATION_METHOD, n_jobs=sklearn_n_jobs
        )
        calibrated_pipeline.fit(X, y)
        pipeline = calibrated_pipeline
    
    # Apply SMOTE and retrain if needed
    if USE_SMOTE_AUGMENTATION:
        X_smote, y_smote = apply_smote_augmentation(X, y, SMOTE_RATIO)
        if not USE_PROBABILITY_CALIBRATION:  # Only retrain if not calibrated
            pipeline.fit(X_smote, y_smote)
        print(f"  SMOTE applied: {len(X)} → {len(X_smote)} samples")
    
    # Save the complete pipeline
    pipeline_path = os.path.join(OUTPUT_DIR, 'best_pipeline.joblib')
    dump(pipeline, pipeline_path)
    
    print(f" Best pipeline saved: {pipeline_path}")
    return pipeline

def save_best_pipeline_and_results(best_config, summary_stats):
    """Save only the best pipeline and essential results"""
    print(f"\nSaving best pipeline and results to {OUTPUT_DIR}...")
    
    # Save summary CSV
    csv_data = []
    for model_name, model_stats in summary_stats.items():
        # Timeseries-level rows (default threshold)
        for strategy_key, strategy_name in VOTING_STRATEGIES.items():
            ts_stats = model_stats['timeseries_performance'][strategy_key]
            csv_data.append({
                'Model': model_name,
                'Strategy': strategy_key,
                'Threshold_Type': 'Default',
                'Accuracy_Mean': ts_stats['accuracy']['mean'],
                'Accuracy_Std': ts_stats['accuracy']['std'],
                'F1_Mean': ts_stats['f1_score']['mean'],
                'F1_Std': ts_stats['f1_score']['std'],
                'ROC_AUC_Mean': ts_stats['roc_auc']['mean'],
                'ROC_AUC_Std': ts_stats['roc_auc']['std']
            })
            
            # Add optimized threshold results if available
            if 'optimized_performance' in model_stats and strategy_key in model_stats['optimized_performance']:
                opt_stats = model_stats['optimized_performance'][strategy_key]
                csv_data.append({
                    'Model': model_name,
                    'Strategy': strategy_key,
                    'Threshold_Type': 'Optimized',
                    'Accuracy_Mean': opt_stats['accuracy']['mean'],
                    'Accuracy_Std': opt_stats['accuracy']['std'],
                    'F1_Mean': opt_stats['f1_score']['mean'],
                    'F1_Std': opt_stats['f1_score']['std'],
                    'ROC_AUC_Mean': opt_stats['roc_auc']['mean'],
                    'ROC_AUC_Std': opt_stats['roc_auc']['std']
                })
    
    summary_csv_path = os.path.join(OUTPUT_DIR, 'cv_summary.csv')
    pd.DataFrame(csv_data).to_csv(summary_csv_path, index=False)
    
    # Save best configuration
    best_config_path = os.path.join(OUTPUT_DIR, 'best_configuration.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f" Results saved:")
    print(f"    Summary CSV: {summary_csv_path}")
    print(f"    Best config: {best_config_path}")

def create_visualization(summary_stats):
    """Create comprehensive visualization of CV results"""
    print("Creating visualization...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    model_names = list(summary_stats.keys())
    strategy_keys = list(VOTING_STRATEGIES.keys())
    
    # Create figure with 2x3 layout for cleaner organization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Cross-Validation Results ({N_CV_SPLITS} splits)', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall Performance Comparison (All metrics, all strategies)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    
    # Create grouped bar chart comparing segment vs best timeseries performance
    x = np.arange(len(metrics))
    width = 0.35
    
    # Get best model performance
    best_model = max(model_names, key=lambda m: max(
        summary_stats[m]['timeseries_performance'][s]['roc_auc']['mean'] 
        for s in strategy_keys
    ))
    
    seg_scores = [summary_stats[best_model]['segment_performance'][m]['mean'] for m in metrics]
    best_ts_scores = [max(summary_stats[best_model]['timeseries_performance'][s][m]['mean'] 
                         for s in strategy_keys) for m in metrics]
    
    bars1 = ax1.bar(x - width/2, seg_scores, width, label='Segment Level', alpha=0.8)
    bars2 = ax1.bar(x + width/2, best_ts_scores, width, label='Best Timeseries', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Performance Comparison\n({best_model})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Strategy Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    strategy_names = [VOTING_STRATEGIES[k].split('(')[0].strip() for k in strategy_keys]
    strategy_scores = [summary_stats[best_model]['timeseries_performance'][s]['roc_auc']['mean'] 
                      for s in strategy_keys]
    
    bars = ax2.bar(strategy_names, strategy_scores, alpha=0.8, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('ROC AUC')
    ax2.set_title('Voting Strategy Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, strategy_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Threshold Optimization Impact (if enabled)
    ax3 = fig.add_subplot(gs[0, 2])
    
    if USE_THRESHOLD_OPTIMIZATION and 'optimized_performance' in summary_stats[best_model]:
        default_scores = [summary_stats[best_model]['timeseries_performance'][s]['f1_score']['mean'] 
                         for s in strategy_keys]
        optimized_scores = [summary_stats[best_model]['optimized_performance'][s]['f1_score']['mean'] 
                           for s in strategy_keys if s in summary_stats[best_model]['optimized_performance']]
        
        if len(optimized_scores) == len(strategy_keys):
            x = np.arange(len(strategy_keys))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, default_scores, width, label='Default (0.5)', alpha=0.8)
            bars2 = ax3.bar(x + width/2, optimized_scores, width, label='Optimized', alpha=0.8)
            
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('Threshold Optimization Impact')
            ax3.set_xticks(x)
            ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategy_keys])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add improvement percentages
            for i, (default, optimized) in enumerate(zip(default_scores, optimized_scores)):
                improvement = ((optimized - default) / default) * 100 if default > 0 else 0
                ax3.text(i, max(default, optimized) + 0.02, f'+{improvement:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', color='green' if improvement > 0 else 'red')
    else:
        # Show segment vs timeseries improvement
        improvements = []
        for strategy in strategy_keys:
            seg_auc = summary_stats[best_model]['segment_performance']['roc_auc']['mean']
            ts_auc = summary_stats[best_model]['timeseries_performance'][strategy]['roc_auc']['mean']
            improvement = ((ts_auc - seg_auc) / seg_auc) * 100
            improvements.append(improvement)
        
        bars = ax3.bar(strategy_names, improvements, alpha=0.8, 
                      color=['green' if x > 0 else 'red' for x in improvements])
        ax3.set_ylabel('ROC AUC Improvement (%)')
        ax3.set_title('Timeseries vs Segment\nPerformance Gain')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 4: Performance Distribution Across CV Splits
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create box plot for ROC AUC across CV splits
    best_strategy = max(strategy_keys, 
                       key=lambda s: summary_stats[best_model]['timeseries_performance'][s]['roc_auc']['mean'])
    
    seg_values = summary_stats[best_model]['segment_performance']['roc_auc']['values']
    ts_values = summary_stats[best_model]['timeseries_performance'][best_strategy]['roc_auc']['values']
    
    box_data = [seg_values, ts_values]
    box_labels = ['Segment', f'Timeseries\n({best_strategy.replace("_", " ").title()})']
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax4.set_ylabel('ROC AUC')
    ax4.set_title(f'Performance Distribution\nAcross {N_CV_SPLITS} CV Splits')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Configuration Summary
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create configuration summary
    config_data = []
    config_data.append(['Best Model', best_model])
    config_data.append(['Best Strategy', best_strategy.replace('_', ' ').title()])
    config_data.append(['CV Splits', str(N_CV_SPLITS)])
    config_data.append(['Feature Selection', 'Yes' if USE_FEATURE_SELECTION else 'No'])
    if USE_FEATURE_SELECTION:
        config_data.append(['Features Used', str(N_FEATURES_TO_SELECT)])
    config_data.append(['Calibration', CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else 'None'])
    config_data.append(['SMOTE', f'{SMOTE_RATIO}x' if USE_SMOTE_AUGMENTATION else 'No'])
    config_data.append(['Threshold Opt.', 'Yes' if USE_THRESHOLD_OPTIMIZATION else 'No'])
    
    table = ax5.table(cellText=config_data,
                     colLabels=['Configuration', 'Value'],
                     cellLoc='left', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Configuration Summary', fontweight='bold', pad=20)
    
    # Plot 6: Performance Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create performance summary table
    perf_data = []
    for strategy in strategy_keys:
        strategy_name = strategy.replace('_', ' ').title()
        perf = summary_stats[best_model]['timeseries_performance'][strategy]
        perf_data.append([
            strategy_name,
            f"{perf['accuracy']['mean']:.3f}±{perf['accuracy']['std']:.3f}",
            f"{perf['f1_score']['mean']:.3f}±{perf['f1_score']['std']:.3f}",
            f"{perf['roc_auc']['mean']:.3f}±{perf['roc_auc']['std']:.3f}"
        ])
    
    table = ax6.table(cellText=perf_data,
                     colLabels=['Strategy', 'Accuracy', 'F1 Score', 'ROC AUC'],
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best strategy
    best_strategy_idx = strategy_keys.index(best_strategy) + 1
    for i in range(4):
        table[(best_strategy_idx, i)].set_facecolor('#FFD700')  # Gold
    
    ax6.set_title('Performance Summary\n(Mean ± Std)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'cv_results_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {plot_path}")
    
    plt.show()
    
    return fig

def run_single_feature_experiment(args):
    """Run CV for a single feature count - parallel wrapper"""
    n_features, fold_splits, pbar = args
    
    try:
        # Run CV splits sequentially for this feature count to avoid nested parallelism
        all_cv_results = []
        
        for cv_split_id, fold_split in enumerate(fold_splits):
            # Thread-safe modification of global variable
            with _global_lock:
                original_n_features = globals()['N_FEATURES_TO_SELECT']
                globals()['N_FEATURES_TO_SELECT'] = n_features
            
            try:
                cv_result = run_single_cv_split(cv_split_id, fold_split)
                all_cv_results.append(cv_result)
            finally:
                # Restore original value in thread-safe manner
                with _global_lock:
                    globals()['N_FEATURES_TO_SELECT'] = original_n_features
        
        # Aggregate results for this feature count
        summary_stats = aggregate_cv_results(all_cv_results)
        
        # Find best performance for this feature count
        best_performance = find_best_performance_across_models(summary_stats)
        
        result = {
            'summary_stats': summary_stats,
            'best_performance': best_performance,
            # Don't store all_cv_results to save memory in parallel execution
            'num_cv_splits': len(all_cv_results)
        }
        
        # Clean up memory
        del all_cv_results
        del summary_stats
        
        # Update progress bar
        if pbar:
            metric_value = best_performance.get(FEATURE_RANGE_METRIC, 0.0)
            pbar.set_postfix({
                'Features': n_features,
                f'Best_{FEATURE_RANGE_METRIC.upper()}': f"{metric_value:.3f}"
            })
            pbar.update(1)
        
        return n_features, result
        
    except Exception as e:
        if pbar:
            pbar.set_postfix({
                'Features': n_features,
                'Status': 'ERROR'
            })
            pbar.update(1)
        return n_features, {
            'error': str(e),
            'best_performance': {FEATURE_RANGE_METRIC: 0.0}
        }

def run_feature_selection_experiment():
    """Run cross-validation across different numbers of selected features"""
    print(f"\nFeature Selection Range Experiment")
    print(f"Testing feature counts: {FEATURE_RANGE_VALUES}")
    print(f"Optimization metric: {FEATURE_RANGE_METRIC}")
    print(f"CV folds per feature count: {N_CV_SPLITS}")
    
    # Load data once
    X, y, groups = load_segment_data()
    
    # Create k-fold splits once for all feature experiments
    print(f"Creating {N_CV_SPLITS}-fold CV splits...")
    fold_splits = create_kfold_splits(X, y, groups, n_splits=N_CV_SPLITS)
    
    # Run feature experiments in parallel with progress bar
    parallel_jobs = min(N_PARALLEL_JOBS, len(FEATURE_RANGE_VALUES))
    print(f"\nRunning {len(FEATURE_RANGE_VALUES)} feature experiments using {parallel_jobs} parallel jobs...")
    
    # Create progress bar
    with tqdm(total=len(FEATURE_RANGE_VALUES), desc=" Feature Selection", 
              unit="exp", ncols=100, position=0) as pbar:
        
        # Prepare arguments for parallel feature experiments (include progress bar)
        feature_args = [(n_features, fold_splits, pbar) for n_features in FEATURE_RANGE_VALUES]
        
        with Parallel(n_jobs=parallel_jobs, backend='threading') as parallel:
            results_list = parallel(delayed(run_single_feature_experiment)(args) for args in feature_args)
    
    # Convert results list to dictionary
    feature_experiment_results = dict(results_list)
    
    return feature_experiment_results

def find_best_performance_across_models(summary_stats):
    """Find the best performance across all models and strategies for a given metric"""
    best_score = 0
    best_config = {}
    
    for model_name, model_stats in summary_stats.items():
        # Check default threshold performance
        for strategy_key in VOTING_STRATEGIES.keys():
            if strategy_key in model_stats['timeseries_performance']:
                score = model_stats['timeseries_performance'][strategy_key][FEATURE_RANGE_METRIC]['mean']
                if score > best_score:
                    best_score = score
                    best_config = {
                        'model': model_name,
                        'strategy': strategy_key,
                        'threshold_type': 'default',
                        FEATURE_RANGE_METRIC: score,
                        'accuracy': model_stats['timeseries_performance'][strategy_key]['accuracy']['mean'],
                        'f1_score': model_stats['timeseries_performance'][strategy_key]['f1_score']['mean'],
                        'roc_auc': model_stats['timeseries_performance'][strategy_key]['roc_auc']['mean']
                    }
        
        # Check optimized threshold performance if available
        if 'optimized_performance' in model_stats:
            for strategy_key in model_stats['optimized_performance'].keys():
                if FEATURE_RANGE_METRIC in model_stats['optimized_performance'][strategy_key]:
                    score = model_stats['optimized_performance'][strategy_key][FEATURE_RANGE_METRIC]['mean']
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'model': model_name,
                            'strategy': strategy_key,
                            'threshold_type': 'optimized',
                            FEATURE_RANGE_METRIC: score,
                            'accuracy': model_stats['optimized_performance'][strategy_key]['accuracy']['mean'],
                            'f1_score': model_stats['optimized_performance'][strategy_key]['f1_score']['mean'],
                            'roc_auc': model_stats['optimized_performance'][strategy_key]['roc_auc']['mean']
                        }
                        
                        # Add optimal threshold info if available
                        if ('calibration_stats' in model_stats and 
                            strategy_key in model_stats['calibration_stats'] and
                            f'optimal_threshold_{THRESHOLD_OPTIMIZATION_METRIC}' in model_stats['calibration_stats'][strategy_key]):
                            best_config['optimal_threshold'] = model_stats['calibration_stats'][strategy_key][f'optimal_threshold_{THRESHOLD_OPTIMIZATION_METRIC}']['mean']
    
    return best_config

def create_feature_selection_plot(feature_experiment_results):
    """Create plot showing performance vs number of features"""
    print("Creating feature selection analysis plot...")
    
    # Extract data for plotting
    feature_counts = sorted(feature_experiment_results.keys())
    
    # Prepare data
    metrics_data = {
        'accuracy': [],
        'f1_score': [],
        'roc_auc': [],
        'best_models': [],
        'best_strategies': [],
        'threshold_types': []
    }
    
    for n_features in feature_counts:
        result = feature_experiment_results[n_features]
        if 'error' in result:
            # Handle error cases
            for metric in ['accuracy', 'f1_score', 'roc_auc']:
                metrics_data[metric].append(0.0)
            metrics_data['best_models'].append('Error')
            metrics_data['best_strategies'].append('Error')
            metrics_data['threshold_types'].append('Error')
        else:
            best_perf = result['best_performance']
            metrics_data['accuracy'].append(best_perf['accuracy'])
            metrics_data['f1_score'].append(best_perf['f1_score'])
            metrics_data['roc_auc'].append(best_perf['roc_auc'])
            metrics_data['best_models'].append(best_perf['model'])
            metrics_data['best_strategies'].append(best_perf['strategy'])
            metrics_data['threshold_types'].append(best_perf['threshold_type'])
    
    # Create comprehensive plot
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Feature Selection Analysis: Performance vs Number of Features', fontsize=18, fontweight='bold')
    
    # Plot 1: Main performance metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(feature_counts, metrics_data['accuracy'], 'o-', linewidth=2, markersize=8, label='Accuracy', alpha=0.8)
    ax1.plot(feature_counts, metrics_data['f1_score'], 's-', linewidth=2, markersize=8, label='F1 Score', alpha=0.8)
    ax1.plot(feature_counts, metrics_data['roc_auc'], '^-', linewidth=2, markersize=8, label='ROC AUC', alpha=0.8)
    
    # Highlight the target metric
    target_values = metrics_data[FEATURE_RANGE_METRIC]
    best_idx = np.argmax(target_values)
    best_n_features = feature_counts[best_idx]
    best_score = target_values[best_idx]
    
    ax1.scatter([best_n_features], [best_score], s=200, c='red', marker='*', 
               label=f'Best {FEATURE_RANGE_METRIC.upper()}: {best_n_features} features', zorder=10)
    
    ax1.set_xlabel('Number of Selected Features')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('Performance Metrics vs Feature Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Focus on target metric
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(feature_counts, target_values, alpha=0.7, 
                   color=['red' if i == best_idx else 'skyblue' for i in range(len(feature_counts))])
    
    ax2.set_xlabel('Number of Selected Features')
    ax2.set_ylabel(f'{FEATURE_RANGE_METRIC.upper()} Score')
    ax2.set_title(f'Best {FEATURE_RANGE_METRIC.upper()} by Feature Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, target_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == best_idx else 'normal')
    
    # Plot 3: Best models distribution
    ax3 = fig.add_subplot(gs[0, 2])
    model_counts = {}
    for model in metrics_data['best_models']:
        if model != 'Error':
            model_counts[model] = model_counts.get(model, 0) + 1
    
    if model_counts:
        ax3.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.0f%%', startangle=90)
        ax3.set_title('Best Model Distribution\nAcross Feature Counts')
    
    # Plot 4: Best strategies distribution
    ax4 = fig.add_subplot(gs[1, 0])
    strategy_counts = {}
    for strategy in metrics_data['best_strategies']:
        if strategy != 'Error':
            strategy_name = VOTING_STRATEGIES.get(strategy, strategy)
            short_name = strategy_name.split('(')[0].strip()
            strategy_counts[short_name] = strategy_counts.get(short_name, 0) + 1
    
    if strategy_counts:
        ax4.pie(strategy_counts.values(), labels=strategy_counts.keys(), autopct='%1.0f%%', startangle=90)
        ax4.set_title('Best Strategy Distribution\nAcross Feature Counts')
    
    # Plot 5: Threshold types distribution
    ax5 = fig.add_subplot(gs[1, 1])
    threshold_counts = {}
    for thresh_type in metrics_data['threshold_types']:
        if thresh_type != 'Error':
            threshold_counts[thresh_type.title()] = threshold_counts.get(thresh_type.title(), 0) + 1
    
    if threshold_counts:
        colors = ['lightcoral' if k == 'Optimized' else 'lightblue' for k in threshold_counts.keys()]
        ax5.pie(threshold_counts.values(), labels=threshold_counts.keys(), autopct='%1.0f%%', 
               startangle=90, colors=colors)
        ax5.set_title('Threshold Type Distribution\nAcross Feature Counts')
    
    # Plot 6: Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table for top 5 feature counts
    sorted_indices = np.argsort(target_values)[::-1][:5]  # Top 5
    table_data = []
    
    for i, idx in enumerate(sorted_indices):
        n_feat = feature_counts[idx]
        score = target_values[idx]
        model = metrics_data['best_models'][idx]
        strategy = metrics_data['best_strategies'][idx]
        thresh_type = metrics_data['threshold_types'][idx]
        
        table_data.append([
            f"#{i+1}",
            str(n_feat),
            f"{score:.3f}",
            model[:8] if model != 'Error' else 'Error',  # Truncate long names
            strategy[:8] if strategy != 'Error' else 'Error',
            thresh_type[:8] if thresh_type != 'Error' else 'Error'
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Rank', 'Features', f'{FEATURE_RANGE_METRIC.upper()[:3]}', 'Model', 'Strategy', 'Threshold'],
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Header styling
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best result
    if len(table_data) > 0:
        for i in range(6):
            table[(1, i)].set_facecolor('#FFD700')  # Gold for best result
    
    ax6.set_title('Top 5 Feature Counts\nby Performance', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'feature_selection_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" Feature selection analysis saved to: {plot_path}")
    
    plt.show()
    
    return fig

def save_feature_experiment_results(feature_experiment_results, best_n_features, best_config):
    """Save feature selection experiment results"""
    print(f"Saving feature selection experiment results...")
    
    # Create summary data
    summary_data = []
    
    for n_features, result in feature_experiment_results.items():
        if 'error' in result:
            summary_data.append({
                'n_features': n_features,
                'best_model': 'Error',
                'best_strategy': 'Error',
                'threshold_type': 'Error',
                'accuracy': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'error': result['error']
            })
        else:
            best_perf = result['best_performance']
            summary_data.append({
                'n_features': n_features,
                'best_model': best_perf['model'],
                'best_strategy': best_perf['strategy'],
                'threshold_type': best_perf['threshold_type'],
                'accuracy': best_perf['accuracy'],
                'f1_score': best_perf['f1_score'],
                'roc_auc': best_perf['roc_auc'],
                'optimal_threshold': best_perf.get('optimal_threshold', 'N/A')
            })
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, 'feature_selection_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save best configuration
    best_config_path = os.path.join(OUTPUT_DIR, 'best_configuration.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f" Feature selection results saved:")
    print(f"    Summary: {summary_path}")
    print(f"    Best config: {best_config_path}")
    
    return summary_df

def main():
    """Main cross-validation pipeline"""
    start_time = time()
    
    if TEST_FEATURE_RANGE:
        print(f" Starting Feature Selection Range Experiment...")
        
        # Run feature selection experiment
        feature_experiment_results = run_feature_selection_experiment()
        
        # Create feature selection analysis
        create_feature_selection_plot(feature_experiment_results)
        
        # Find best configuration from feature experiment results
        best_n_features = None
        best_score = 0
        best_config = None
        
        for n_features, result in feature_experiment_results.items():
            if 'error' not in result:
                score = result['best_performance'][FEATURE_RANGE_METRIC]
                if score > best_score:
                    best_score = score
                    best_n_features = n_features
                    best_config = result['best_performance']
        
        print(f"\n Feature selection experiment completed!")
        if best_config:
            print(f"    Best: {best_n_features} features with {FEATURE_RANGE_METRIC.upper()} = {best_score:.4f}")
        
        # Create best configuration for feature experiment
        if best_config:
            feature_best_config = {
                'model': best_config['model'],
                'strategy': best_config['strategy'],
                'threshold_type': best_config['threshold_type'],
                'n_features': best_n_features,
                'use_feature_selection': True,
                'use_calibration': USE_PROBABILITY_CALIBRATION,
                'calibration_method': CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else None,
                'use_smote': USE_SMOTE_AUGMENTATION,
                'smote_ratio': SMOTE_RATIO if USE_SMOTE_AUGMENTATION else None,
                'use_threshold_optimization': USE_THRESHOLD_OPTIMIZATION,
                'threshold_optimization_metric': THRESHOLD_OPTIMIZATION_METRIC if USE_THRESHOLD_OPTIMIZATION else None,
                'f1_score': best_score,
                'accuracy': best_config['accuracy'],
                'roc_auc': best_config['roc_auc']
            }
            if 'optimal_threshold' in best_config:
                feature_best_config['optimal_threshold'] = best_config['optimal_threshold']
        else:
            feature_best_config = {}
        
        # Save feature experiment results
        save_feature_experiment_results(feature_experiment_results, best_n_features, feature_best_config)
        
        # Print feature selection summary
        # Print detailed summary
        if best_config:
            print(f"\n{'='*80}")
            print("FEATURE SELECTION EXPERIMENT SUMMARY")
            print(f"{'='*80}")
            print(f" OPTIMAL FEATURE COUNT: {best_n_features}")
            print(f"   Best {FEATURE_RANGE_METRIC.upper()}: {best_score:.4f}")
            print(f"   Model: {best_config['model']}")
            print(f"   Strategy: {VOTING_STRATEGIES[best_config['strategy']]}")
            print(f"   Threshold: {best_config['threshold_type'].title()}")
            if 'optimal_threshold' in best_config:
                print(f"   Optimal threshold: {best_config['optimal_threshold']:.3f}")
            print(f"   Accuracy: {best_config['accuracy']:.4f}")
            print(f"   F1 Score: {best_config['f1_score']:.4f}")
            print(f"   ROC AUC: {best_config['roc_auc']:.4f}")
            
            print(f"\n RECOMMENDATION:")
            print(f"   Use N_FEATURES_TO_SELECT = {best_n_features} for optimal {FEATURE_RANGE_METRIC} performance")
    
    else:
        print(f"Starting {N_CV_SPLITS}-fold cross-validation pipeline...")
        
        # Load data
        X, y, groups = load_segment_data()
        
        # Create k-fold splits
        print(f"Creating {N_CV_SPLITS}-fold CV splits...")
        fold_splits = create_kfold_splits(X, y, groups, n_splits=N_CV_SPLITS)
        
        # Run cross-validation splits in parallel with progress bar
        parallel_jobs = min(N_PARALLEL_JOBS, N_CV_SPLITS)
        print(f"\nRunning {N_CV_SPLITS} CV folds using {parallel_jobs} parallel jobs...")
        
        # Create progress bar
        with tqdm(total=N_CV_SPLITS, desc=" Cross-Validation", 
                  unit="fold", ncols=100, position=0) as pbar:
            
            # Prepare CV split arguments for parallel execution (include progress bar)
            cv_args = [(cv_split_id, fold_splits[cv_split_id], pbar) 
                       for cv_split_id in range(N_CV_SPLITS)]
            
            with Parallel(n_jobs=parallel_jobs, backend='threading') as parallel:
                all_cv_results = parallel(delayed(run_single_cv_split_parallel)(args) for args in cv_args)
        
        # Aggregate results
        summary_stats = aggregate_cv_results(all_cv_results)
        
        # Print summary
        print_summary_results(summary_stats)
        
        # Find best configuration and train final pipeline
        best_model = None
        best_strategy = None
        best_score = 0
        best_is_optimized = False
        
        for model_name, model_stats in summary_stats.items():
            for strategy_key in VOTING_STRATEGIES.keys():
                # Check default performance
                mean_auc = model_stats['timeseries_performance'][strategy_key]['roc_auc']['mean']
                if mean_auc > best_score:
                    best_score = mean_auc
                    best_model = model_name
                    best_strategy = strategy_key
                    best_is_optimized = False
                    
                # Check optimized performance if available
                if 'optimized_performance' in model_stats and strategy_key in model_stats['optimized_performance']:
                    opt_auc = model_stats['optimized_performance'][strategy_key]['roc_auc']['mean']
                    if opt_auc > best_score:
                        best_score = opt_auc
                        best_model = model_name
                        best_strategy = strategy_key
                        best_is_optimized = True
        
        # Create best configuration
        best_config = {
            'model': best_model,
            'strategy': best_strategy,
            'threshold_type': 'optimized' if best_is_optimized else 'default',
            'n_features': N_FEATURES_TO_SELECT if USE_FEATURE_SELECTION else len(MODEL_FEATURES),
            'use_feature_selection': USE_FEATURE_SELECTION,
            'use_calibration': USE_PROBABILITY_CALIBRATION,
            'calibration_method': CALIBRATION_METHOD if USE_PROBABILITY_CALIBRATION else None,
            'use_smote': USE_SMOTE_AUGMENTATION,
            'smote_ratio': SMOTE_RATIO if USE_SMOTE_AUGMENTATION else None,
            'use_threshold_optimization': USE_THRESHOLD_OPTIMIZATION,
            'threshold_optimization_metric': THRESHOLD_OPTIMIZATION_METRIC if USE_THRESHOLD_OPTIMIZATION else None,
            'roc_auc': best_score
        }
        
        if best_is_optimized and 'calibration_stats' in summary_stats[best_model]:
            cal_stats = summary_stats[best_model]['calibration_stats'][best_strategy]
            best_config['optimal_threshold'] = cal_stats[f'optimal_threshold_{THRESHOLD_OPTIMIZATION_METRIC}']['mean']
        
        # Train and save best pipeline
        train_and_save_best_pipeline(X, y, groups, best_config)
        
        # Save results
        save_best_pipeline_and_results(best_config, summary_stats)
        
        # Create visualization
        create_visualization(summary_stats)
        
        print(f"\n BEST CONFIGURATION:")
        print(f"   Model: {best_config['model']}")
        print(f"   Strategy: {VOTING_STRATEGIES[best_config['strategy']]}")
        print(f"   Threshold: {best_config['threshold_type'].title()}")
        print(f"   Mean ROC AUC: {best_config['roc_auc']:.4f}")
        
        if best_config['threshold_type'] == 'optimized' and 'optimized_performance' in summary_stats[best_config['model']]:
            best_stats = summary_stats[best_config['model']]['optimized_performance'][best_config['strategy']]
            print(f"   Mean Accuracy: {best_stats['accuracy']['mean']:.4f} ± {best_stats['accuracy']['std']:.4f}")
            print(f"   Mean F1 Score: {best_stats['f1_score']['mean']:.4f} ± {best_stats['f1_score']['std']:.4f}")
            if 'optimal_threshold' in best_config:
                print(f"   Optimal Threshold: {best_config['optimal_threshold']:.3f}")
        else:
            best_stats = summary_stats[best_config['model']]['timeseries_performance'][best_config['strategy']]
            print(f"   Mean Accuracy: {best_stats['accuracy']['mean']:.4f} ± {best_stats['accuracy']['std']:.4f}")
            print(f"   Mean F1 Score: {best_stats['f1_score']['mean']:.4f} ± {best_stats['f1_score']['std']:.4f}")
        
        # Summary of threshold optimization benefits
        if USE_THRESHOLD_OPTIMIZATION:
            print(f"\n THRESHOLD OPTIMIZATION SUMMARY:")
            total_improvements = 0
            positive_improvements = 0
            
            for model_name, model_stats in summary_stats.items():
                if 'optimized_performance' in model_stats:
                    for strategy_key in VOTING_STRATEGIES.keys():
                        if strategy_key in model_stats['optimized_performance']:
                            default_f1 = model_stats['timeseries_performance'][strategy_key]['f1_score']['mean']
                            opt_f1 = model_stats['optimized_performance'][strategy_key]['f1_score']['mean']
                            improvement = ((opt_f1 - default_f1) / default_f1) * 100 if default_f1 > 0 else 0
                            total_improvements += improvement
                            if improvement > 0:
                                positive_improvements += 1
            
            if total_improvements > 0:
                avg_improvement = total_improvements / (len(summary_stats) * len(VOTING_STRATEGIES))
                print(f"   Average F1 improvement: +{avg_improvement:.1f}%")
                print(f"   Configurations with positive improvement: {positive_improvements}/{len(summary_stats) * len(VOTING_STRATEGIES)}")
                print(f"   Recommendation: {'USE optimized thresholds' if avg_improvement > 1 else 'Default thresholds sufficient'}")
    
    # Final summary (common to both modes)
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {time() - start_time:.1f}s")
    if TEST_FEATURE_RANGE:
        print(f"Tested {len(FEATURE_RANGE_VALUES)} feature counts across {N_CV_SPLITS} CV folds")
    else:
        print(f"Tested {len(get_models())} models across {N_CV_SPLITS} CV folds")
    print(f"Evaluated {len(VOTING_STRATEGIES)} voting strategies")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()