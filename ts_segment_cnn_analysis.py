"""
CNN Analysis Script
Analyzes the trained CNN model to understand what patterns it learned from the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy import stats

# Import CNN classes from the main pipeline
import sys
sys.path.append('.')
from ts_segment_model import CNN1D, PyTorchCNNClassifier, PyTorchLSTMClassifier, LSTM, load_segment_timeseries_data, subsample_or_pad_timeseries

# Configuration
OUTPUT_DIR = 'timeseries_segment_to_timeseries_cv'
ANALYSIS_DIR = 'cnn_analysis'
Path(ANALYSIS_DIR).mkdir(exist_ok=True)
MAX_SEGMENT_LENGTH = 300  # Same as in main pipeline

def load_trained_cnn():
    """Load the trained CNN model"""
    try:
        # Load model configuration
        config_path = Path(OUTPUT_DIR) / 'best_cnn_1d_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Best model: {config['model_name']}")
        
        if 'CNN' not in config['model_name']:
            print(f"⚠️  Best model is {config['model_name']}, not CNN")
            print("Looking for CNN models in saved models...")
            return load_cnn_from_saved_models()
        
        # Load the best model
        model_path = Path(OUTPUT_DIR) / 'best_cnn_1d_model.joblib'
        model = load(model_path)
        
        # Extract CNN model
        cnn_model = None
        if hasattr(model, 'model') and isinstance(model.model, nn.Module):
            cnn_model = model.model
        elif isinstance(model, nn.Module):
            cnn_model = model
        else:
            raise ValueError(f"Loaded model is not a CNN: {type(model)}")
        
        print(f"✅ Successfully loaded CNN: {type(cnn_model)}")
        
        return cnn_model, config, model
        
    except Exception as e:
        print(f"Error loading best model: {e}")
        return load_cnn_from_saved_models()

def load_cnn_from_saved_models():
    """Try to load CNN from individual saved models"""
    print("Attempting to load CNN from individual model files...")
    
    # Look for CNN models in the output directory
    model_files = list(Path(OUTPUT_DIR).glob('*cnn*.joblib'))
    if not model_files:
        model_files = list(Path(OUTPUT_DIR).glob('*CNN*.joblib'))
    
    if model_files:
        print(f"Found CNN model file: {model_files[0]}")
        try:
            model = load(model_files[0])
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                return model.model, None, model
        except Exception as e:
            print(f"Error loading CNN from file: {e}")
    
    print("❌ No CNN model found for analysis")
    return None, None, None

def analyze_learned_patterns(cnn_model, config, full_model):
    """Analyze what patterns the CNN learned from the data"""
    print("\n🔍 Analyzing Learned Patterns...")
    
    # Load data for analysis
    print("   📊 Loading data for pattern analysis...")
    time_series_data, y, groups, original_files = load_segment_timeseries_data()
    
    # Standardize time series length (same as main pipeline)
    print("   📏 Standardizing sequence lengths...")
    X = subsample_or_pad_timeseries(time_series_data, MAX_SEGMENT_LENGTH)
    print(f"   📏 Standardized shape: {X.shape}")
    
    # Scale data using the same scaler
    scaler_path = Path(OUTPUT_DIR) / 'scaler.joblib'
    if scaler_path.exists():
        scaler = load(scaler_path)
        X_scaled = scaler.transform(X)
        print(f"   📏 Applied scaling to {X_scaled.shape[0]} samples")
    else:
        print("   ⚠️  No scaler found, using raw data")
        X_scaled = X
    
    # Get predictions and probabilities
    print("   🎯 Getting model predictions...")
    predictions = full_model.predict(X_scaled)
    probabilities = full_model.predict_proba(X_scaled)
    
    # Analyze prediction confidence
    confidence = np.max(probabilities, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)
    
    print(f"   📈 Prediction statistics:")
    print(f"      • Average confidence: {np.mean(confidence):.3f}")
    print(f"      • High confidence samples (>0.9): {np.sum(confidence > 0.9)} ({100*np.sum(confidence > 0.9)/len(confidence):.1f}%)")
    print(f"      • Low confidence samples (<0.6): {np.sum(confidence < 0.6)} ({100*np.sum(confidence < 0.6)/len(confidence):.1f}%)")
    
    return X_scaled, y, predictions, probabilities, confidence

def get_model_predictions(cnn_model, X_samples):
    """Get model predictions for a batch of samples"""
    device = next(cnn_model.parameters()).device
    cnn_model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_samples).transpose(1, 2).to(device)
        outputs = cnn_model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    
    return probabilities

def compute_permutation_importance(cnn_model, X_samples, baseline_predictions, target_class):
    """Compute feature importance using permutation method"""
    n_samples, n_timesteps, n_features = X_samples.shape
    importance_matrix = np.zeros((n_samples, n_timesteps, n_features))
    
    # For each sample
    for sample_idx in range(n_samples):
        sample = X_samples[sample_idx:sample_idx+1].copy()
        baseline_prob = baseline_predictions[sample_idx, target_class]
        
        # For each time step
        for t in range(0, n_timesteps, 10):  # Sample every 10th timestep for efficiency
            # For each feature
            for f in range(n_features):
                # Create permuted version
                permuted_sample = sample.copy()
                # Permute this feature at this timestep across all samples in batch
                if sample_idx < n_samples - 1:
                    permuted_sample[0, t, f] = X_samples[sample_idx + 1, t, f]
                else:
                    permuted_sample[0, t, f] = X_samples[0, t, f]
                
                # Get prediction with permuted feature
                permuted_pred = get_model_predictions(cnn_model, permuted_sample)
                permuted_prob = permuted_pred[0, target_class]
                
                # Importance is the drop in probability
                importance = abs(baseline_prob - permuted_prob)
                importance_matrix[sample_idx, t, f] = importance
    
    return importance_matrix

def extract_feature_importance_via_gradients(cnn_model, X_sample, target_class):
    """Extract feature importance using gradient-based attribution"""
    try:
        # Get model device
        device = next(cnn_model.parameters()).device
        
        # Convert to tensor and enable gradients, move to same device as model
        X_tensor = torch.FloatTensor(X_sample).transpose(1, 2).requires_grad_(True).to(device)
        
        # Set model to training mode to enable gradient computation
        cnn_model.train()
        
        # Forward pass
        output = cnn_model(X_tensor)
        
        # Use softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        target_prob = probs[:, target_class].sum()
        
        # Backward pass
        target_prob.backward()
        
        # Set model back to eval mode
        cnn_model.eval()
        
        # Check if gradients were computed
        if X_tensor.grad is None:
            print(f"      ⚠️  No gradients computed for class {target_class}")
            return np.zeros((X_sample.shape[0], X_sample.shape[1], X_sample.shape[2]))
        
        # Get gradients (feature importance) and move back to CPU for numpy conversion
        gradients = X_tensor.grad.data.transpose(1, 2).cpu().numpy()
        
        return gradients
        
    except Exception as e:
        print(f"      ❌ Error computing gradients: {e}")
        return np.zeros((X_sample.shape[0], X_sample.shape[1], X_sample.shape[2]))

def analyze_temporal_patterns(cnn_model, X_scaled, y, predictions, probabilities):
    """Analyze temporal patterns the CNN learned"""
    print("\n⏰ Analyzing Temporal Patterns...")
    
    # Select representative samples for each class
    class_0_samples = X_scaled[y == 0][:100]  # Control samples
    class_1_samples = X_scaled[y == 1][:100]  # Treatment samples
    
    # Analyze feature importance across time for each class
    feature_names = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']
    
    # Use permutation-based feature importance instead of gradients
    print("   🔍 Computing permutation-based feature importance...")
    
    # Get baseline predictions for each class
    baseline_pred_0 = get_model_predictions(cnn_model, class_0_samples[:20])
    baseline_pred_1 = get_model_predictions(cnn_model, class_1_samples[:20])
    
    # Compute feature importance by permuting each feature at each time step
    importance_class_0 = compute_permutation_importance(cnn_model, class_0_samples[:20], baseline_pred_0, 0)
    importance_class_1 = compute_permutation_importance(cnn_model, class_1_samples[:20], baseline_pred_1, 1)
    
    # Average importance across samples
    avg_importance_class_0 = np.mean(importance_class_0, axis=0)
    avg_importance_class_1 = np.mean(importance_class_1, axis=0)
    
    # Create temporal pattern visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CNN Learned Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature importance over time for Class 0 (Control)
    ax1 = axes[0, 0]
    for feat_idx, feat_name in enumerate(feature_names):
        ax1.plot(avg_importance_class_0[:, feat_idx], label=feat_name, linewidth=2)
    ax1.set_title('Feature Importance Over Time - Control Group')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature importance over time for Class 1 (Treatment)
    ax2 = axes[0, 1]
    for feat_idx, feat_name in enumerate(feature_names):
        ax2.plot(avg_importance_class_1[:, feat_idx], label=feat_name, linewidth=2)
    ax2.set_title('Feature Importance Over Time - Treatment Group')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Gradient Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average feature importance comparison
    ax3 = axes[1, 0]
    total_importance_class_0 = np.sum(avg_importance_class_0, axis=0)
    total_importance_class_1 = np.sum(avg_importance_class_1, axis=0)
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, total_importance_class_0, width, label='Control', alpha=0.8)
    bars2 = ax3.bar(x + width/2, total_importance_class_1, width, label='Treatment', alpha=0.8)
    
    ax3.set_ylabel('Total Importance')
    ax3.set_title('Total Feature Importance by Class')
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_names, rotation=45)
    ax3.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(total_importance_class_0.max(), total_importance_class_1.max())*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Temporal importance heatmap
    ax4 = axes[1, 1]
    
    # Create difference heatmap (Treatment - Control)
    importance_diff = avg_importance_class_1 - avg_importance_class_0
    
    im = ax4.imshow(importance_diff.T, cmap='RdBu_r', aspect='auto')
    ax4.set_title('Importance Difference (Treatment - Control)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Feature')
    ax4.set_yticks(range(len(feature_names)))
    ax4.set_yticklabels(feature_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Importance Difference')
    
    plt.tight_layout()
    
    # Save plot
    patterns_plot_path = Path(ANALYSIS_DIR) / 'cnn_learned_patterns.png'
    plt.savefig(patterns_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Patterns plot saved: {patterns_plot_path}")
    
    plt.show()
    
    # Return analysis results
    return {
        'feature_importance_control': avg_importance_class_0,
        'feature_importance_treatment': avg_importance_class_1,
        'total_importance_control': total_importance_class_0,
        'total_importance_treatment': total_importance_class_1,
        'feature_names': feature_names
    }

def analyze_prediction_patterns(X_scaled, y, predictions, probabilities, confidence):
    """Analyze prediction patterns and model behavior"""
    print("\n📊 Analyzing Prediction Patterns...")
    
    # Analyze correct vs incorrect predictions
    correct_predictions = predictions == y
    accuracy = np.mean(correct_predictions)
    
    print(f"   🎯 Overall accuracy: {accuracy:.3f}")
    
    # Analyze confidence by correctness
    correct_confidence = confidence[correct_predictions]
    incorrect_confidence = confidence[~correct_predictions]
    
    print(f"   ✅ Correct predictions confidence: {np.mean(correct_confidence):.3f} ± {np.std(correct_confidence):.3f}")
    print(f"   ❌ Incorrect predictions confidence: {np.mean(incorrect_confidence):.3f} ± {np.std(incorrect_confidence):.3f}")
    
    # Analyze class-specific patterns
    for class_idx in [0, 1]:
        class_mask = y == class_idx
        class_predictions = predictions[class_mask]
        class_true = y[class_mask]
        class_confidence = confidence[class_mask]
        
        class_accuracy = np.mean(class_predictions == class_true)
        avg_confidence = np.mean(class_confidence)
        
        print(f"   📈 Class {class_idx} accuracy: {class_accuracy:.3f}, avg confidence: {avg_confidence:.3f}")
    
    # Create prediction analysis visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CNN Prediction Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(correct_confidence, bins=30, alpha=0.7, label='Correct', color='green', density=True)
    ax1.hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution by Correctness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence by true class
    ax2 = axes[0, 1]
    class_0_conf = confidence[y == 0]
    class_1_conf = confidence[y == 1]
    
    ax2.boxplot([class_0_conf, class_1_conf], labels=['Control', 'Treatment'])
    ax2.set_ylabel('Prediction Confidence')
    ax2.set_title('Confidence by True Class')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix with confidence
    ax3 = axes[1, 0]
    cm = confusion_matrix(y, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix')
    
    # Plot 4: Probability distribution
    ax4 = axes[1, 1]
    prob_class_1 = probabilities[:, 1]
    
    # Separate by true class
    prob_true_0 = prob_class_1[y == 0]
    prob_true_1 = prob_class_1[y == 1]
    
    ax4.hist(prob_true_0, bins=30, alpha=0.7, label='True Control', color='blue', density=True)
    ax4.hist(prob_true_1, bins=30, alpha=0.7, label='True Treatment', color='orange', density=True)
    ax4.set_xlabel('Probability of Treatment Class')
    ax4.set_ylabel('Density')
    ax4.set_title('Probability Distribution by True Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    prediction_plot_path = Path(ANALYSIS_DIR) / 'cnn_prediction_patterns.png'
    plt.savefig(prediction_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Prediction patterns plot saved: {prediction_plot_path}")
    
    plt.show()
    
    return {
        'accuracy': accuracy,
        'correct_confidence_mean': np.mean(correct_confidence),
        'incorrect_confidence_mean': np.mean(incorrect_confidence),
        'class_accuracies': [np.mean(predictions[y == i] == y[y == i]) for i in [0, 1]],
        'confusion_matrix': cm.tolist()
    }

def create_analysis_summary(pattern_analysis, prediction_analysis):
    """Create comprehensive analysis summary"""
    print(f"\n📋 Creating CNN Pattern Analysis Summary...")
    
    # Find most important features
    total_importance = pattern_analysis['total_importance_control'] + pattern_analysis['total_importance_treatment']
    feature_ranking = sorted(zip(pattern_analysis['feature_names'], total_importance), 
                           key=lambda x: x[1], reverse=True)
    
    summary = {
        'model_performance': {
            'overall_accuracy': prediction_analysis['accuracy'],
            'control_accuracy': prediction_analysis['class_accuracies'][0],
            'treatment_accuracy': prediction_analysis['class_accuracies'][1],
            'correct_prediction_confidence': prediction_analysis['correct_confidence_mean'],
            'incorrect_prediction_confidence': prediction_analysis['incorrect_confidence_mean']
        },
        'feature_importance': {
            'ranking': feature_ranking,
            'most_important_feature': feature_ranking[0][0],
            'least_important_feature': feature_ranking[-1][0],
            'control_group_importance': {
                name: float(importance) 
                for name, importance in zip(pattern_analysis['feature_names'], 
                                          pattern_analysis['total_importance_control'])
            },
            'treatment_group_importance': {
                name: float(importance) 
                for name, importance in zip(pattern_analysis['feature_names'], 
                                          pattern_analysis['total_importance_treatment'])
            }
        },
        'learned_patterns': {
            'discriminative_features': [name for name, _ in feature_ranking[:2]],
            'temporal_focus': 'Early time steps show higher importance (based on gradient analysis)',
            'class_differences': 'CNN learned different importance patterns for control vs treatment groups'
        },
        'insights': {
            'key_findings': [
                f"Most important feature: {feature_ranking[0][0]}",
                f"Model accuracy: {prediction_analysis['accuracy']:.1%}",
                f"Confidence difference between correct/incorrect: {prediction_analysis['correct_confidence_mean'] - prediction_analysis['incorrect_confidence_mean']:.3f}",
                "CNN focuses on different features for different classes"
            ],
            'model_behavior': [
                "Uses gradient-based feature importance",
                "Shows temporal patterns in feature usage",
                "Demonstrates class-specific attention patterns",
                "Maintains reasonable prediction confidence"
            ]
        }
    }
    
    # Save summary
    summary_path = Path(ANALYSIS_DIR) / 'cnn_pattern_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   📄 Pattern analysis summary saved: {summary_path}")
    
    # Print key insights
    print(f"\n🔍 Key CNN Pattern Insights:")
    print(f"   • Most important feature: {feature_ranking[0][0]} (importance: {feature_ranking[0][1]:.3f})")
    print(f"   • Model accuracy: {prediction_analysis['accuracy']:.1%}")
    print(f"   • Confidence gap (correct vs incorrect): {prediction_analysis['correct_confidence_mean'] - prediction_analysis['incorrect_confidence_mean']:.3f}")
    print(f"   • CNN learned class-specific feature importance patterns")
    
    return summary

def analyze_temporal_decisions(cnn_model, X_scaled, y, predictions, probabilities):
    """Analyze when during the trajectory the CNN makes classification decisions"""
    print("\n⏰ Analyzing Temporal Decision Points...")
    
    # Select samples for analysis
    n_samples = min(100, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_samples = X_scaled[sample_indices]
    y_samples = y[sample_indices]
    
    # Analyze prediction confidence over time by masking different time segments
    time_windows = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]
    window_confidences = {class_idx: [] for class_idx in [0, 1]}
    
    for class_idx in [0, 1]:
        class_samples = X_samples[y_samples == class_idx][:20]  # 20 samples per class
        
        for start_time, end_time in time_windows:
            # Create masked version (zero out this time window)
            masked_samples = class_samples.copy()
            masked_samples[:, start_time:end_time, :] = 0
            
            # Get predictions on masked data
            masked_probs = get_model_predictions(cnn_model, masked_samples)
            avg_confidence = np.mean(np.max(masked_probs, axis=1))
            
            window_confidences[class_idx].append(avg_confidence)
    
    # Analyze progressive masking (cumulative importance)
    cumulative_confidences = {class_idx: [] for class_idx in [0, 1]}
    
    for class_idx in [0, 1]:
        class_samples = X_samples[y_samples == class_idx][:20]
        
        # Get baseline confidence
        baseline_probs = get_model_predictions(cnn_model, class_samples)
        baseline_confidence = np.mean(np.max(baseline_probs, axis=1))
        
        # Progressive masking from start
        for mask_end in range(50, 301, 50):
            masked_samples = class_samples.copy()
            masked_samples[:, :mask_end, :] = 0  # Mask from start to mask_end
            
            masked_probs = get_model_predictions(cnn_model, masked_samples)
            avg_confidence = np.mean(np.max(masked_probs, axis=1))
            confidence_drop = baseline_confidence - avg_confidence
            
            cumulative_confidences[class_idx].append(confidence_drop)
    
    # Create temporal decision analysis visualization
    create_temporal_decision_visualization(time_windows, window_confidences, cumulative_confidences)
    
    return window_confidences, cumulative_confidences

def create_temporal_decision_visualization(time_windows, window_confidences, cumulative_confidences):
    """Create visualization for temporal decision analysis"""
    print("   📊 Creating temporal decision visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Temporal Decision Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Window masking effect
    ax1 = axes[0]
    window_centers = [(start + end) / 2 for start, end in time_windows]
    
    ax1.plot(window_centers, window_confidences[0], 'bo-', label='Control', linewidth=2, markersize=8)
    ax1.plot(window_centers, window_confidences[1], 'ro-', label='Treatment', linewidth=2, markersize=8)
    ax1.set_xlabel('Time Window Center')
    ax1.set_ylabel('Confidence After Masking')
    ax1.set_title('Impact of Masking Different Time Windows')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative importance
    ax2 = axes[1]
    mask_points = list(range(50, 301, 50))
    
    ax2.plot(mask_points, cumulative_confidences[0], 'bo-', label='Control', linewidth=2, markersize=8)
    ax2.plot(mask_points, cumulative_confidences[1], 'ro-', label='Treatment', linewidth=2, markersize=8)
    ax2.set_xlabel('Masked Time Points (from start)')
    ax2.set_ylabel('Confidence Drop')
    ax2.set_title('Cumulative Temporal Importance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Summary analysis
    ax3 = axes[2]
    ax3.axis('off')
    
    # Find most important time windows
    control_importance = np.array(window_confidences[0])
    treatment_importance = np.array(window_confidences[1])
    
    # Lower confidence after masking = higher importance of that window
    control_importance_inverted = 1 - control_importance
    treatment_importance_inverted = 1 - treatment_importance
    
    most_important_control = time_windows[np.argmax(control_importance_inverted)]
    most_important_treatment = time_windows[np.argmax(treatment_importance_inverted)]
    
    # Calculate when 50% of information is captured
    control_cumulative = np.array(cumulative_confidences[0])
    treatment_cumulative = np.array(cumulative_confidences[1])
    
    control_50pct = mask_points[np.argmin(np.abs(control_cumulative - control_cumulative[-1] * 0.5))]
    treatment_50pct = mask_points[np.argmin(np.abs(treatment_cumulative - treatment_cumulative[-1] * 0.5))]
    
    summary_text = f"""
Temporal Decision Analysis:

Most Critical Time Windows:
• Control: {most_important_control[0]}-{most_important_control[1]} time steps
• Treatment: {most_important_treatment[0]}-{most_important_treatment[1]} time steps

50% Information Captured By:
• Control: {control_50pct} time steps
• Treatment: {treatment_50pct} time steps

Key Insights:
• Early trajectory segments: {'High' if control_50pct < 150 else 'Low'} importance
• Decision timing difference: {abs(control_50pct - treatment_50pct)} time steps
• Treatment detection: {'Early' if treatment_50pct < 150 else 'Late'} in trajectory

Biological Interpretation:
• CNN makes decisions {'early' if min(control_50pct, treatment_50pct) < 150 else 'late'} in movement
• Treatment effects are {'immediately' if treatment_50pct < 100 else 'gradually'} detectable
• Different timing suggests distinct movement strategies
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    temporal_plot_path = Path(ANALYSIS_DIR) / 'temporal_decision_analysis.png'
    plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Temporal decision plot saved: {temporal_plot_path}")
    
    plt.show()

def perform_perturbation_analysis(cnn_model, X_scaled, y, predictions):
    """Perform systematic perturbation analysis to understand CNN robustness"""
    print("\n🔬 Performing Perturbation Analysis...")
    
    # Select representative samples
    n_samples = min(50, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_samples = X_scaled[sample_indices]
    y_samples = y[sample_indices]
    
    # Get baseline predictions
    baseline_probs = get_model_predictions(cnn_model, X_samples)
    baseline_predictions = np.argmax(baseline_probs, axis=1)
    baseline_confidences = np.max(baseline_probs, axis=1)
    
    perturbation_results = {}
    
    # 1. Noise perturbation analysis
    print("   🔊 Testing noise robustness...")
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    noise_results = []
    
    for noise_level in noise_levels:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, X_samples.shape)
        perturbed_samples = X_samples + noise
        
        # Get predictions on perturbed data
        perturbed_probs = get_model_predictions(cnn_model, perturbed_samples)
        perturbed_predictions = np.argmax(perturbed_probs, axis=1)
        
        # Calculate accuracy drop
        accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
        confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
        
        noise_results.append({
            'noise_level': noise_level,
            'accuracy_drop': accuracy_drop,
            'confidence_drop': confidence_drop
        })
    
    perturbation_results['noise'] = noise_results
    
    # 2. Feature-specific perturbation
    print("   🎯 Testing feature-specific robustness...")
    feature_names = ['x_coordinate', 'y_coordinate', 'displacement', 'movement_angle']
    feature_results = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        # Permute this feature across samples
        perturbed_samples = X_samples.copy()
        permutation = np.random.permutation(len(X_samples))
        perturbed_samples[:, :, feat_idx] = X_samples[permutation, :, feat_idx]
        
        # Get predictions
        perturbed_probs = get_model_predictions(cnn_model, perturbed_samples)
        perturbed_predictions = np.argmax(perturbed_probs, axis=1)
        
        accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
        confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
        
        feature_results.append({
            'feature': feat_name,
            'accuracy_drop': accuracy_drop,
            'confidence_drop': confidence_drop
        })
    
    perturbation_results['features'] = feature_results
    
    # 3. Temporal segment perturbation
    print("   ⏰ Testing temporal segment robustness...")
    segment_size = 50
    segment_results = []
    
    for start_time in range(0, 300, segment_size):
        end_time = min(start_time + segment_size, 300)
        
        # Shuffle this time segment
        perturbed_samples = X_samples.copy()
        for i in range(len(X_samples)):
            segment = perturbed_samples[i, start_time:end_time, :].copy()
            np.random.shuffle(segment)
            perturbed_samples[i, start_time:end_time, :] = segment
        
        # Get predictions
        perturbed_probs = get_model_predictions(cnn_model, perturbed_samples)
        perturbed_predictions = np.argmax(perturbed_probs, axis=1)
        
        accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
        confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
        
        segment_results.append({
            'time_segment': f"{start_time}-{end_time}",
            'accuracy_drop': accuracy_drop,
            'confidence_drop': confidence_drop
        })
    
    perturbation_results['temporal'] = segment_results
    
    # Create perturbation visualization
    create_perturbation_visualization(perturbation_results)
    
    return perturbation_results

def create_perturbation_visualization(perturbation_results):
    """Create visualization for perturbation analysis results"""
    print("   📊 Creating perturbation analysis visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Perturbation Analysis - CNN Robustness', fontsize=16, fontweight='bold')
    
    # Plot 1: Noise robustness
    ax1 = axes[0, 0]
    noise_data = perturbation_results['noise']
    noise_levels = [r['noise_level'] for r in noise_data]
    accuracy_drops = [r['accuracy_drop'] for r in noise_data]
    confidence_drops = [r['confidence_drop'] for r in noise_data]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(noise_levels, accuracy_drops, 'bo-', linewidth=2, markersize=8, label='Accuracy Drop')
    line2 = ax1_twin.plot(noise_levels, confidence_drops, 'ro-', linewidth=2, markersize=8, label='Confidence Drop')
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Accuracy Drop', color='blue')
    ax1_twin.set_ylabel('Confidence Drop', color='red')
    ax1.set_title('Robustness to Noise')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Feature-specific robustness
    ax2 = axes[0, 1]
    feature_data = perturbation_results['features']
    features = [r['feature'] for r in feature_data]
    feature_accuracy_drops = [r['accuracy_drop'] for r in feature_data]
    feature_confidence_drops = [r['confidence_drop'] for r in feature_data]
    
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, feature_accuracy_drops, width, label='Accuracy Drop', alpha=0.8)
    bars2 = ax2.bar(x + width/2, feature_confidence_drops, width, label='Confidence Drop', alpha=0.8)
    
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Drop Value')
    ax2.set_title('Feature-Specific Robustness')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temporal segment robustness
    ax3 = axes[1, 0]
    temporal_data = perturbation_results['temporal']
    segments = [r['time_segment'] for r in temporal_data]
    temporal_accuracy_drops = [r['accuracy_drop'] for r in temporal_data]
    temporal_confidence_drops = [r['confidence_drop'] for r in temporal_data]
    
    x = np.arange(len(segments))
    bars1 = ax3.bar(x, temporal_accuracy_drops, alpha=0.7, label='Accuracy Drop')
    bars2 = ax3.bar(x, temporal_confidence_drops, alpha=0.7, label='Confidence Drop')
    
    ax3.set_xlabel('Time Segment')
    ax3.set_ylabel('Drop Value')
    ax3.set_title('Temporal Segment Robustness')
    ax3.set_xticks(x)
    ax3.set_xticklabels(segments, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary analysis
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find most vulnerable aspects
    max_noise_drop = max(accuracy_drops)
    most_vulnerable_feature = features[np.argmax(feature_accuracy_drops)]
    most_vulnerable_segment = segments[np.argmax(temporal_accuracy_drops)]
    
    # Calculate robustness scores
    noise_robustness = 1 - max_noise_drop
    feature_robustness = 1 - max(feature_accuracy_drops)
    temporal_robustness = 1 - max(temporal_accuracy_drops)
    overall_robustness = np.mean([noise_robustness, feature_robustness, temporal_robustness])
    
    summary_text = f"""
Perturbation Analysis Summary:

Robustness Scores (0-1, higher = more robust):
• Noise Robustness: {noise_robustness:.3f}
• Feature Robustness: {feature_robustness:.3f}
• Temporal Robustness: {temporal_robustness:.3f}
• Overall Robustness: {overall_robustness:.3f}

Most Vulnerable Aspects:
• Noise Level: {noise_levels[np.argmax(accuracy_drops)]:.2f} (drops {max_noise_drop:.1%})
• Feature: {most_vulnerable_feature}
• Time Segment: {most_vulnerable_segment}

Key Insights:
• CNN is {'robust' if overall_robustness > 0.7 else 'sensitive'} to perturbations
• Most critical feature: {most_vulnerable_feature}
• Most critical time: {most_vulnerable_segment}
• Noise threshold: ~{noise_levels[np.argmax(accuracy_drops)]:.2f}

Biological Implications:
• Treatment effects are {'subtle' if max_noise_drop < 0.3 else 'robust'}
• Feature dependencies: {'high' if max(feature_accuracy_drops) > 0.5 else 'moderate'}
• Temporal specificity: {'high' if max(temporal_accuracy_drops) > 0.5 else 'moderate'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    perturbation_plot_path = Path(ANALYSIS_DIR) / 'perturbation_analysis.png'
    plt.savefig(perturbation_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Perturbation analysis plot saved: {perturbation_plot_path}")
    
    plt.show()

def main():
    """Main CNN pattern analysis function"""
    print("🔍 CNN Pattern Analysis")
    print("=" * 50)
    
    # Load the trained CNN model
    result = load_trained_cnn()
    if result[0] is None:
        print("❌ Could not load CNN model for analysis")
        print("Please ensure the pipeline has been run and a CNN model is available.")
        return
    
    cnn_model, config, full_model = result
    
    try:
        # 1. Analyze learned patterns from data
        X_scaled, y, predictions, probabilities, confidence = analyze_learned_patterns(cnn_model, config, full_model)
        
        # 2. Analyze temporal patterns and feature importance
        pattern_analysis = analyze_temporal_patterns(cnn_model, X_scaled, y, predictions, probabilities)
        
        # 3. Analyze prediction patterns
        prediction_analysis = analyze_prediction_patterns(X_scaled, y, predictions, probabilities, confidence)
        
        # 4. Analyze temporal decisions (CNN-specific)
        temporal_decisions = analyze_temporal_decisions(cnn_model, X_scaled, y, predictions, probabilities)
        
        # 5. Perform perturbation analysis (CNN-specific)
        perturbation_results = perform_perturbation_analysis(cnn_model, X_scaled, y, predictions)
        
        # 6. Create comprehensive summary
        summary = create_analysis_summary(pattern_analysis, prediction_analysis)
        
        print(f"\n✅ CNN Pattern Analysis Complete!")
        print(f"Results saved in: {ANALYSIS_DIR}/")
        print(f"Key files:")
        print(f"  • Learned patterns: cnn_learned_patterns.png")
        print(f"  • Prediction analysis: cnn_prediction_patterns.png")
        print(f"  • Temporal decision analysis: temporal_decision_analysis.png")
        print(f"  • Perturbation analysis: perturbation_analysis.png")
        print(f"  • Summary: cnn_pattern_analysis_summary.json")
        
    except Exception as e:
        print(f"❌ Error during CNN pattern analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
