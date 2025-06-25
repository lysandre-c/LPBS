"""
Random Forest Analysis Script
Analyzes the trained Random Forest model to understand its decision-making process
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For tree visualization
from sklearn.tree import plot_tree, export_text

# Configuration
OUTPUT_DIR = 'timeseries_segment_to_timeseries_cv'
ANALYSIS_DIR = 'random_forest_analysis'
Path(ANALYSIS_DIR).mkdir(exist_ok=True)

# Define FlattenTransformer to match the one used in the main script
class FlattenTransformer:
    """Transformer to flatten time series data for non-time-series models"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(X.shape[0], -1)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

# Import data loading functions
import sys
sys.path.append('.')
from ts_segment_model import load_segment_timeseries_data, subsample_or_pad_timeseries

def load_trained_model():
    """Load the trained model and check if it's Random Forest"""
    try:
        # Load model configuration
        config_path = Path(OUTPUT_DIR) / 'best_random_forest_ts_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Best model: {config['model_name']}")
        
        if 'Random_Forest' not in config['model_name']:
            print(f"⚠️  Best model is {config['model_name']}, not Random Forest")
            print("This analysis is specific to Random Forest models.")
            
            # Try to load a Random Forest model from CV results
            return load_rf_from_cv_results()
        
        # Load the best model
        model_path = Path(OUTPUT_DIR) / 'best_random_forest_ts_model.joblib'
        model = load(model_path)
        
        # Extract Random Forest from pipeline if needed
        rf_model = None
        
        # Try different ways to extract the Random Forest
        if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
            rf_model = model.named_steps['rf']
        elif hasattr(model, 'steps'):
            # Find Random Forest in pipeline steps
            for step_name, step_model in model.steps:
                if 'rf' in step_name.lower() or 'forest' in step_name.lower():
                    rf_model = step_model
                    break
        elif hasattr(model, '_final_estimator'):
            rf_model = model._final_estimator
        elif str(type(model)).find('RandomForest') != -1:
            rf_model = model
        else:
            # Try to access the last step of the pipeline
            if hasattr(model, 'steps') and len(model.steps) > 0:
                rf_model = model.steps[-1][1]  # Get the last step's model
        
        if rf_model is None:
            raise ValueError("Could not find Random Forest in the loaded model")
        
        # Verify it's actually a Random Forest
        if not hasattr(rf_model, 'estimators_') or not hasattr(rf_model, 'feature_importances_'):
            raise ValueError(f"Extracted model is not a Random Forest: {type(rf_model)}")
        
        print(f"✅ Successfully extracted Random Forest: {type(rf_model)}")
        print(f"   Number of trees: {len(rf_model.estimators_)}")
        print(f"   Number of features: {len(rf_model.feature_importances_)}")
        
        return rf_model, config, model
        
    except Exception as e:
        print(f"Error loading best model: {e}")
        return load_rf_from_cv_results()

def load_rf_from_cv_results():
    """Try to load Random Forest from detailed CV results"""
    print("Attempting to load Random Forest from CV results...")
    
    # This would require training a Random Forest specifically for analysis
    # For now, we'll create a simple demonstration
    return None, None, None

def analyze_feature_importance(rf_model, feature_names, config):
    """Analyze and visualize feature importance"""
    print("\n🔍 Analyzing Feature Importance...")
    
    # Get feature importance
    importance = rf_model.feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Random Forest Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Top 20 features
    ax1 = axes[0, 0]
    top_20 = importance_df.head(20)
    bars = ax1.barh(range(len(top_20)), top_20['importance'], color='skyblue')
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['feature'], fontsize=8)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 20 Most Important Features')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=7)
    
    # Plot 2: Feature importance distribution
    ax2 = axes[0, 1]
    ax2.hist(importance, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Feature Importance')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Importance Distribution')
    ax2.axvline(np.mean(importance), color='red', linestyle='--', 
                label=f'Mean: {np.mean(importance):.4f}')
    ax2.axvline(np.median(importance), color='orange', linestyle='--', 
                label=f'Median: {np.median(importance):.4f}')
    ax2.legend()
    
    # Plot 3: Cumulative importance
    ax3 = axes[1, 0]
    cumulative_importance = np.cumsum(importance_df['importance'].values)
    ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             color='purple', linewidth=2)
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('Cumulative Importance')
    ax3.set_title('Cumulative Feature Importance')
    ax3.grid(True, alpha=0.3)
    
    # Add lines for 80%, 90%, 95% importance
    for pct in [0.8, 0.9, 0.95]:
        idx = np.where(cumulative_importance >= pct)[0]
        if len(idx) > 0:
            ax3.axhline(y=pct, color='red', linestyle='--', alpha=0.5)
            ax3.axvline(x=idx[0] + 1, color='red', linestyle='--', alpha=0.5)
            ax3.text(idx[0] + 1, pct + 0.02, f'{pct*100}%\n({idx[0] + 1} features)', 
                    ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Feature type analysis (if we can infer feature types)
    ax4 = axes[1, 1]
    
    # Try to categorize features based on naming patterns
    feature_categories = categorize_features(feature_names)
    category_importance = {}
    
    for category, features in feature_categories.items():
        category_importance[category] = sum(importance_df[importance_df['feature'].isin(features)]['importance'])
    
    if category_importance:
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax4.pie(importances, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax4.set_title('Feature Importance by Category')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax4.text(0.5, 0.5, 'Feature categorization\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Feature Categories')
    
    plt.tight_layout()
    
    # Save plot
    importance_plot_path = Path(ANALYSIS_DIR) / 'feature_importance_analysis.png'
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Feature importance plot saved: {importance_plot_path}")
    
    plt.show()
    
    # Save detailed importance data
    importance_csv_path = Path(ANALYSIS_DIR) / 'feature_importance.csv'
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"   📄 Feature importance data saved: {importance_csv_path}")
    
    return importance_df

def categorize_features(feature_names):
    """Categorize features based on naming patterns"""
    categories = {
        'X Coordinate': [],
        'Y Coordinate': [],
        'Speed': [],
        'Turning Angle': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        if 'x_coordinate' in feature_lower or 'x_coord' in feature_lower:
            categories['X Coordinate'].append(feature)
        elif 'y_coordinate' in feature_lower or 'y_coord' in feature_lower:
            categories['Y Coordinate'].append(feature)
        elif 'speed' in feature_lower:
            categories['Speed'].append(feature)
        elif 'turning_angle' in feature_lower or 'angle' in feature_lower:
            categories['Turning Angle'].append(feature)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def analyze_tree_structure(rf_model, max_trees_to_analyze=5):
    """Analyze the structure of individual trees"""
    print(f"\n🌳 Analyzing Tree Structure (first {max_trees_to_analyze} trees)...")
    
    tree_stats = []
    
    for i, tree in enumerate(rf_model.estimators_[:max_trees_to_analyze]):
        tree_info = {
            'tree_id': i,
            'n_nodes': tree.tree_.node_count,
            'max_depth': tree.tree_.max_depth,
            'n_leaves': tree.tree_.n_leaves,
            'n_features_used': len(np.unique(tree.tree_.feature[tree.tree_.feature >= 0]))
        }
        tree_stats.append(tree_info)
    
    tree_df = pd.DataFrame(tree_stats)
    print("\nTree Structure Statistics:")
    print(tree_df.to_string(index=False))
    
    # Overall forest statistics
    all_tree_stats = {
        'total_trees': len(rf_model.estimators_),
        'avg_depth': np.mean([tree.tree_.max_depth for tree in rf_model.estimators_]),
        'avg_nodes': np.mean([tree.tree_.node_count for tree in rf_model.estimators_]),
        'avg_leaves': np.mean([tree.tree_.n_leaves for tree in rf_model.estimators_])
    }
    
    print(f"\nForest Statistics:")
    print(f"  Total Trees: {all_tree_stats['total_trees']}")
    print(f"  Average Depth: {all_tree_stats['avg_depth']:.2f}")
    print(f"  Average Nodes: {all_tree_stats['avg_nodes']:.2f}")
    print(f"  Average Leaves: {all_tree_stats['avg_leaves']:.2f}")
    
    return tree_df, all_tree_stats

def visualize_sample_trees(rf_model, feature_names, max_trees=3, max_depth_plot=3):
    """Visualize a few sample decision trees"""
    print(f"\n🎨 Visualizing {max_trees} Sample Decision Trees...")
    
    # For trees with many features, we need to be more careful
    try:
        fig, axes = plt.subplots(1, max_trees, figsize=(20, 8))
        if max_trees == 1:
            axes = [axes]
        
        fig.suptitle('Sample Decision Trees from Random Forest', fontsize=16, fontweight='bold')
        
        for i in range(max_trees):
            tree = rf_model.estimators_[i]
            
            # Plot tree with very limited depth and no feature names for readability
            plot_tree(tree, 
                      feature_names=None,  # Don't use feature names to avoid index errors
                      class_names=['Class 0', 'Class 1'],
                      filled=True,
                      rounded=True,
                      fontsize=6,
                      max_depth=max_depth_plot,
                      ax=axes[i])
            
            axes[i].set_title(f'Tree {i+1}\n(Depth: {tree.tree_.max_depth}, Nodes: {tree.tree_.node_count})')
        
        plt.tight_layout()
        
        # Save plot
        trees_plot_path = Path(ANALYSIS_DIR) / 'sample_decision_trees.png'
        plt.savefig(trees_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📊 Decision trees plot saved: {trees_plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️  Could not visualize decision trees due to complexity: {e}")
        print(f"   💡 Trees are very deep (avg depth: {np.mean([tree.tree_.max_depth for tree in rf_model.estimators_]):.1f}) with many features ({len(feature_names)})")
        
        # Create a simplified tree structure visualization instead
        create_tree_structure_summary(rf_model)

def create_tree_structure_summary(rf_model):
    """Create a summary visualization of tree structures"""
    print(f"   📊 Creating tree structure summary instead...")
    
    # Collect tree statistics
    depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    nodes = [tree.tree_.node_count for tree in rf_model.estimators_]
    leaves = [tree.tree_.n_leaves for tree in rf_model.estimators_]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Random Forest Tree Structure Summary', fontsize=14, fontweight='bold')
    
    # Plot depth distribution
    axes[0].hist(depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Tree Depth')
    axes[0].set_ylabel('Number of Trees')
    axes[0].set_title(f'Tree Depth Distribution\n(Mean: {np.mean(depths):.1f})')
    axes[0].grid(True, alpha=0.3)
    
    # Plot node count distribution
    axes[1].hist(nodes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Number of Nodes')
    axes[1].set_ylabel('Number of Trees')
    axes[1].set_title(f'Node Count Distribution\n(Mean: {np.mean(nodes):.0f})')
    axes[1].grid(True, alpha=0.3)
    
    # Plot leaves distribution
    axes[2].hist(leaves, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[2].set_xlabel('Number of Leaves')
    axes[2].set_ylabel('Number of Trees')
    axes[2].set_title(f'Leaf Count Distribution\n(Mean: {np.mean(leaves):.0f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    structure_plot_path = Path(ANALYSIS_DIR) / 'tree_structure_summary.png'
    plt.savefig(structure_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Tree structure summary saved: {structure_plot_path}")
    
    plt.show()

def analyze_decision_paths(rf_model, feature_names, sample_size=100):
    """Analyze common decision paths in the forest"""
    print(f"\n🛤️  Analyzing Decision Paths...")
    
    # Get the first tree for detailed analysis
    first_tree = rf_model.estimators_[0]
    
    # Export tree as text (without feature names to avoid issues)
    tree_text = export_text(first_tree, 
                           feature_names=None,  # Use feature indices instead of names
                           max_depth=5)
    
    print("\nFirst Tree Structure (max depth 5):")
    print(tree_text[:2000] + "..." if len(tree_text) > 2000 else tree_text)
    
    # Save full tree text
    tree_text_path = Path(ANALYSIS_DIR) / 'first_tree_structure.txt'
    with open(tree_text_path, 'w') as f:
        f.write(tree_text)
    print(f"   📄 Full tree structure saved: {tree_text_path}")

def create_feature_correlation_analysis(importance_df, original_features):
    """Analyze correlation between original time series features"""
    print(f"\n🔗 Analyzing Original Feature Patterns...")
    
    # Since Random Forest works on flattened time series, let's analyze patterns
    # in the most important features
    
    top_features = importance_df.head(50)['feature'].tolist()
    
    # Try to extract time step information from feature names
    feature_patterns = analyze_feature_patterns(top_features, original_features)
    
    if feature_patterns:
        # Create visualization of feature patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time Series Feature Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Plot importance by time step (if we can extract time steps)
        plot_importance_by_timestep(feature_patterns, importance_df, axes)
        
        plt.tight_layout()
        
        pattern_plot_path = Path(ANALYSIS_DIR) / 'feature_patterns.png'
        plt.savefig(pattern_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📊 Feature patterns plot saved: {pattern_plot_path}")
        
        plt.show()

def analyze_feature_patterns(top_features, original_features):
    """Analyze patterns in the most important features"""
    patterns = {
        'time_steps': {},
        'feature_types': {feat: [] for feat in original_features}
    }
    
    for feature in top_features:
        # Try to extract time step and feature type
        # Assuming flattened features are named like: feature_timestep
        parts = feature.split('_')
        if len(parts) >= 2:
            try:
                time_step = int(parts[-1])
                feature_type = '_'.join(parts[:-1])
                
                if time_step not in patterns['time_steps']:
                    patterns['time_steps'][time_step] = []
                patterns['time_steps'][time_step].append(feature)
                
                if feature_type in patterns['feature_types']:
                    patterns['feature_types'][feature_type].append(time_step)
            except ValueError:
                continue
    
    return patterns

def plot_importance_by_timestep(patterns, importance_df, axes):
    """Plot feature importance patterns by time step"""
    
    # Plot 1: Importance by time step
    ax1 = axes[0, 0]
    time_steps = sorted(patterns['time_steps'].keys())
    step_importances = []
    
    for step in time_steps:
        step_features = patterns['time_steps'][step]
        step_importance = importance_df[importance_df['feature'].isin(step_features)]['importance'].sum()
        step_importances.append(step_importance)
    
    if time_steps and step_importances:
        ax1.plot(time_steps, step_importances, 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Importance')
        ax1.set_title('Feature Importance by Time Step')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No time step\npatterns found', ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Importance by feature type
    ax2 = axes[0, 1]
    feature_type_importance = {}
    
    for feat_type, time_steps in patterns['feature_types'].items():
        if time_steps:
            # Get all features of this type
            type_features = [f for f in importance_df['feature'] if feat_type in f]
            type_importance = importance_df[importance_df['feature'].isin(type_features)]['importance'].sum()
            feature_type_importance[feat_type] = type_importance
    
    if feature_type_importance:
        types = list(feature_type_importance.keys())
        importances = list(feature_type_importance.values())
        
        bars = ax2.bar(types, importances, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(types)])
        ax2.set_ylabel('Total Importance')
        ax2.set_title('Feature Importance by Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No feature type\npatterns found', ha='center', va='center', transform=ax2.transAxes)
    
    # Plots 3 & 4: Additional analysis placeholders
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Additional analysis\ncan be added here', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Future Analysis')
    
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Model interpretation\ninsights', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Interpretation Summary')

def create_analysis_summary(rf_model, importance_df, config):
    """Create a comprehensive analysis summary"""
    print(f"\n📋 Creating Analysis Summary...")
    
    summary = {
        'model_info': {
            'model_type': 'Random Forest',
            'n_estimators': rf_model.n_estimators,
            'max_depth': rf_model.max_depth,
            'min_samples_split': rf_model.min_samples_split,
            'min_samples_leaf': rf_model.min_samples_leaf,
            'max_features': rf_model.max_features,
            'random_state': rf_model.random_state
        },
        'feature_analysis': {
            'total_features': len(importance_df),
            'top_feature': importance_df.iloc[0]['feature'],
            'top_importance': importance_df.iloc[0]['importance'],
            'features_80_pct': len(importance_df[importance_df['importance'].cumsum() <= 0.8]),
            'mean_importance': importance_df['importance'].mean(),
            'std_importance': importance_df['importance'].std()
        },
        'interpretation': {
            'most_important_feature_type': 'Analysis needed',
            'temporal_patterns': 'Analysis needed',
            'decision_complexity': 'Medium to High (Random Forest)',
            'interpretability': 'Moderate (ensemble method)'
        }
    }
    
    # Save summary
    summary_path = Path(ANALYSIS_DIR) / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📄 Analysis summary saved: {summary_path}")
    
    # Print key insights
    print(f"\n🔍 Key Insights:")
    print(f"   • Model uses {rf_model.n_estimators} decision trees")
    print(f"   • Most important feature: {summary['feature_analysis']['top_feature']}")
    print(f"   • Top feature importance: {summary['feature_analysis']['top_importance']:.4f}")
    print(f"   • Features for 80% importance: {summary['feature_analysis']['features_80_pct']}")
    print(f"   • Average feature importance: {summary['feature_analysis']['mean_importance']:.6f}")
    
    return summary

def analyze_temporal_decisions_rf(rf_model, X_scaled, y, feature_names, max_length):
    """Analyze when during the trajectory the Random Forest makes classification decisions"""
    print("\n⏰ Analyzing Temporal Decision Points (Random Forest)...")
    
    # Select samples for analysis
    n_samples = min(100, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_samples = X_scaled[sample_indices]
    y_samples = y[sample_indices]
    
    # Get baseline predictions
    baseline_probs = rf_model.predict_proba(X_samples)
    baseline_confidences = np.max(baseline_probs, axis=1)
    
    # Analyze prediction confidence over time by masking different time segments
    time_windows = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]
    window_confidences = {class_idx: [] for class_idx in [0, 1]}
    
    for class_idx in [0, 1]:
        class_samples = X_samples[y_samples == class_idx][:20]  # 20 samples per class
        
        for start_time, end_time in time_windows:
            # Create masked version (zero out this time window)
            masked_samples = class_samples.copy()
            
            # For Random Forest, we need to mask the flattened features
            n_features = 4  # x_coordinate, y_coordinate, speed, turning_angle
            for t in range(start_time, end_time):
                for f in range(n_features):
                    feature_idx = t * n_features + f
                    if feature_idx < masked_samples.shape[1]:
                        masked_samples[:, feature_idx] = 0
            
            # Get predictions on masked data
            try:
                masked_probs = rf_model.predict_proba(masked_samples)
                avg_confidence = np.mean(np.max(masked_probs, axis=1))
                window_confidences[class_idx].append(avg_confidence)
            except:
                window_confidences[class_idx].append(0.5)  # Default if prediction fails
    
    # Analyze progressive masking (cumulative importance)
    cumulative_confidences = {class_idx: [] for class_idx in [0, 1]}
    
    for class_idx in [0, 1]:
        class_samples = X_samples[y_samples == class_idx][:20]
        
        # Get baseline confidence
        baseline_probs = rf_model.predict_proba(class_samples)
        baseline_confidence = np.mean(np.max(baseline_probs, axis=1))
        
        # Progressive masking from start
        for mask_end in range(50, 301, 50):
            masked_samples = class_samples.copy()
            
            # Mask from start to mask_end
            n_features = 4
            for t in range(0, mask_end):
                for f in range(n_features):
                    feature_idx = t * n_features + f
                    if feature_idx < masked_samples.shape[1]:
                        masked_samples[:, feature_idx] = 0
            
            try:
                masked_probs = rf_model.predict_proba(masked_samples)
                avg_confidence = np.mean(np.max(masked_probs, axis=1))
                confidence_drop = baseline_confidence - avg_confidence
                cumulative_confidences[class_idx].append(confidence_drop)
            except:
                cumulative_confidences[class_idx].append(0.0)
    
    # Create temporal decision analysis visualization
    create_temporal_decision_visualization_rf(time_windows, window_confidences, cumulative_confidences)
    
    return window_confidences, cumulative_confidences

def create_temporal_decision_visualization_rf(time_windows, window_confidences, cumulative_confidences):
    """Create visualization for Random Forest temporal decision analysis"""
    print("   📊 Creating temporal decision visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Random Forest Temporal Decision Analysis', fontsize=16, fontweight='bold')
    
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
    
    if len(control_cumulative) > 0 and control_cumulative[-1] > 0:
        control_50pct = mask_points[np.argmin(np.abs(control_cumulative - control_cumulative[-1] * 0.5))]
    else:
        control_50pct = 150
        
    if len(treatment_cumulative) > 0 and treatment_cumulative[-1] > 0:
        treatment_50pct = mask_points[np.argmin(np.abs(treatment_cumulative - treatment_cumulative[-1] * 0.5))]
    else:
        treatment_50pct = 150
    
    summary_text = f"""
Random Forest Temporal Analysis:

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

Random Forest Characteristics:
• Uses ensemble of decision trees
• Considers all time points simultaneously
• Feature importance distributed across time
• Less temporal locality than CNN
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    temporal_plot_path = Path(ANALYSIS_DIR) / 'rf_temporal_decision_analysis.png'
    plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Random Forest temporal decision plot saved: {temporal_plot_path}")
    
    plt.show()

def perform_perturbation_analysis_rf(rf_model, X_scaled, y, feature_names):
    """Perform systematic perturbation analysis to understand Random Forest robustness"""
    print("\n🔬 Performing Perturbation Analysis (Random Forest)...")
    
    # Select representative samples
    n_samples = min(50, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_samples = X_scaled[sample_indices]
    y_samples = y[sample_indices]
    
    # Get baseline predictions
    baseline_probs = rf_model.predict_proba(X_samples)
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
        try:
            perturbed_probs = rf_model.predict_proba(perturbed_samples)
            perturbed_predictions = np.argmax(perturbed_probs, axis=1)
            
            # Calculate accuracy drop
            accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
            confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
            
            noise_results.append({
                'noise_level': noise_level,
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop
            })
        except:
            noise_results.append({
                'noise_level': noise_level,
                'accuracy_drop': 0.0,
                'confidence_drop': 0.0
            })
    
    perturbation_results['noise'] = noise_results
    
    # 2. Feature-specific perturbation (by time step groups)
    print("   🎯 Testing feature-specific robustness...")
    feature_types = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']
    feature_results = []
    
    for feat_idx, feat_name in enumerate(feature_types):
        # Permute this feature type across all time steps
        perturbed_samples = X_samples.copy()
        
        # For Random Forest with flattened features, permute all instances of this feature
        n_features = 4
        max_length = X_samples.shape[1] // n_features
        
        for t in range(max_length):
            feature_col_idx = t * n_features + feat_idx
            if feature_col_idx < X_samples.shape[1]:
                permutation = np.random.permutation(len(X_samples))
                perturbed_samples[:, feature_col_idx] = X_samples[permutation, feature_col_idx]
        
        # Get predictions
        try:
            perturbed_probs = rf_model.predict_proba(perturbed_samples)
            perturbed_predictions = np.argmax(perturbed_probs, axis=1)
            
            accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
            confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
            
            feature_results.append({
                'feature': feat_name,
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop
            })
        except:
            feature_results.append({
                'feature': feat_name,
                'accuracy_drop': 0.0,
                'confidence_drop': 0.0
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
        n_features = 4
        
        for t in range(start_time, end_time):
            for f in range(n_features):
                feature_idx = t * n_features + f
                if feature_idx < X_samples.shape[1]:
                    # Shuffle this feature across samples
                    permutation = np.random.permutation(len(X_samples))
                    perturbed_samples[:, feature_idx] = X_samples[permutation, feature_idx]
        
        # Get predictions
        try:
            perturbed_probs = rf_model.predict_proba(perturbed_samples)
            perturbed_predictions = np.argmax(perturbed_probs, axis=1)
            
            accuracy_drop = np.mean(baseline_predictions != perturbed_predictions)
            confidence_drop = np.mean(baseline_confidences) - np.mean(np.max(perturbed_probs, axis=1))
            
            segment_results.append({
                'time_segment': f"{start_time}-{end_time}",
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop
            })
        except:
            segment_results.append({
                'time_segment': f"{start_time}-{end_time}",
                'accuracy_drop': 0.0,
                'confidence_drop': 0.0
            })
    
    perturbation_results['temporal'] = segment_results
    
    # Create perturbation visualization
    create_perturbation_visualization_rf(perturbation_results)
    
    return perturbation_results

def create_perturbation_visualization_rf(perturbation_results):
    """Create visualization for Random Forest perturbation analysis results"""
    print("   📊 Creating perturbation analysis visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Random Forest Perturbation Analysis - Model Robustness', fontsize=16, fontweight='bold')
    
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
    max_noise_drop = max(accuracy_drops) if accuracy_drops else 0
    most_vulnerable_feature = features[np.argmax(feature_accuracy_drops)] if feature_accuracy_drops else 'None'
    most_vulnerable_segment = segments[np.argmax(temporal_accuracy_drops)] if temporal_accuracy_drops else 'None'
    
    # Calculate robustness scores
    noise_robustness = 1 - max_noise_drop
    feature_robustness = 1 - max(feature_accuracy_drops) if feature_accuracy_drops else 1
    temporal_robustness = 1 - max(temporal_accuracy_drops) if temporal_accuracy_drops else 1
    overall_robustness = np.mean([noise_robustness, feature_robustness, temporal_robustness])
    
    summary_text = f"""
Random Forest Perturbation Summary:

Robustness Scores (0-1, higher = more robust):
• Noise Robustness: {noise_robustness:.3f}
• Feature Robustness: {feature_robustness:.3f}
• Temporal Robustness: {temporal_robustness:.3f}
• Overall Robustness: {overall_robustness:.3f}

Most Vulnerable Aspects:
• Noise Level: {noise_levels[np.argmax(accuracy_drops)] if accuracy_drops else 'N/A':.2f}
• Feature: {most_vulnerable_feature}
• Time Segment: {most_vulnerable_segment}

Random Forest Characteristics:
• Ensemble method: Generally robust
• Feature averaging reduces overfitting
• Less sensitive to individual features
• Handles noise well through voting

Biological Implications:
• Treatment effects are {'robust' if overall_robustness > 0.7 else 'subtle'}
• Feature dependencies: {'distributed' if max(feature_accuracy_drops) < 0.3 else 'concentrated'}
• Temporal patterns: {'global' if max(temporal_accuracy_drops) < 0.3 else 'localized'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    perturbation_plot_path = Path(ANALYSIS_DIR) / 'rf_perturbation_analysis.png'
    plt.savefig(perturbation_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   📊 Random Forest perturbation analysis plot saved: {perturbation_plot_path}")
    
    plt.show()

def main():
    """Main analysis function"""
    print("🔍 Random Forest Model Analysis")
    print("=" * 50)
    
    # Load the trained model
    result = load_trained_model()
    if result[0] is None:
        print("❌ Could not load Random Forest model for analysis")
        print("Please ensure the pipeline has been run and a Random Forest model is available.")
        return
    
    rf_model, config, full_model = result
    
    # Get feature names (for flattened time series)
    max_length = config.get('max_segment_length', 300)
    n_features = config.get('n_features', 4)
    original_features = config.get('time_series_features', ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle'])
    
    # Create feature names for flattened time series
    feature_names = []
    for i in range(max_length):
        for feat in original_features:
            feature_names.append(f"{feat}_{i}")
    
    print(f"Analyzing Random Forest with {len(feature_names)} features")
    print(f"Original time series features: {original_features}")
    print(f"Max segment length: {max_length}")
    
    try:
        # Load and prepare data for temporal and perturbation analyses
        print("\n📊 Loading time series data for advanced analyses...")
        time_series_data, y, groups, original_files = load_segment_timeseries_data()
        
        # Standardize time series length
        X_timeseries = subsample_or_pad_timeseries(time_series_data, max_length)
        print(f"Standardized time series shape: {X_timeseries.shape}")
        
        # Scale data using the same scaler
        scaler_path = Path('timeseries_segment_to_timeseries_cv') / 'scaler.joblib'
        if scaler_path.exists():
            scaler = load(scaler_path)
            X_timeseries_scaled = scaler.transform(X_timeseries)
            print(f"Applied scaling to {X_timeseries_scaled.shape[0]} samples")
        else:
            print("⚠️  No scaler found, using raw data")
            X_timeseries_scaled = X_timeseries
        
        # Flatten for Random Forest
        flattener = FlattenTransformer()
        X_scaled = flattener.transform(X_timeseries_scaled)
        print(f"Flattened data shape for Random Forest: {X_scaled.shape}")
        
        # 1. Feature Importance Analysis
        importance_df = analyze_feature_importance(rf_model, feature_names, config)
        
        # 2. Tree Structure Analysis
        tree_df, forest_stats = analyze_tree_structure(rf_model)
        
        # 3. Visualize Sample Trees
        visualize_sample_trees(rf_model, feature_names)
        
        # 4. Decision Path Analysis
        analyze_decision_paths(rf_model, feature_names)
        
        # 5. Feature Pattern Analysis
        create_feature_correlation_analysis(importance_df, original_features)
        
        # 6. Create Summary
        summary = create_analysis_summary(rf_model, importance_df, config)
        
        # 7. Temporal Decision Analysis (Random Forest specific)
        analyze_temporal_decisions_rf(rf_model, X_scaled, y, feature_names, max_length)
        
        # 8. Perturbation Analysis (Random Forest specific)
        perform_perturbation_analysis_rf(rf_model, X_scaled, y, feature_names)
        
        print(f"\n✅ Random Forest Analysis Complete!")
        print(f"Results saved in: {ANALYSIS_DIR}/")
        print(f"Key files:")
        print(f"  • Feature importance: feature_importance.csv")
        print(f"  • Visualizations: *.png files")
        print(f"  • Tree structure: first_tree_structure.txt")
        print(f"  • Temporal analysis: rf_temporal_decision_analysis.png")
        print(f"  • Perturbation analysis: rf_perturbation_analysis.png")
        print(f"  • Summary: analysis_summary.json")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
