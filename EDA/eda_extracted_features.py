import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import argparse
import gc

def setup_plotting_style():
    """Set up the style for all plots"""
    plt.style.use('ggplot')
    sns.set_theme(font_scale=1.2)
    sns.set_style("whitegrid")
    # Increase the maximum figures warning threshold
    plt.rcParams['figure.max_open_warning'] = 50

def load_data(data_type):
    """
    Load the data based on the data type.
    
    Args:
        data_type (str): Either 'full' or 'segments'
        
    Returns:
        tuple: (DataFrame, plot directory path)
    """
    if data_type == 'full':
        data_file = 'feature_data/full_features.csv'
        plot_dir = 'EDA/full_data/eda_plots'
    else:
        data_file = 'feature_data/segments_features.csv'
        plot_dir = 'EDA/segments_data/eda_plots'
    
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Loading {data_type} data from {data_file}...")
    df = pd.read_csv(data_file)
    return df, plot_dir

def print_data_overview(df):
    """Print basic information about the dataset"""
    print("\n=== Data Overview ===")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Labels distribution: {df['label'].value_counts().to_dict()}")
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.any() > 0 else "No missing values found")

def get_numerical_features(df):
    """Get list of numerical features excluding metadata columns"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return [f for f in numerical_features if f not in ['label', 'death_index']]

def create_distribution_plots(df, numerical_features, plot_dir):
    """Create histograms for each numerical feature, separated by label"""
    print(f"Creating distribution plots for {len(numerical_features)} features...")
    # Process features in smaller batches to avoid memory issues
    batch_size = 10
    for i in range(0, len(numerical_features), batch_size):
        batch_features = numerical_features[i:i+batch_size]
        print(f"  Processing features {i+1}-{i+len(batch_features)} of {len(numerical_features)}")
        
        for feature in batch_features:
            plt.figure(figsize=(12, 6))
            
            for label_value in sorted(df['label'].unique()):
                subset = df[df['label'] == label_value]
                sns.histplot(subset[feature], kde=True, label=f'Label {label_value}', 
                            alpha=0.6, color=plt.cm.tab10(label_value))
            
            plt.title(f'Distribution of {feature} by Label')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{feature}_by_label.png')
            plt.close()
        
        # Force garbage collection after each batch
        plt.close('all')
        gc.collect()

def create_boxplots(df, numerical_features, plot_dir):
    """Create boxplots to compare distributions by label"""
    print(f"Creating boxplots for {len(numerical_features)} features...")
    # Process features in smaller batches to avoid memory issues
    batch_size = 10
    for i in range(0, len(numerical_features), batch_size):
        batch_features = numerical_features[i:i+batch_size]
        print(f"  Processing features {i+1}-{i+len(batch_features)} of {len(numerical_features)}")
        
        for feature in batch_features:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='label', y=feature, data=df, hue='label', legend=False)
            plt.title(f'Boxplot of {feature} by Label')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{feature}_boxplot.png')
            plt.close(fig)
            
        # Force garbage collection after each batch
        plt.close('all')
        gc.collect()

def analyze_feature_importance(df, numerical_features, plot_dir):
    """Analyze which features show the most significant differences between labels"""
    feature_significance = []
    
    for feature in numerical_features:
        groups = [df[df['label'] == label][feature].values for label in sorted(df['label'].unique())]
        statistic, p_value = stats.f_oneway(*groups)
        
        feature_significance.append({
            'feature': feature,
            'statistic': statistic,
            'p_value': p_value
        })
    
    significance_df = pd.DataFrame(feature_significance)
    significance_df = significance_df.sort_values('p_value')
    
    print("\n=== Feature Significance (ANOVA) ===")
    print("Features ranked by significance in separating labels:")
    print(significance_df)
    
    top_features = significance_df.head(10)['feature'].tolist()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[top_features + ['label']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/top_features_correlation.png')
    plt.close()
    
    # Force garbage collection
    plt.close('all')
    gc.collect()
    
    return top_features

def create_pairplots(df, features, plot_dir):
    """Create pairplots for the most significant features"""
    # Ensure we don't try to create a pairplot with too many features
    if len(features) > 5:
        print("  Limiting pairplot to top 5 features to avoid memory issues")
        features = features[:5]
        
    plt.figure(figsize=(15, 15))
    subset = df[features + ['label']].sample(min(500, len(df)))
    g = sns.pairplot(subset, hue='label', corner=True)
    g.fig.suptitle('Pairplot of Most Significant Features', y=1.02)
    plt.savefig(f'{plot_dir}/top_features_pairplot.png')
    plt.close(g.fig)
    plt.close('all')
    gc.collect()

def create_correlation_matrix(df, numerical_features, plot_dir):
    """Create correlation matrix heatmap for all features"""
    # If there are too many features, limit the correlation matrix
    if len(numerical_features) > 30:
        print("  Limiting correlation matrix to 30 features to avoid memory issues")
        numerical_features = numerical_features[:30]
        
    plt.figure(figsize=(20, 16))
    corr_matrix = df[numerical_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of All Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/correlation_matrix.png')
    plt.close()
    plt.close('all')
    gc.collect()

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Exploratory data analysis for worm movement data')
    parser.add_argument('--data_type', type=str, choices=['full', 'segments'], default='full',
                        help='Type of data to analyze: full or segments')
    args = parser.parse_args()
    
    # Close any existing plots to start clean
    plt.close('all')
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load data
    df, plot_dir = load_data(args.data_type)
    
    # Print data overview
    print_data_overview(df)
    
    # Get numerical features
    numerical_features = get_numerical_features(df)
    
    # Create various plots and analyses
    print("\n=== Analyzing Feature Distributions by Label ===")
    create_distribution_plots(df, numerical_features, plot_dir)
    
    print("Creating boxplots...")
    create_boxplots(df, numerical_features, plot_dir)
    
    print("Analyzing feature importance...")
    top_features = analyze_feature_importance(df, numerical_features, plot_dir)
    
    print("Creating pairplots for top features...")
    create_pairplots(df, top_features, plot_dir)
    
    print("Creating correlation matrix...")
    create_correlation_matrix(df, numerical_features, plot_dir)
    
    # Final cleanup
    plt.close('all')
    gc.collect()
    
    print(f"\nEDA completed for {args.data_type} data. Results saved to the '{plot_dir}' directory.")

if __name__ == "__main__":
    main() 