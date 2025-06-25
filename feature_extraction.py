import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
from sklearn.preprocessing import RobustScaler

import pywt
from tqdm import tqdm

# Suppress PyWavelets warnings for short signals
warnings.filterwarnings('ignore', category=UserWarning, module='pywt')

def calculate_enhanced_physics_features(df):
    """Calculate physics-inspired movement features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0:
        features.update({
            'kinetic_energy_proxy': 0,
            'movement_efficiency': 0,
            'speed_persistence': 0,
            'exploration_ratio': 0
        })
        return features
    
    # Kinetic energy proxy (mass assumed constant)
    features['kinetic_energy_proxy'] = np.mean(0.5 * speed**2)
    
    # Movement efficiency
    max_speed = speed.max() if len(speed) > 0 else 1
    features['movement_efficiency'] = speed.sum() / (max_speed * len(speed) + 1e-6)
    
    # Speed persistence (consistency)
    features['speed_persistence'] = speed.mean() / (speed.std() + 1e-6)
    
    # Exploration ratio (turning vs speed)
    if len(turning) > 0:
        turning_freq = np.sum(np.abs(turning) > 30) / len(turning)  # Significant turns
        features['exploration_ratio'] = turning_freq / (speed.mean() + 1e-6)
    else:
        features['exploration_ratio'] = 0
    
    return features

def calculate_enhanced_statistical_features(df):
    """Calculate advanced statistical features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0:
        features.update({
            'speed_skewness': 0,
            'speed_kurtosis': 0,
            'speed_iqr': 0,
            'speed_cv': 0
        })
        return features
    
    # Skewness and kurtosis for speed
    features['speed_skewness'] = skew(speed) if len(speed) > 2 else 0
    features['speed_kurtosis'] = kurtosis(speed) if len(speed) > 2 else 0
    
    # Interquartile range
    features['speed_iqr'] = np.percentile(speed, 75) - np.percentile(speed, 25)
    
    # Coefficient of variation
    features['speed_cv'] = speed.std() / (speed.mean() + 1e-6)
    
    return features

def calculate_enhanced_temporal_features(df):
    """Calculate temporal pattern features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    
    if len(speed) < 3:
        features.update({
            'speed_trend': 0,
            'speed_acceleration_mean': 0,
            'speed_acceleration_std': 0
        })
        return features
    
    # Speed trend (linear fit slope)
    try:
        features['speed_trend'] = np.polyfit(range(len(speed)), speed, 1)[0]
    except:
        features['speed_trend'] = 0
    
    # Speed acceleration (rate of change)
    speed_accel = np.diff(speed)
    features['speed_acceleration_mean'] = np.mean(speed_accel)
    features['speed_acceleration_std'] = np.std(speed_accel)
    
    return features

def calculate_enhanced_frequency_features(df):
    """Calculate frequency domain features using FFT."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    def get_dominant_frequency(x):
        if len(x) < 8:  # Need minimum points for meaningful FFT
            return 0
        try:
            fft_vals = np.abs(fft(x))[1:len(x)//2]
            if len(fft_vals) == 0:
                return 0
            return np.argmax(fft_vals)
        except:
            return 0
    
    def get_spectral_centroid(x):
        if len(x) < 8:
            return 0
        try:
            fft_vals = np.abs(fft(x))[1:len(x)//2]
            if len(fft_vals) == 0:
                return 0
            freqs = np.arange(1, len(fft_vals) + 1)
            return np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)
        except:
            return 0
    
    # Frequency analysis of speed
    features['speed_dominant_freq'] = get_dominant_frequency(speed)
    features['speed_spectral_centroid'] = get_spectral_centroid(speed)
    
    # Frequency analysis of turning
    features['turning_dominant_freq'] = get_dominant_frequency(turning)
    features['turning_spectral_centroid'] = get_spectral_centroid(turning)
    
    return features

def calculate_behavioral_state_features(df):
    """Calculate features representing different behavioral states."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0 or len(turning) == 0:
        features.update({
            'activity_level': 0,
            'high_activity_fraction': 0,
            'low_activity_fraction': 0,
            'mixed_activity_fraction': 0
        })
        return features
    
    # Calculate activity level combining speed and turning
    turning_activity = np.abs(turning)
    activity_scores = []
    
    min_len = min(len(speed), len(turning_activity))
    if min_len > 0:
        speed_norm = speed[:min_len] / (speed.max() + 1e-6)
        turning_norm = turning_activity[:min_len] / (turning_activity.max() + 1e-6)
        activity_scores = speed_norm * 0.7 + turning_norm * 0.3  # Weight speed more
    
    if len(activity_scores) > 0:
        features['activity_level'] = np.mean(activity_scores)
        
        # Activity state fractions
        high_threshold = np.percentile(activity_scores, 75)
        low_threshold = np.percentile(activity_scores, 25)
        
        features['high_activity_fraction'] = np.mean(activity_scores > high_threshold)
        features['low_activity_fraction'] = np.mean(activity_scores < low_threshold)
        features['mixed_activity_fraction'] = np.mean((activity_scores >= low_threshold) & 
                                                    (activity_scores <= high_threshold))
    else:
        features['activity_level'] = 0
        features['high_activity_fraction'] = 0
        features['low_activity_fraction'] = 0
        features['mixed_activity_fraction'] = 0
    
    return features


def calculate_basic_features(df, speed_threshold=0.05):
    """Calculate basic movement features.
    
    Args:
        df: DataFrame containing movement data
        speed_threshold: Minimum speed to consider as movement
    """
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    x = df['x'].dropna()
    y = df['y'].dropna()
    
    if len(speed) == 0 or len(x) == 0 or len(y) == 0:
        features.update({
            'mean_speed': 0,
            'std_speed': 0,
            'max_speed': 0,
            'min_speed': 0,
            'total_distance': 0,
            'time_paused': 0,
            'fraction_paused': 0
        })
        return features
    
    # Speed features
    features['mean_speed'] = speed.mean()
    features['std_speed'] = speed.std()
    features['max_speed'] = speed.max()
    features['min_speed'] = speed.min()
    
    # Total distance traveled
    dx = x.diff()
    dy = y.diff()
    distances = np.sqrt(dx**2 + dy**2)
    features['total_distance'] = distances.sum()
    
    # Time paused (speed below threshold)
    features['time_paused'] = (speed < speed_threshold).sum()
    features['fraction_paused'] = features['time_paused'] / len(speed)
    
    return features

def calculate_turning_features(df, window_size=30, height_threshold=90):
    """Calculate features related to turning behavior.
    
    Args:
        df: DataFrame containing movement data
        window_size: Size of windows for calculating meandering ratio
        height_threshold: Minimum height for peaks in turning angle
    """
    features = {}
    
    # Drop NaN values before calculating features
    turning_angle = df['turning_angle'].dropna()
    x = df['x'].dropna()
    y = df['y'].dropna()
    
    if len(turning_angle) == 0 or len(x) == 0 or len(y) == 0:
        features.update({
            'mean_turning_angle': 0,
            'std_turning_angle': 0,
            'max_turning_angle': 0,
            'turning_frequency': 0,
            'mean_meandering_ratio': 0,
            'std_meandering_ratio': 0,
            'min_meandering_ratio': 0,
            'max_meandering_ratio': 0,
            'median_meandering_ratio': 0,
            'fraction_efficient_movement': 0,
        })
        return features
    
    # Turning angle statistics
    features['mean_turning_angle'] = turning_angle.mean()
    features['std_turning_angle'] = turning_angle.std()
    features['max_turning_angle'] = turning_angle.abs().max()
    
    # Turning frequency (peaks in turning angle)
    peaks, _ = find_peaks(turning_angle.abs(), height=height_threshold)
    features['turning_frequency'] = len(peaks) / len(turning_angle)
    
    # Calculate meandering ratio in non-overlapping windows
    meandering_ratios = []
    for i in range(0, len(x), window_size):
        if i + window_size > len(x):
            continue
            
        window_x = x.iloc[i:i+window_size]
        window_y = y.iloc[i:i+window_size]
        
        net_displacement = np.sqrt((window_x.iloc[-1] - window_x.iloc[0])**2 + 
                                 (window_y.iloc[-1] - window_y.iloc[0])**2)
        
        dx = window_x.diff()
        dy = window_y.diff()
        total_distance = np.sqrt(dx**2 + dy**2).sum()
        
        if total_distance > 0:
            meandering_ratios.append(net_displacement / total_distance)
    
    if meandering_ratios:
        features['mean_meandering_ratio'] = np.mean(meandering_ratios)
        features['std_meandering_ratio'] = np.std(meandering_ratios)
        features['min_meandering_ratio'] = np.min(meandering_ratios)
        features['max_meandering_ratio'] = np.max(meandering_ratios)
        features['median_meandering_ratio'] = np.median(meandering_ratios)
        features['fraction_efficient_movement'] = np.mean(np.array(meandering_ratios) > 0.5)
    else:
        features['mean_meandering_ratio'] = 0
        features['std_meandering_ratio'] = 0
        features['min_meandering_ratio'] = 0
        features['max_meandering_ratio'] = 0
        features['median_meandering_ratio'] = 0
        features['fraction_efficient_movement'] = 0
    
    return features

def calculate_higuchi_fd(x, k_max=10):
    """Calculate Higuchi Fractal Dimension."""
    try:
        n = len(x)
        if n < 2 * k_max:  # Need sufficient data points
            return 0
            
        l = []
        for k in range(1, k_max + 1):
            lk = 0
            for m in range(k):
                lkm = 0
                num_points = (n - m) // k
                if num_points <= 1:  # Need at least 2 points for calculation
                    continue
                    
                for i in range(1, num_points):
                    lkm += abs(x[m + i * k] - x[m + (i - 1) * k])
                
                if num_points > 0 and k > 0:
                    lkm = lkm * (n - 1) / (num_points * k)
                    lk += lkm
            
            # Avoid log of zero or negative values
            if lk > 0 and k > 0:
                l.append(np.log(lk / k))
            else:
                return 0  # Return 0 if we encounter invalid values
        
        if len(l) < 2:  # Need at least 2 points for polyfit
            return 0
            
        hfd = np.polyfit(np.log(range(1, len(l) + 1)), l, 1)[0]
        return hfd
    except Exception as e:
        return 0

def calculate_entropy_features(df):
    """Calculate entropy and randomness measures."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    # Only calculate entropy if we have enough non-NaN values
    if len(speed) > 0:
        features['speed_entropy'] = entropy(np.histogram(speed, bins=20)[0])
        features['speed_fractal_dim'] = calculate_higuchi_fd(speed)
    else:
        features['speed_entropy'] = 0
        features['speed_fractal_dim'] = 0
    
    if len(turning) > 0:
        features['turning_entropy'] = entropy(np.histogram(turning, bins=20)[0])
        features['turning_fractal_dim'] = calculate_higuchi_fd(turning)
    else:
        features['turning_entropy'] = 0
        features['turning_fractal_dim'] = 0
    
    return features

def calculate_wavelet_features(df, max_level=3):
    """Calculate features using wavelet analysis with dynamic level assignment."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    def get_safe_level(signal_length, max_level):
        """Calculate the maximum safe decomposition level for a given signal length."""
        if signal_length < 4:  # Need minimum 4 points for any wavelet decomposition
            return 0
        # Rule: signal length should be at least 2^level, with buffer for boundary effects
        safe_level = min(max_level, int(np.log2(signal_length)) - 1)
        return max(0, safe_level)  # Can be 0 for very short signals
    
    # Initialize all possible levels to 0
    for i in range(max_level + 1):
        features[f'wavelet_speed_level{i}'] = 0
        features[f'wavelet_turning_level{i}'] = 0
    
    # Calculate wavelet features for speed
    if len(speed) > 0:
        speed_level = get_safe_level(len(speed), max_level)
        if speed_level > 0:
            try:
                coeffs = pywt.wavedec(speed, 'db4', level=speed_level)
                for i, c in enumerate(coeffs):
                    if i <= max_level:  # Ensure we don't exceed our feature space
                        features[f'wavelet_speed_level{i}'] = np.std(c) if len(c) > 0 else 0
            except Exception:
                # If wavelet decomposition fails, features remain 0
                pass
    
    # Calculate wavelet features for turning
    if len(turning) > 0:
        turning_level = get_safe_level(len(turning), max_level)
        if turning_level > 0:
            try:
                coeffs = pywt.wavedec(turning, 'db4', level=turning_level)
                for i, c in enumerate(coeffs):
                    if i <= max_level:  # Ensure we don't exceed our feature space
                        features[f'wavelet_turning_level{i}'] = np.std(c) if len(c) > 0 else 0
            except Exception:
                # If wavelet decomposition fails, features remain 0
                pass
    
    return features

def calculate_roaming_dwelling_score(df, window_size=30):
    """Calculate roaming vs dwelling score."""
    # Early exit for insufficient data
    if len(df) < window_size or 'speed' not in df.columns or 'turning_angle' not in df.columns:
        return {
            'mean_roaming_score': 0,
            'std_roaming_score': 0,
            'fraction_roaming': 0
        }
    
    # Drop NaN values and convert to numpy arrays for faster processing
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    if len(speed) < window_size or len(turning) < window_size:
        return {
            'mean_roaming_score': 0,
            'std_roaming_score': 0,
            'fraction_roaming': 0
        }
    
    # Initialize result dictionary
    features = {
        'mean_roaming_score': 0,
        'std_roaming_score': 0,
        'fraction_roaming': 0
    }
    
    try:
        # Apply gaussian filter for smoothing
        speed_smooth = gaussian_filter1d(speed, sigma=3)
        turning_smooth = gaussian_filter1d(turning, sigma=3)
        
        # Calculate scores in non-overlapping windows
        roaming_scores = []
        
        for i in range(0, len(speed_smooth) - window_size + 1, window_size):
            if i + window_size <= len(speed_smooth) and i + window_size <= len(turning_smooth):
                speed_window = speed_smooth[i:i+window_size]
                turning_window = turning_smooth[i:i+window_size]
                
                # Skip windows with all zeros or NaNs
                if np.all(np.isnan(speed_window)) or np.all(np.isnan(turning_window)):
                    continue
                
                # Replace NaNs with zeros to avoid calculation errors
                speed_window = np.nan_to_num(speed_window, nan=0.0)
                turning_window = np.nan_to_num(turning_window, nan=0.0)
                
                # Calculate mean speed and turning scores
                speed_score = np.mean(speed_window)
                turning_score = np.mean(np.abs(turning_window))
                
                # Calculate roaming score
                # High speed and low turning = roaming (positive score)
                # Low speed and high turning = dwelling (negative score)
                score = (speed_score - turning_score/180) / 2
                roaming_scores.append(score)
        
        # Calculate features if we have valid scores
        if roaming_scores:
            roaming_scores_array = np.array(roaming_scores)
            features['mean_roaming_score'] = np.mean(roaming_scores_array)
            features['std_roaming_score'] = np.std(roaming_scores_array)
            features['fraction_roaming'] = np.mean(roaming_scores_array > 0)
    except Exception as e:
        # In case of any error, return zeros
        print(f"Error in calculate_roaming_dwelling_score: {e}")
    
    return features

def calculate_frenetic_movement(df, window_size=10, overlap=5):
    """Calculate metrics for short-term erratic/frenetic movement.
    
    Args:
        df: DataFrame containing movement data
        window_size: Size of windows for calculating frenetic movement (smaller than other functions)
        overlap: Overlap between consecutive windows
    """
    # Fast path for empty or insufficient data
    if len(df) < window_size or 'speed' not in df.columns or 'turning_angle' not in df.columns:
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Drop NaN values and convert to numpy arrays for faster processing
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    if len(speed) < window_size or len(turning) < window_size:
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Calculate metrics in sliding windows
    frenetic_scores = []
    jerk_values = []
    
    # Pre-compute step size
    step = window_size - overlap
    
    # Pre-allocate temporary arrays to avoid repeated allocations
    windows_count = max(1, (len(speed) - window_size) // step + 1)
    
    try:
        for i in range(0, len(speed) - window_size + 1, step):
            window_speed = speed[i:i+window_size]
            window_turning = turning[i:i+window_size]
            
            # Calculate speed acceleration
            speed_accel = np.diff(window_speed)
            
            # Calculate jerk (rate of change of acceleration)
            if len(speed_accel) > 1:
                jerk = np.diff(speed_accel)
                if len(jerk) > 0:
                    mean_jerk = np.mean(np.abs(jerk))
                    if not np.isnan(mean_jerk):
                        jerk_values.append(mean_jerk)
            
            # Calculate turning acceleration
            turning_accel = np.abs(np.diff(window_turning))
            
            # Calculate frenetic score
            if len(speed_accel) > 1 and len(turning_accel) > 1:
                speed_std = np.std(speed_accel)
                turning_std = np.std(turning_accel)
                
                if not np.isnan(speed_std) and not np.isnan(turning_std) and speed_std > 0 and turning_std > 0:
                    frenetic_score = speed_std * turning_std
                    frenetic_scores.append(frenetic_score)
    except Exception:
        # If any calculation fails, return zeros
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Initialize results
    features = {
        'mean_frenetic_score': 0,
        'max_frenetic_score': 0,
        'std_frenetic_score': 0,
        'pct_high_frenetic': 0,
        'mean_jerk': 0,
        'max_jerk': 0
    }
    
    # Calculate overall features if we have valid scores
    if frenetic_scores:
        # Convert to numpy array once for faster calculations
        frenetic_scores_array = np.array(frenetic_scores)
        features['mean_frenetic_score'] = np.mean(frenetic_scores_array)
        features['max_frenetic_score'] = np.max(frenetic_scores_array)
        features['std_frenetic_score'] = np.std(frenetic_scores_array)
        
        # Safely calculate percentile
        if len(frenetic_scores) >= 4:
            percentile_75 = np.percentile(frenetic_scores_array, 75)
            features['pct_high_frenetic'] = np.mean(frenetic_scores_array > percentile_75)
    
    # Calculate jerk features if we have valid values
    if jerk_values:
        jerk_values_array = np.array(jerk_values)
        features['mean_jerk'] = np.mean(jerk_values_array)
        features['max_jerk'] = np.max(jerk_values_array)
    
    return features

def extract_all_features(df):
    """Extract all features from a worm's time series data."""
    features = {}
    
    # Original features
    features.update(calculate_basic_features(df))
    features.update(calculate_turning_features(df))
    features.update(calculate_entropy_features(df))
    features.update(calculate_wavelet_features(df))
    features.update(calculate_roaming_dwelling_score(df))
    features.update(calculate_frenetic_movement(df))
    
    # Enhanced features
    features.update(calculate_enhanced_physics_features(df))
    features.update(calculate_enhanced_statistical_features(df))
    features.update(calculate_enhanced_temporal_features(df))
    features.update(calculate_enhanced_frequency_features(df))
    features.update(calculate_behavioral_state_features(df))
    
    return features

def normalize_features_func(features_df, normalize_features=True):
    """
    Normalize extracted features to ensure they're on comparable scales.
    
    Args:
        features_df (pd.DataFrame): DataFrame with extracted features
        normalize_features (bool): Whether to apply feature normalization
        
    Returns:
        pd.DataFrame: DataFrame with normalized features
    """
    if not normalize_features:
        return features_df
    
    # Identify columns to normalize (exclude metadata columns)
    metadata_columns = ['filename', 'label', 'death_index', 'segment_index', 'original_file']
    feature_columns = [col for col in features_df.columns if col not in metadata_columns]
    
    if not feature_columns:
        print("Warning: No feature columns found for normalization")
        return features_df
    
    # Create a copy to avoid modifying the original
    normalized_df = features_df.copy()
    
    # Use RobustScaler for feature normalization (less sensitive to outliers than StandardScaler)
    scaler = RobustScaler()
    
    # Only normalize if we have enough samples
    if len(features_df) > 1:
        try:
            # Fit and transform the feature columns
            normalized_features = scaler.fit_transform(features_df[feature_columns])
            
            # Replace the feature columns with normalized values
            normalized_df[feature_columns] = normalized_features
            
            print(f"Normalized {len(feature_columns)} features using RobustScaler")
            
            # Print some statistics about the normalization
            original_ranges = features_df[feature_columns].max() - features_df[feature_columns].min()
            normalized_ranges = normalized_df[feature_columns].max() - normalized_df[feature_columns].min()
            
            print(f"Original feature ranges: min={original_ranges.min():.6f}, max={original_ranges.max():.6f}, median={original_ranges.median():.6f}")
            print(f"Normalized feature ranges: min={normalized_ranges.min():.6f}, max={normalized_ranges.max():.6f}, median={normalized_ranges.median():.6f}")
            
        except Exception as e:
            print(f"Warning: Feature normalization failed: {e}")
            return features_df
    else:
        print("Warning: Not enough samples for feature normalization")
    
    return normalized_df

def process_all_files(input_dir, output_dir="feature_data", normalize_features=True):
    """
    Process all preprocessed files and save features to output directory.
    
    Args:
        input_dir (str): Base directory containing 'segments' and 'full' subdirectories
        output_dir (str): Directory to save feature files
        normalize_features (bool): Whether to normalize features to comparable scales
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for data_type in ['full', 'segments']:
        data_dir = os.path.join(input_dir, data_type)
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found. Skipping.")
            continue
        
        output_file = os.path.join(output_dir, f"{data_type}_features.csv")
        
        metadata_path = os.path.join(data_dir, "labels_and_metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}. Skipping.")
            continue
        
        metadata = pd.read_csv(metadata_path)
        print(f"Found {len(metadata)} files in {data_type} metadata")
        
        all_features = []
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Processing {data_type}"):
            try:
                if 'relative_path' in metadata.columns:
                    file_path = os.path.join(data_dir, row['relative_path'], row['file'])
                else:
                    file_path = os.path.join(data_dir, row['file'])
                
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found")
                    continue
                
                # Read and extract features
                df = pd.read_csv(file_path)
                features = extract_all_features(df)
                
                # Add metadata information
                features['filename'] = row['file']
                features['label'] = row['label']
                
                # Add additional metadata if available
                if 'death_index' in row:
                    features['death_index'] = row['death_index']
                if 'segment_index' in row:
                    features['segment_index'] = row['segment_index']
                if 'original_file' in row:
                    features['original_file'] = row['original_file']
                
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {row['file']}: {e}")
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Apply feature normalization if requested
            if normalize_features:
                print(f"\nNormalizing features for {data_type} data...")
                features_df = normalize_features_func(features_df, normalize_features=True)
            
            features_df.to_csv(output_file, index=False)
            print(f"Saved features for {len(features_df)} {data_type} to {output_file}")
        else:
            print(f"Warning: No features extracted for {data_type}")
    
    return True

if __name__ == "__main__":
    input_dir = "preprocessed_data"
    output_dir = "feature_data"
    process_all_files(input_dir, output_dir, normalize_features=True)