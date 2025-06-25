import numpy as np
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# Constants
FRAME_RESET_VALUE = 10799
SEGMENT_LENGTH = 900
GAP_INTERPOLATION_LIMIT = 6
LONG_GAP_THRESHOLD = 7



def calculate_turning_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the turning angle at each point in a trajectory.

    The turning angle at row 'i' (time t) is defined as the change in
    direction from the movement vector (t-1 -> t) to the movement vector (t -> t+1).

    Args:
        df (pd.DataFrame): DataFrame with trajectory data, sorted by time.
                           Must contain x, y coordinates.

    Returns:
        pd.DataFrame: The original DataFrame with a new column 'turning_angle'
                      added. The angle is in degrees, ranging from -180 to +180.
                      Positive values indicate counter-clockwise turns, negative
                      values indicate clockwise turns.
                      The first and last rows will have 0 for the turning angle.
    """
    # Check if we have sufficient data
    if len(df) < 3:  # Need at least 3 points for turning angle
        df['turning_angle'] = 0
        return df
    
    try:
        if 'x' not in df.columns or 'y' not in df.columns:
            df['turning_angle'] = 0
            return df
        
        # Check if x and y columns contain valid trajectory data
        if df['x'].isna().all() or df['y'].isna().all():
            df['turning_angle'] = 0
            return df
        
        dx = df['x'].diff()
        dy = df['y'].diff()
        
        # Convert to numpy arrays to avoid pandas ufunc issues
        dx_values = dx.values
        dy_values = dy.values
        
        angle = np.arctan2(dy_values, dx_values)
        angle_next = np.roll(angle, -1)  # Shift forward by 1
        turning_angle_rad = angle_next - angle
        
        turning_angle_rad = (turning_angle_rad + np.pi) % (2 * np.pi) - np.pi
        df['turning_angle'] = np.degrees(turning_angle_rad)
        
        # Set first and last points to 0
        df.loc[0, 'turning_angle'] = 0
        df.loc[df.index[-1], 'turning_angle'] = 0
        
    except Exception as e:
        print(f"Warning: Could not calculate turning angles: {e}")
        df['turning_angle'] = 0

    return df

def detect_death(df, min_pause_duration=30, percentile_threshold=5):
    """
    Detect death by looking backwards from the end of the recording.
    A worm is considered dead at the start of the last sustained period of low movement.
    
    Args:
        df: DataFrame with x, y coordinates
        min_pause_duration: Minimum number of consecutive frames with low movement to consider as death
        percentile_threshold: Percentile of movement to use as threshold (5 = 5th percentile)
        
    Returns:
        Index of first frame where the last sustained low movement begins
    """
    # print("Death detection: ", end="")
    df_clean = df.dropna(subset=['x', 'y'])
    
    if len(df_clean) < 2:
        return len(df)
    
    movement = np.sqrt(np.diff(df_clean['x'])**2 + np.diff(df_clean['y'])**2)
    
    if len(movement) == 0:
        return len(df)
    
    threshold = np.percentile(movement, percentile_threshold)
    if threshold <= 0:
        threshold = np.mean(movement) * 0.1
        # print(f"adjusted threshold {threshold:.6f}, ", end="")
    
    low_movement = movement < threshold
    
    current_sequence = []
    death_candidates = []
    for i in range(len(low_movement) - 1, -1, -1):
        if low_movement[i]:
            current_sequence.append(i)
        else:
            if len(current_sequence) >= min_pause_duration:
                start_idx = current_sequence[-1] + 1
                end_idx = current_sequence[0] + 1
                death_candidates.append((start_idx, end_idx, len(current_sequence)))
            current_sequence = []
    
    if len(current_sequence) >= min_pause_duration:
        start_idx = current_sequence[-1] + 1
        end_idx = current_sequence[0] + 1
        death_candidates.append((start_idx, end_idx, len(current_sequence)))
    
    if not death_candidates:
        # print("no death detected")
        return len(df)
    
    death_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Just print the top candidate
    start, end, duration = death_candidates[0]
    # print(f"found at {start}, duration {duration}")
    
    death_start_idx = death_candidates[0][0]
    
    # Verify this is a reasonable death (not in middle of recording with lots of activity after)
    remaining_movement = movement[death_start_idx:]
    if len(remaining_movement) > 100 and np.mean(remaining_movement > threshold) > 0.4: # More than 40% of frames after death show movement - suspicious
        # print("     > WARNING: Significant movement detected after presumed death, using end of recording instead")
        return len(df)
    
    original_idx = df_clean.index[death_start_idx] if death_start_idx < len(df_clean) else len(df)  # Map back to the original dataframe index
    
    perc_elapsed = (original_idx / len(df)) * 100
    return original_idx

def split_into_segments(df, segment_length=SEGMENT_LENGTH):
    """
    Split a DataFrame into segments of specified length.
    
    Args:
        df (pd.DataFrame): DataFrame to split
        segment_length (int): Length of each segment
        
    Returns:
        list: List of DataFrames, each representing a segment
    """
    segments = []
    num_segments = (len(df) // segment_length) + (1 if len(df) % segment_length > 0 else 0)
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(df))
        segment_df = df.iloc[start_idx:end_idx].copy()
        segment_df['segment_index'] = i  # Add segment index to each segment
        segments.append(segment_df)
    
    return segments

def normalize_trajectory_data(df):
    """
    Normalize trajectory data to make features scale-invariant.
    Applied at the end of preprocessing to ensure consistent scales.
    """
    df_normalized = df.copy()
    
    # Normalize coordinates by bounding box (per trajectory)
    if 'x' in df.columns and 'y' in df.columns:
        x_clean = df['x'].dropna()
        y_clean = df['y'].dropna()
        
        if len(x_clean) > 0 and len(y_clean) > 0:
            # Calculate bounding box
            x_min, x_max = x_clean.min(), x_clean.max()
            y_min, y_max = y_clean.min(), y_clean.max()
            
            # Normalize to [0, 1] range
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range > 0:
                df_normalized['x'] = (df['x'] - x_min) / x_range
            if y_range > 0:
                df_normalized['y'] = (df['y'] - y_min) / y_range
    
    # Normalize speed using robust statistics (per trajectory)
    if 'speed' in df.columns:
        speed_clean = df['speed'].dropna()
        if len(speed_clean) > 1:  # Need at least 2 values
            # Use robust normalization (median and IQR)
            median_speed = speed_clean.median()
            q75 = speed_clean.quantile(0.75)
            q25 = speed_clean.quantile(0.25)
            iqr = q75 - q25
            
            if iqr > 0:
                df_normalized['speed'] = (df['speed'] - median_speed) / iqr
            else:
                # Fallback to mean normalization if IQR is 0
                mean_speed = speed_clean.mean()
                std_speed = speed_clean.std()
                if std_speed > 0:
                    df_normalized['speed'] = (df['speed'] - mean_speed) / std_speed
                else:
                    # If no variation, set to 0
                    df_normalized['speed'] = 0
    
    return df_normalized

def preprocess_data(file_path, full_output_dir, segments_output_dir):
    """
    Preprocess a single data file, creating both full and segmented versions.
    
    Args:
        file_path (str): Path to the raw data file
        full_output_dir (str): Directory to save full preprocessed data
        segments_output_dir (str): Directory to save segmented preprocessed data
        
    Returns:
        Tuple of (full_result, segments_results) where:
        - full_result: (preprocessed_filename, death_index) or (None, None) if failed
        - segments_results: List of tuples [(segment_filename, segment_death_index)] or empty list if failed
    """
    filename = os.path.basename(file_path)
    try:
        # First, read the header to determine the file format
        header_line = pd.read_csv(file_path, nrows=0).columns.tolist()
        
        # Detect file format based on headers
        if 'GlobalFrame' in header_line and 'Timestamp' in header_line and 'Fragment' in header_line:
            # New TERBINAFINE format: GlobalFrame,Timestamp,Speed,Fragment,LocalFrame,X,Y
            # print(f"Detected new TERBINAFINE format for {filename}")
            df = pd.read_csv(file_path)
            # Map to standard column names
            df = df.rename(columns={
                'GlobalFrame': 'frame',
                'Speed': 'speed',
                'X': 'x',
                'Y': 'y'
            })
            # Add changed_pixels column as NaN since it's not available in new format
            df['changed_pixels'] = np.nan
            # Keep only the columns we need
            df = df[['frame', 'speed', 'x', 'y', 'changed_pixels']]
            
        elif 'Frame' in header_line and 'Changed Pixels' in header_line:
            # Old format: Frame,Speed,X,Y,Changed Pixels
            # print(f"Detected old TERBINAFINE format for {filename}")
            df = pd.read_csv(file_path, skiprows=1, names=['frame', 'speed', 'x', 'y', 'changed_pixels'])
            
        else:
            # Default format (original format used in other datasets)
            # print(f"Detected original format for {filename}")
            df = pd.read_csv(file_path, skiprows=1, names=['frame', 'speed', 'x', 'y', 'changed_pixels'])
            
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return (None, None), []

    # Detect death before any modifications to the dataframe
    death_index = detect_death(df)

    # Handle frame index reset at FRAME_RESET_VALUE (only for original format)
    if filename.startswith('speeds_and_coordinates_'):
        df = pd.concat([df, df[df['frame'] == FRAME_RESET_VALUE]])
        df = df.sort_index().reset_index(drop=True)
    
    # Renumber frames
    df['frame'] = range(1, len(df) + 1)
    
    # Add segment information
    df['segment'] = (df['frame'] - 1) // SEGMENT_LENGTH
    
    # Process full data first
    df_full = df.copy()
    
    # Handle missing values for full data
    gap_mask = df_full['x'].isna()
    
    # To properly identify gaps (consecutive NaN values), we need to identify 
    # where NaN sequences start and end
    is_nan = gap_mask.astype(int)
    # Mark the start of NaN sequences with 1, other positions with 0
    starts = (is_nan.diff() == 1).astype(int)
    # Handle first position if it's NaN
    if is_nan.iloc[0] == 1:
        starts.iloc[0] = 1
        
    # Assign each gap a unique ID
    gap_ids = starts.cumsum()
    # Only keep gap IDs for actual NaN positions
    gap_ids = gap_ids * is_nan
    
    # Group NaN values by their gap ID to get each gap's size and indices
    gap_sizes = {}
    gap_indices = {}
    
    # Only process non-zero gap IDs (actual gaps)
    for gap_id in gap_ids[gap_ids > 0].unique():
        indices = df_full.index[gap_ids == gap_id].tolist()
        gap_sizes[gap_id] = len(indices)
        gap_indices[gap_id] = indices
    
    rows_to_remove = []
    for i, (gap_id, size) in enumerate(gap_sizes.items(), 1):
        if size <= GAP_INTERPOLATION_LIMIT:
            # Get indices for this gap
            gap_idx = gap_indices[gap_id]
            # Ensure we have surrounding points for interpolation
            if len(gap_idx) > 0:
                start_idx = max(0, gap_idx[0] - 1)
                end_idx = min(len(df_full) - 1, gap_idx[-1] + 1)
                # Only interpolate the x, y, speed columns for this specific gap range
                df_full.loc[start_idx:end_idx, ['x', 'y', 'speed']] = df_full.loc[start_idx:end_idx, ['x', 'y', 'speed']].interpolate(method='linear')
        elif size >= LONG_GAP_THRESHOLD:
            # For long gaps, remove all points except the first (if needed)
            indices = gap_indices[gap_id]
            if len(indices) > 1:
                rows_to_remove.extend(indices[1:])

    df_full = df_full.drop(index=rows_to_remove).reset_index(drop=True)
    
    # Add turning angle feature
    df_full = calculate_turning_angle(df_full)

    # Apply normalization to time series data
    df_full = normalize_trajectory_data(df_full)

    # Save full preprocessed data
    base_filename = os.path.basename(file_path).replace(".csv", "-preprocessed.csv")
    full_preprocessed_filename = os.path.join(full_output_dir, base_filename)
    full_result = (None, None)
    try:
        df_full.to_csv(full_preprocessed_filename, index=False)
        full_result = (full_preprocessed_filename, death_index)
    except Exception as e:
        print(f"Error saving full preprocessed data for {filename}: {e}")
    
    # Process segments
    # Process each segment independently
    segments = split_into_segments(df)
    
    segments_results = []
    
    # Use tqdm for progress bar instead of prints for each segment
    for i, segment_df in tqdm(enumerate(segments), total=len(segments), desc="Processing segments", leave=False):
        # Handle missing values within this segment
        gap_mask = segment_df['x'].isna()
        
        # To properly identify gaps (consecutive NaN values), we need to identify 
        # where NaN sequences start and end
        is_nan = gap_mask.astype(int)
        # Mark the start of NaN sequences with 1, other positions with 0
        starts = (is_nan.diff() == 1).astype(int)
        # Handle first position if it's NaN
        if is_nan.iloc[0] == 1 and len(is_nan) > 0:
            starts.iloc[0] = 1
        
        # Assign each gap a unique ID
        gap_ids = starts.cumsum()
        # Only keep gap IDs for actual NaN positions
        gap_ids = gap_ids * is_nan
        
        # Group NaN values by their gap ID to get each gap's size and indices
        gap_sizes = {}
        gap_indices = {}
        
        # Only process non-zero gap IDs (actual gaps)
        for gap_id in gap_ids[gap_ids > 0].unique():
            indices = segment_df.index[gap_ids == gap_id].tolist()
            gap_sizes[gap_id] = len(indices)
            gap_indices[gap_id] = indices
        
        rows_to_remove = []
        for j, (gap_id, size) in enumerate(gap_sizes.items(), 1):                    
            if 1 <= size <= GAP_INTERPOLATION_LIMIT:
                # Get indices for this gap
                gap_idx = gap_indices[gap_id]
                # Ensure we have surrounding points for interpolation
                if len(gap_idx) > 0:
                    start_idx = max(0, gap_idx[0] - 1)
                    end_idx = min(len(segment_df) - 1, gap_idx[-1] + 1)
                    # Only interpolate the x, y, speed columns for this specific gap range
                    segment_df.loc[start_idx:end_idx, ['x', 'y', 'speed']] = segment_df.loc[start_idx:end_idx, ['x', 'y', 'speed']].interpolate(method='linear')
            elif size >= LONG_GAP_THRESHOLD:
                # For long gaps, remove all points except the first (if needed)
                indices = gap_indices[gap_id]
                if len(indices) > 1:
                    rows_to_remove.extend(indices[1:])

        segment_df = segment_df.drop(index=rows_to_remove).reset_index(drop=True)
        
        # Add turning angle feature
        segment_df = calculate_turning_angle(segment_df)
        
        # Apply normalization to segment time series data
        segment_df = normalize_trajectory_data(segment_df)
        
        # Save segment file
        segment_filename = os.path.basename(file_path).replace(".csv", f"-segment{i}-preprocessed.csv")
        segment_path = os.path.join(segments_output_dir, segment_filename)
        
        try:
            segment_df.to_csv(segment_path, index=False)
            segments_results.append((segment_path, death_index))
        except Exception as e:
            print(f"Error saving segment {i} for {filename}: {e}")
            segments_results.append((None, None))
    
    return full_result, segments_results

def process_directory(input_dir, output_dir):
    """
    Process all CSV files in a directory and its subdirectories.
    Creates both full and segmented preprocessed data.
    
    Args:
        input_dir (str): Directory containing raw data files
        output_dir (str): Directory to save preprocessed data
    """
    # Create output directories if they don't exist
    segments_output_dir = os.path.join(output_dir, "segments")
    full_output_dir = os.path.join(output_dir, "full")
    os.makedirs(segments_output_dir, exist_ok=True)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Initialize lists to store metadata
    segments_metadata = []
    full_metadata = []
    
    # First, collect all files and their labels, organized by folder
    file_labels = {}
    files_by_folder = {}
    
    # Find all CSV files in input directory and subdirectories
    print(f"Scanning directory: {input_dir} for CSV files...")
    for root, dirs, files in os.walk(os.path.join(input_dir)):
        # Skip Optogenetics folder and its subdirectories
        if 'Optogenetics' in root:
            print(f"Skipping Optogenetics folder: {root}")
            continue
            
        csv_files_in_folder = []
        for filename in files:
            if filename.endswith(".csv"):
                # Skip summary files and other non-trajectory files
                if any(skip_word in filename.lower() for skip_word in ['summary', 'metadata', 'labels']):
                    # print(f"Skipping non-trajectory file: {filename}")
                    continue
                    
                file_path = os.path.join(root, filename)
                csv_files_in_folder.append(file_path)

                if "+" in root:
                    label = 1
                elif "-" in root:
                    label = 0
                else:
                    # Default to 0 if no label information is found
                    print(f"Warning: No label indicator (+ or -) found in path for {file_path}, defaulting to label 0")
                    label = 0
                file_labels[filename] = label
        
        if csv_files_in_folder:
            folder_name = os.path.relpath(root, input_dir)
            files_by_folder[folder_name] = csv_files_in_folder
    
    total_files = sum(len(files) for files in files_by_folder.values())
    print(f"Found {total_files} CSV files in {len(files_by_folder)} folders to process\n")
    
    # Process each folder
    for folder_name, csv_files in files_by_folder.items():
        print(f"\nProcessing folder: {folder_name}")
        # Process each CSV file in the folder with its own progress bar
        for file_path in tqdm(csv_files, desc=f"Files in {os.path.basename(folder_name)}", leave=True):
            filename = os.path.basename(file_path)
            
            # Get the label for this file
            label = file_labels.get(filename, 0)  # Default to 0 if label not found
            
            # Get relative path from input_dir to maintain directory structure
            rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)
            
            # Create corresponding output directories
            segments_rel_dir = os.path.join(segments_output_dir, rel_path)
            full_rel_dir = os.path.join(full_output_dir, rel_path)
            os.makedirs(segments_rel_dir, exist_ok=True)
            os.makedirs(full_rel_dir, exist_ok=True)
            
            # Process the file (creates both full and segments)
            full_result, segments_results = preprocess_data(file_path, full_rel_dir, segments_rel_dir)
            
            # Handle full file result
            if full_result[0] is not None:
                preprocessed_file, death_index = full_result
                preprocessed_basename = os.path.basename(preprocessed_file)
                full_metadata.append({
                    'file': preprocessed_basename,
                    'original_file': filename,
                    'death_index': death_index,
                    'label': label,
                    'relative_path': rel_path
                })
            
            # Handle segment results
            for segment_path, segment_death_index in segments_results:
                if segment_path is not None:
                    segment_basename = os.path.basename(segment_path)
                    # Extract segment index from filename
                    segment_match = re.search(r'-segment(\d+)-', segment_basename)
                    segment_index = int(segment_match.group(1)) if segment_match else None
                    
                    # Move segment file to correct directory structure
                    new_segment_path = os.path.join(segments_rel_dir, segment_basename)
                    os.rename(segment_path, new_segment_path)
                    
                    segments_metadata.append({
                        'file': segment_basename,
                        'original_file': filename,
                        'segment_index': segment_index,
                        'death_index': segment_death_index,
                        'label': label,
                        'relative_path': rel_path
                    })
    
    if segments_metadata:
        segments_metadata_df = pd.DataFrame(segments_metadata)
        segments_metadata_path = os.path.join(segments_output_dir, "labels_and_metadata.csv")
        segments_metadata_df.to_csv(segments_metadata_path, index=False)
        print(f"Saved segments metadata to {segments_metadata_path}")
        
        # Print label distribution for segments
        if 'label' in segments_metadata_df.columns:
            label_counts = segments_metadata_df['label'].value_counts()
            print("\nLabel distribution in processed segments:")
            for label, count in label_counts.items():
                print(f"Label {label}: {count} segments")
    
    if full_metadata:
        full_metadata_df = pd.DataFrame(full_metadata)
        full_metadata_path = os.path.join(full_output_dir, "labels_and_metadata.csv")
        full_metadata_df.to_csv(full_metadata_path, index=False)
        print(f"Saved full data metadata to {full_metadata_path}")
        
        # Print label distribution for full data
        if 'label' in full_metadata_df.columns:
            label_counts = full_metadata_df['label'].value_counts()
            print("\nLabel distribution in processed full data:")
            for label, count in label_counts.items():
                print(f"Label {label}: {count} files")
    
    if not segments_metadata and not full_metadata:
        print("\nWarning: No metadata was generated")
    
    print(f"\n✅ Processing complete!")
    print(f"   Full data files processed: {len(full_metadata)}")
    print(f"   Segment files processed: {len(segments_metadata)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess trajectory data - creates both full and segmented versions')
    parser.add_argument('--input_dir', type=str, default='data', help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data', help='Directory to save preprocessed data')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)