"""
split_data.py

Split processed data into train and test splits.
Supports random splitting and stratified splitting by db_name.
"""

import os
import argparse
import random
from collections import defaultdict
from data_utils.common_utils import read_json, write_json


def split_data_random(data, train_ratio=0.8, seed=None):
    """
    Randomly split data into train and test sets.
    
    Args:
        data: List of data items
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        train_data, test_data: Two lists of data items
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    split_idx = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data


def split_data_stratified(data, train_ratio=0.8, seed=None, stratify_by='db_name'):
    """
    Stratified split by db_name to ensure each database appears in both train and test.
    
    Args:
        data: List of data items
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
        stratify_by: Field name to stratify by (default: 'db_name')
    
    Returns:
        train_data, test_data: Two lists of data items
    """
    if seed is not None:
        random.seed(seed)
    
    # Group data by stratification key
    grouped_data = defaultdict(list)
    for item in data:
        key = item.get(stratify_by, 'unknown')
        grouped_data[key].append(item)
    
    # Shuffle each group
    for key in grouped_data:
        random.shuffle(grouped_data[key])
    
    train_data = []
    test_data = []
    
    # Split each group proportionally
    for key, items in grouped_data.items():
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data


def reindex_data(data, start_idx=0):
    """
    Reindex data items to have sequential idx starting from start_idx.
    
    Args:
        data: List of data items
        start_idx: Starting index (default: 0)
    
    Returns:
        List of data items with updated idx
    """
    reindexed = []
    for i, item in enumerate(data):
        new_item = item.copy()
        new_item['idx'] = start_idx + i
        reindexed.append(new_item)
    return reindexed


def main():
    parser = argparse.ArgumentParser(
        description='Split processed data into train and test splits'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to processed data JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save train and test split files'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data to use for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Use stratified splitting by db_name (ensures each database appears in both splits)'
    )
    parser.add_argument(
        '--reindex',
        action='store_true',
        help='Reindex data items to have sequential idx starting from 0'
    )
    parser.add_argument(
        '--train_suffix',
        type=str,
        default='_train',
        help='Suffix for train file (default: _train)'
    )
    parser.add_argument(
        '--test_suffix',
        type=str,
        default='_test',
        help='Suffix for test file (default: _test)'
    )
    parser.add_argument(
        '--train_config',
        type=str,
        default=None,
        help='Path to config file for the train split (optional)'
    )
    parser.add_argument(
        '--test_config',
        type=str,
        default=None,
        help='Path to config file for the test split (optional)'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help='Task name for config entries (e.g., nl2sqlite). Auto-detected if not provided.'
    )
    parser.add_argument(
        '--data_aug',
        action='store_true',
        help='Set data_aug to true in config entries (default: false)'
    )
    
    args = parser.parse_args()
    
    # Validate train_ratio
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {args.train_ratio}")
    
    # Read input data
    print(f"[INFO] Reading data from {args.input_file}")
    data = read_json(args.input_file)
    print(f"[INFO] Total samples: {len(data)}")
    
    # Split data
    if args.stratified:
        print(f"[INFO] Using stratified splitting by db_name")
        train_data, test_data = split_data_stratified(
            data, 
            train_ratio=args.train_ratio, 
            seed=args.seed
        )
    else:
        print(f"[INFO] Using random splitting")
        train_data, test_data = split_data_random(
            data, 
            train_ratio=args.train_ratio, 
            seed=args.seed
        )
    
    print(f"[INFO] Train samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"[INFO] Test samples: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    # Reindex if requested
    if args.reindex:
        print(f"[INFO] Reindexing data items")
        train_data = reindex_data(train_data, start_idx=0)
        test_data = reindex_data(test_data, start_idx=0)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"[INFO] Created output directory: {args.output_dir}")
    
    # Generate output file names
    input_basename = os.path.basename(args.input_file)
    input_name, input_ext = os.path.splitext(input_basename)
    
    train_filename = f"{input_name}{args.train_suffix}{input_ext}"
    test_filename = f"{input_name}{args.test_suffix}{input_ext}"
    
    train_path = os.path.join(args.output_dir, train_filename)
    test_path = os.path.join(args.output_dir, test_filename)
    
    # Save splits
    print(f"[INFO] Saving train split to {train_path}")
    write_json(train_path, train_data)
    
    print(f"[INFO] Saving test split to {test_path}")
    write_json(test_path, test_data)
    
    # Print statistics
    if args.stratified:
        # Count databases in each split
        train_dbs = set(item.get('db_name', 'unknown') for item in train_data)
        test_dbs = set(item.get('db_name', 'unknown') for item in test_data)
        print(f"[INFO] Unique databases in train: {len(train_dbs)}")
        print(f"[INFO] Unique databases in test: {len(test_dbs)}")
        print(f"[INFO] Databases in both splits: {len(train_dbs & test_dbs)}")
    
    # Update config files if provided
    task_name = args.task_name
    if not task_name:
        if 'nl2sqlite' in input_name.lower():
            task_name = 'nl2sqlite'
        elif 'nl2postgres' in input_name.lower():
            task_name = 'nl2postgres'
        elif 'nl2mysql' in input_name.lower():
            task_name = 'nl2mysql'
        else:
            task_name = 'nl2sql'

    def update_config(config_path, config_key, data_path, data_len, is_train):
        if not config_path:
            return
        print(f"[INFO] Updating config file: {config_path}")
        
        # Read existing config or create new
        if os.path.exists(config_path):
            config_data = read_json(config_path)
        else:
            config_data = {}
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
        
        # Prepare path for config
        path_for_config = data_path
        if os.path.isabs(args.output_dir) and '/data/' in args.output_dir:
            data_idx = args.output_dir.index('/data/') + 6
            relative_base = args.output_dir[data_idx:]
            path_for_config = os.path.join(relative_base, os.path.basename(data_path))
        elif not os.path.isabs(data_path):
            path_for_config = os.path.join(args.output_dir, os.path.basename(data_path))
            
        config_data[config_key] = {
            "data_path": path_for_config,
            "sample_num": -1,
            "sum_num": data_len,
            "task_name": task_name,
            "data_aug": args.data_aug if is_train else False
        }
        write_json(config_path, config_data)
        print(f"  - Added/Updated: {config_key}")

    train_config_key = f"{input_name}{args.train_suffix}"
    test_config_key = f"{input_name}{args.test_suffix}"

    update_config(args.train_config, train_config_key, train_path, len(train_data), True)
    update_config(args.test_config, test_config_key, test_path, len(test_data), False)

    print(f"[INFO] Split completed successfully!")


if __name__ == '__main__':
    main()
