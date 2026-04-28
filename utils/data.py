import pickle
import sys
import os
import glob

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
from utils.laac_module import complete_attributes_laac


def clear_laac_cache(prefix='DBLP'):
    """
    clear LAAC cache for the specified dataset
    
    Args:
        prefix: dataset name (e.g., 'DBLP', 'IMDB')
    """
    cache_dir = f'data/{prefix}/laac_cache'
    if os.path.exists(cache_dir):
        cache_files = glob.glob(f'{cache_dir}/*.pkl')
        if cache_files:
            print(f"Clearing {len(cache_files)} LAAC cache files from {cache_dir}...")
            for cache_file in cache_files:
                os.remove(cache_file)
                print(f"  Removed: {cache_file}")
            print("Cache cleared successfully.")
        else:
            print(f"No cache files found in {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


def list_laac_cache(prefix='DBLP'):
    """
    list LAAC cache files for the specified dataset
    
    Args:
        prefix: dataset name (e.g., 'DBLP', 'IMDB')
    """
    cache_dir = f'data/{prefix}/laac_cache'
    if os.path.exists(cache_dir):
        cache_files = glob.glob(f'{cache_dir}/*.pkl')
        if cache_files:
            print(f"\nLAAC Cache files in {cache_dir}:")
            print("-" * 80)
            for cache_file in sorted(cache_files):
                file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
                mod_time = os.path.getmtime(cache_file)
                from datetime import datetime
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {os.path.basename(cache_file):40s} | Size: {file_size:8.2f} MB | Modified: {mod_date}")
            print("-" * 80)
            print(f"Total: {len(cache_files)} cache files")
        else:
            print(f"No cache files found in {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")

def load_data(prefix='DBLP'):
    from utils.data_loader import data_loader
    dl = data_loader('data/'+prefix)
    
    # Need to calculate adjM first because LAAC prompts depend on topology
    adjM = sum(dl.links['data'].values())
    
    # Create cache directory
    cache_dir = f'data/{prefix}/laac_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            # ======= HINAC LAAC Replacement Area =======
            cache_file = f'{cache_dir}/node_type_{i}_features.pkl'
            
            # Check if there is a cache
            if os.path.exists(cache_file):
                print(f"\n[+] Node Type {i}: Loading cached LAAC features from {cache_file}")
                sys.stdout.flush()
                try:
                    with open(cache_file, 'rb') as f:
                        completed_features = pickle.load(f)
                    print(f"[+] Cache loaded successfully. Shape: {completed_features.shape}")
                    sys.stdout.flush()
                    features.append(completed_features)
                    continue
                except Exception as e:
                    print(f"[!] Failed to load cache: {e}. Will regenerate...")
                    sys.stdout.flush()
            
            # No cache or cache loading failed, execute LAAC inference
            print(f"\n[+] Detecting missing attributes for Node Type {i}.")
            print(f"[+] Node Type {i} has {dl.nodes['count'][i]} nodes without pre-computed features.")
            print(f"[+] Invoking LAAC (LLM-Augmented Attribute Completion)...")
            sys.stdout.flush()
            
            # Dynamically generate dense semantic feature vectors for completion
            completed_features = complete_attributes_laac(i, dl, adjM)
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(completed_features, f)
                print(f"[+] LAAC result saved to cache: {cache_file}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[!] Failed to save cache: {e}")
                sys.stdout.flush()
            
            features.append(completed_features)
            print(f"[+] LAAC completed for Node Type {i}. Feature shape: {completed_features.shape}")
            sys.stdout.flush()
            # ====================================
        else:
            features.append(th)
            print(f"[+] Node Type {i}: Using pre-computed features. Shape: {th.shape}")
            sys.stdout.flush()
            
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB' and prefix != 'IMDB-HGB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl
