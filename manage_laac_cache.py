#!/usr/bin/env python3
"""
LAAC 缓存管理工具

用法:
    python manage_laac_cache.py --dataset DBLP --action list
    python manage_laac_cache.py --dataset DBLP --action clear
    python manage_laac_cache.py --dataset IMDB --action list
"""

import argparse
import sys
sys.path.append('utils/')

from data import clear_laac_cache, list_laac_cache

def main():
    parser = argparse.ArgumentParser(description='LAAC Cache Management Tool')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (e.g., DBLP, IMDB)')
    parser.add_argument('--action', type=str, required=True, 
                        choices=['list', 'clear'],
                        help='Action to perform: list (show cache files) or clear (remove cache)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"LAAC Cache Manager - Dataset: {args.dataset}")
    print(f"{'='*80}\n")
    
    if args.action == 'list':
        list_laac_cache(args.dataset)
    elif args.action == 'clear':
        # 确认删除
        confirm = input(f"Are you sure you want to clear all LAAC cache for '{args.dataset}'? (yes/no): ")
        if confirm.lower() in ['yes', 'y']:
            clear_laac_cache(args.dataset)
        else:
            print("Operation cancelled.")
    
    print()

if __name__ == '__main__':
    main()
