import os
from pathlib import Path
from scripts.download_data import EEGDataDownloader
from scripts.train import main
import traceback
from tqdm import tqdm

def setup_project():
    """设置项目目录结构"""
    dirs = [
        'data',
        'checkpoints',
        'logs'
    ]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"Created directory: {d}")

def download_datasets():
    """下载并准备所有数据集"""
    try:
        downloader = EEGDataDownloader(root_dir="data")
        print("\nDownloading all datasets...")
        datasets = downloader.download_all()
        print("\nDatasets downloaded to:")
        for name, path in datasets.items():
            print(f"- {name}: {path}")
        return datasets
    except Exception as e:
        print("Error downloading datasets:")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    try:
        # 1. 设置项目结构
        print("Setting up project structure...")
        setup_project()
        
        # 2. 下载所有数据集
        print("\nDownloading and preparing datasets...")
        datasets = download_datasets()
        
        # 3. 开始训练
        print("\nStarting training...")
        main()  # 移除data_dir参数
        
    except Exception as e:
        print("\nError occurred during execution:")
        print(traceback.format_exc())