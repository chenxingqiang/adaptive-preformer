import os
import mne
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

class EEGDataDownloader:
    def __init__(self, root_dir="data"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
        
    def download_sample_data(self):
        """Download sample EEG data using MNE"""
        dataset_fold = self.root_dir / "SampleEEG"
        dataset_fold.mkdir(exist_ok=True)
        
        try:
            print("Downloading sample EEG dataset...")
            
            # 使用MNE示例数据
            sample_data_folder = mne.datasets.sample.data_path()
            raw = mne.io.read_raw_fif(os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw.fif'), 
                                    verbose=False)
            raw.pick_types(meg=False, eeg=True)  # 只保留EEG通道
            
            # 创建事件
            events = mne.make_fixed_length_events(raw, duration=2.0)
            
            # 提取epochs
            epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=2.0,
                              baseline=None, preload=True, verbose=False)
            
            # 保存处理后的数据
            print("Processing and saving data...")
            data = epochs.get_data()
            labels = np.random.randint(0, 2, size=len(data))  # 随机生成标签
            
            # 保存为单个文件
            torch.save({
                'data': torch.tensor(data, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            }, dataset_fold / "sample_data.pt")
            
            print(f"Dataset saved to {dataset_fold}")
            return dataset_fold
            
        except Exception as e:
            print(f"Error preparing sample dataset: {str(e)}")
            raise e
            
    def download_eegbci(self):
        """Download EEG Motor Movement/Imagery Dataset"""
        dataset_fold = self.root_dir / "EEGBCI"
        dataset_fold.mkdir(exist_ok=True)
        
        try:
            print("Downloading EEG Motor Movement/Imagery dataset...")
            subjects = list(range(1, 3))  # 下载前2个被试的数据
            
            all_data = []
            all_labels = []
            
            for subject in tqdm(subjects, desc="Processing subjects"):
                runs = [4, 8, 12]  # 选择运动想象任务
                raw_fnames = mne.datasets.eegbci.load_data(subject, runs, verbose=False)
                raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False)
                                             for f in raw_fnames])
                
                events, _ = mne.events_from_annotations(raw, verbose=False)
                
                epochs = mne.Epochs(raw, events, tmin=-1., tmax=4., 
                                  baseline=None, preload=True, verbose=False)
                
                data = epochs.get_data()
                labels = events[:len(data), -1] - 1  # 转换为0-based标签
                
                all_data.append(torch.tensor(data, dtype=torch.float32))
                all_labels.append(torch.tensor(labels % 2, dtype=torch.long))
                
            # 合并所有数据并保存为单个文件
            all_data = torch.cat(all_data, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            torch.save({
                'data': all_data,
                'labels': all_labels
            }, dataset_fold / "eegbci_data.pt")
                    
            return dataset_fold
            
        except Exception as e:
            print(f"Error downloading EEGBCI dataset: {str(e)}")
            raise e

    def download_all(self):
        """Download all available datasets"""
        datasets = {}
        print("Downloading all datasets...")
        
        try:
            datasets['sample'] = self.download_sample_data()
            print("\nSample dataset downloaded successfully.")
            
            datasets['eegbci'] = self.download_eegbci()
            print("\nEEGBCI dataset downloaded successfully.")
            
        except Exception as e:
            print(f"\nError during download: {str(e)}")
            
        return datasets