import os
import mne
import torch
import tqdm
from pathlib import Path

class EEGDataDownloader:
    def __init__(self, root_dir="data"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
        
    def download_physionet_p300(self):
        """Download PhysioNet P300 dataset"""
        dataset_fold = self.root_dir / "PhysioNetP300"
        dataset_fold.mkdir(exist_ok=True)

        all_chans = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 
                     'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 
                     'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 
                     'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 
                     'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 
                     'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 
                     'P10', 'PO8', 'PO4', 'O2']

        fmin, fmax = 0, 120
        tmin, tmax = -0.1, 2

        for sub in [2,3,4,5,6,7,9,11]:
            path = f"erp-based-brain-computer-interface-recordings-1.0.0/files/s{sub:02d}"
            
            # Create directories for each class
            for label in [0, 1]:
                (dataset_fold / str(label)).mkdir(exist_ok=True)

            for file in os.listdir(path):
                if not file.endswith(".edf"):
                    continue
                    
                raw = mne.io.read_raw_edf(os.path.join(path, file))
                raw.pick_channels(all_chans)
                
                events, event_id = mne.events_from_annotations(raw)
                
                event_map = {}
                tgt = None
                for k,v in event_id.items():
                    if k[0:4]=='#Tgt':
                        tgt = k[4]
                    event_map[v] = k

                assert tgt is not None
                
                epochs = mne.Epochs(raw, events, event_id=event_id, 
                                  tmin=tmin, tmax=tmax,
                                  event_repeated='drop', 
                                  preload=True, 
                                  proj=False)
                                  
                epochs.filter(fmin, fmax, method='iir')
                epochs.resample(256)
                
                stims = [x[2] for x in epochs.events]
                data = epochs.get_data()
                
                for i, (d, t) in enumerate(zip(data, stims)):
                    t = event_map[t]
                    if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0]=='#':
                        continue
                    
                    label = 1 if tgt in t else 0
                    x = torch.tensor(d * 1e3)
                    
                    save_path = dataset_fold / f"{label}" / f"{i}.sub{sub}"
                    torch.save(x, save_path)

        return dataset_fold