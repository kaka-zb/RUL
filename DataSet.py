import torch
import numpy as np
import torch.utils.data as Data

class CMAPSS_Dataset_train(Data.Dataset):
    def __init__(self, src_seq, trg_seq, RUL_label):
        self.src_seq = (torch.from_numpy(src_seq)).float()
        self.trg_seq = (torch.from_numpy(trg_seq)).float()
        self.RUL_label = torch.FloatTensor(RUL_label)
    
    def __getitem__(self, idx):
        return self.src_seq[idx], self.trg_seq[idx], self.RUL_label[idx]
    
    def __len__(self):
        return len(self.RUL_label)
    
class CMAPSS_Dataset_valid_or_test(Data.Dataset):
    def __init__(self, src_seq, RUL_label):
        self.src_seq = (torch.from_numpy(src_seq)).float()
        self.trg_seq = torch.zeros_like(self.src_seq)
        self.RUL_label = torch.FloatTensor(RUL_label)
        
    def __getitem__(self, idx):
        return self.src_seq[idx], self.trg_seq[idx], self.RUL_label[idx]
    
    def __len__(self):
        return len(self.RUL_label)