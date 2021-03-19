import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

class MyDataset(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all, labels):
        """Reads source and target sequences from processing file ."""
        enc_input_all = np.delete(enc_input_all, enc_input_all.shape[0] - 1, axis = 0)
        self.enc_input_all = (torch.from_numpy(enc_input_all)).float()
        dec_input_all = np.delete(dec_input_all, 0, axis = 0)
        dec_output_all = np.delete(dec_output_all, 0, axis = 0)
        self.dec_input_all = (torch.from_numpy(dec_input_all)).float()
        self.dec_output_all = (torch.from_numpy(dec_output_all)).float()
        labels = np.delete(labels, labels.shape[0] - 1, axis = 0)
        self.labels = (torch.torch.FloatTensor(labels))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.enc_input_all[index], self.dec_input_all[index], self.dec_output_all[index], self.labels[index]

    def __len__(self):
        return len(self.enc_input_all)


# def create_dataset(data, batch_size=10, shuffle=True, drop_last=True):
#     trainX, validX, testX, trainY, validY, testY = data
#     train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
#     valid_dl = DataLoader(MyDataset(validX, validY), batch_size=10, shuffle=False, drop_last=False)
#     test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
#     return train_dl, valid_dl, test_dl
# def create_dataset_full(data, batch_size=10, shuffle=True, drop_last=True):
#     trainX, testX, trainY, testY = data
#     train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
#     test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
#     return train_dl, test_dl

