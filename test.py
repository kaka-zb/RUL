# import numpy as np
# read_dictionary = np.load('data.npy',allow_pickle='TRUE').item()
# a = read_dictionary['train_data']
# print(a.shape)
# b = np.delete(a, a.shape[0] - 1, axis = 0)
# print(b.shape)


# import numpy as np
# read_dictionary = np.load('data.npy',allow_pickle='TRUE').item()
# a = read_dictionary['valid_labels']
# b = np.delete(a, a.shape[0] - 1, axis = 0)
# print(a.shape)
# print(b.shape)

# letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
# letter2idx = {n: i for i, n in enumerate(letter)}
# print(len(letter2idx))
 
 
# import torch

# a = torch.zeros(1, 2, 2)
# print(a.shape)
# a = a.squeeze(0)
# print(a.shape)

import torch
from torch import nn
from seq2seq import Encoder, Decoder, Attention, Seq2Seq, RUL_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attn = Attention(512, 512)
enc = Encoder(14, 512, 512)
dec = Decoder(14, 512, 512, attn)
RUL_pre = RUL_pred(512)

model = Seq2Seq(enc, dec, RUL_pre, device).to(device)

src = torch.rand(2, 10, 30, 14)
trg = torch.rand(2, 10, 30, 14)
labels = torch.rand(2, 10, 30)

model.train()
for i in range(0, 2):
    src_ = src[i]
    trg_ = trg[i]
    labels_ = labels[i]
    
    output, RUL_label = model(src_, trg_)
    print(output.shape)
    print(RUL_label.shape)
    
    