import torch
from torch import nn
import torch.nn.functional as F
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
    
    def forward(self, src):
        '''
        src -> [batch_size, seq_len, sensors]
        '''
        
        # src -> [seq_len, batch_size, sensors]
        src = src.transpose(0, 1)
        enc_output, enc_hidden = self.rnn(src)
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim = 1)))

        # enc_output -> [seq_len, batch_size, enc_hid_dim * num_directions]
        # s -> [batch_size, dec_hid_dim]
        return enc_output, s
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim, bias = False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, s, enc_output):
        '''
        s & enc_output is the output of Encoder
        '''
        
        seq_len = enc_output.shape[0]
        
        # s -> [batch_size, seq_len, dec_hid_dim]
        # enc_output -> [batch_size, seq_len, enc_hid_dim * num_directions]
        s = s.unsqueeze(1).repeat(1, seq_len, 1)
        enc_output = enc_output.transpose(0, 1)
        
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        attention = self.v(energy).squeeze(2)
        
        # attention -> [batch_size, seq_len]
        # dim = 1 -> do softmax for every row
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.GRU(enc_hid_dim * 2 + output_dim, dec_hid_dim)
        self.fc = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + output_dim, output_dim)
        
    def forward(self, dec_input, s, enc_output):
        '''
        s & enc_output is the output of Encoder
        
        dec_input -> [batch_size, 1, output_dim]
        '''
        
        dec_input = dec_input.transpose(0, 1)
        a = self.attention(s, enc_output).unsqueeze(1)
        enc_output = enc_output.transpose(0, 1)
        
        # c -> [1, batch_szie, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)
        
        rnn_input = torch.cat((dec_input, c), dim = 2)
        
        # dec_output = [seq_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        
        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        pred = self.fc(torch.cat((dec_output, c, dec_input), dim = 1))
        
        return pred, dec_hidden.squeeze(0)
    

class RUL_pred(nn.Module):
    def __init__(self, dec_hid_dim, dropout_rate = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(dec_hid_dim * 2, dec_hid_dim)
        self.fc2 = nn.Linear(dec_hid_dim, dec_hid_dim // 2)
        self.fc3 = nn.Linear(dec_hid_dim // 2, 1)
        
    def forward(self, enc_output, dec_output):
        out = torch.cat((enc_output, dec_output), dim = 1)
        out = self.fc1(out)
        out = F.dropout(F.relu(out), p = self.dropout_rate, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout_rate, training=self.training)
        out = self.fc3(out)
        
        return out
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, RUL_pred, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.RUL_pred = RUL_pred
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        '''
        src -> [batch_size, seq_len, sensors]
        trg -> [batch_size, seq_len, output_dim]
        sensors = output_dim
        '''
        
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        sensors = src.shape[2]
        
        # tensor to store the predicted sequence
        outputs = torch.zeros(batch_size, seq_len, sensors).to(self.device)
        
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)
        
        # save the output features of encoder
        encoder_features = s.clone()
        
        dec_input = torch.zeros(batch_size, 1, sensors, device=self.device)
        
        for t in range(0, seq_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)
            outputs[:, t, :] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t, :].unsqueeze(1) if teacher_force else dec_output.unsqueeze(1)
        
        RUL_label_pred = self.RUL_pred(encoder_features, s)
        
        return outputs, RUL_label_pred      