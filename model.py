import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from DataSet import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=9, hidden_size=128, dropout=0.5) # encoder
        self.decoder = nn.RNN(input_size=9, hidden_size=128, dropout=0.5) # decoder
        self.fc = nn.Linear(128, 9)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input(=input_batch): [batch_size, n_step+1, n_class]
        # dec_inpu(=output_batch): [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1) # enc_input: [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [n_step+1, batch_size, n_class]

        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.encoder(enc_input, enc_hidden)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.decoder(dec_input, h_t)

        model = self.fc(outputs) # model : [n_step+1, batch_size, n_class]
        return model
    
read_dictionary = np.load('data.npy',allow_pickle='TRUE').item()
enc_input_all = read_dictionary['train_data']
dec_input_all = read_dictionary['train_data']
dec_output_all = read_dictionary['train_data']
labels = read_dictionary['train_labels']
loader = Data.DataLoader(MyDataset(enc_input_all, dec_input_all, dec_output_all, labels), 100, True, drop_last=True)

model = Seq2Seq().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
for epoch in range(5000):
  for enc_input_batch, dec_input_batch, dec_output_batch, labels in loader:
      # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
      h_0 = torch.zeros(1, 100, 128).to(device)

      (enc_input_batch, dec_intput_batch, dec_output_batch) = (enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))
      # enc_input_batch : [batch_size, n_step+1, n_class]
      # dec_intput_batch : [batch_size, n_step+1, n_class]
      # dec_output_batch : [batch_size, n_step+1], not one-hot
      pred = model(enc_input_batch, h_0, dec_intput_batch)
      # pred : [n_step+1, batch_size, n_class]
      pred = pred.transpose(0, 1) # [batch_size, n_step+1(=6), n_class]
      loss = 0
      for i in range(len(dec_output_batch - 1)):
          # pred[i] : [n_step+1, n_class]
          # dec_output_batch[i] : [n_step+1]
          loss += criterion(pred[i], dec_output_batch[i])
          print(pred[i])
      if (epoch + 1) % 1000 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
          
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()