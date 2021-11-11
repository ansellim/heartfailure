import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        self.FC32 = nn.Sequential( nn.Linear(dim_input,32) )
        self.GRU =nn.GRU(input_size=32, hidden_size=16,
                         num_layers=2, #added another layer
                         bias=True, batch_first=True,
                         dropout=0.8, #added dropout
                         bidirectional=False)
        self.FC2 =nn.Sequential(nn.Linear(16,8),
                                nn.ReLU(),
                                nn.Linear(8,2) # added another linear layer!
                                )
    def forward(self, input_tuple):
        batch = input_tuple
        seqs = batch
        output = self.FC32(seqs)
        output = torch.tanh(output)
        h0 = torch.zeros(2, #D* num_layers
                         output.size(0), #batch size
                         16 #hidden size
                         ).requires_grad_()
        output2, _ = self.GRU(output, h0.detach())
        output3 = output2[:,-1,:]
        output4 = self.FC2(output3)
        return output4