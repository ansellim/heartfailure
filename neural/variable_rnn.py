# George Seah, Ansel Lim

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

# Set maximum sequence length for Variable RNN model

MAX_SEQUENCE_LENGTH = 2500


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        self.FC32 = nn.Sequential(nn.Linear(dim_input, 32))
        self.GRU = nn.GRU(input_size=32, hidden_size=16,
                          num_layers=2,  # added another layer
                          bias=True, batch_first=True,
                          dropout=0.8,  # added dropout
                          bidirectional=False)
        self.FC2 = nn.Sequential(nn.Linear(16, 8),
                                 nn.ReLU(),
                                 nn.Linear(8, 2)  # added another linear layer!
                                 )

    def forward(self, input_tuple):
        batch = input_tuple
        seqs = batch
        output = self.FC32(seqs)
        output = torch.tanh(output)
        h0 = torch.zeros(2,  # D* num_layers
                         output.size(0),  # batch size
                         16  # hidden size
                         , device=input_tuple.device
                         ).requires_grad_()

        output2, _ = self.GRU(output, h0.detach())
        output3 = output2[:, -1, :]
        output4 = self.FC2(output3)
        return output4


class HeartFailureDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        seqs_arrays = self.encodings[idx].toarray()
        labels_arrays = self.labels[idx]
        return seqs_arrays, labels_arrays

    def __len__(self):
        return len(self.labels)


def to_tensors(batch):
    seqs_array = np.array([x[0] for x in batch])
    labels_array = np.array([x[1] for x in batch])
    seqs_tensor = torch.FloatTensor(seqs_array)
    labels_tensor = torch.LongTensor(labels_array)
    return seqs_tensor, labels_tensor


train = pd.read_csv("../preprocessing/processed_data/train.csv")[['text', 'label']]
val = pd.read_csv("../preprocessing/processed_data/val.csv")[['text', 'label']]
test = pd.read_csv("../preprocessing/processed_data/test.csv")[['text', 'label']]

train['text'] = train.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])
train['val'] = val.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])
test['val'] = test.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=to_tensors,
                          num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=to_tensors,
                          num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=to_tensors,
                         num_workers=NUM_WORKERS)
