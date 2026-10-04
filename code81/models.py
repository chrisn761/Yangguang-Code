"""Original GEL topology; mask-aware full sequences. No engineered performance ranking."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODELS = ['GEL', 'GEL-no-attention', 'GEL-no-residual', 'GRU']

def recurrent(layer, x, lengths):
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    result, _ = layer(packed)
    return pad_packed_sequence(result, batch_first=True, total_length=x.shape[1])[0]

class Block(nn.Module):
    def __init__(self, residual=True):
        super().__init__()
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.drop = nn.Dropout(.3)
        self.residual = residual

    def forward(self, x, lengths):
        out = recurrent(self.lstm, x, lengths)
        return self.drop(self.norm(out + x if self.residual else out))

class Model(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in MODELS:
            raise ValueError(name)
        self.name = name
        self.projection = nn.Linear(81, 64)
        self.drop = nn.Dropout(.3)
        if name == 'GEL-no-attention':
            # Deliberately simple ablation: one unidirectional LSTM and masked mean.
            self.rnn = nn.LSTM(64, 64, 1, batch_first=True)
            width = 64
            self.blocks = nn.ModuleList()
        elif name == 'GRU':
            self.rnn = nn.GRU(64, 64, 2, batch_first=True, dropout=.3, bidirectional=False)
            width = 64
            self.blocks = nn.ModuleList()
        else:
            self.rnn = nn.LSTM(64, 64, 2, batch_first=True, dropout=.3, bidirectional=True)
            width = 128
            # Pure residual-connection ablation: retain the recurrent transformations.
            self.blocks = nn.ModuleList([Block(name != 'GEL-no-residual') for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width, 1) if name in ('GEL', 'GEL-no-residual') else None
        if name == 'GEL-no-attention':
            self.head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 3))
        else:
            self.head = nn.Sequential(nn.Linear(width, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                      nn.Dropout(.3), nn.Linear(64, 32), nn.BatchNorm1d(32),
                                      nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 3))

    def forward(self, x, lengths):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        x = self.drop(torch.relu(self.projection(x)))
        x = self.norm(recurrent(self.rnn, x, lengths))
        for block in self.blocks:
            x = block(x, lengths)
        if self.attention is not None:
            scores = self.attention(x).squeeze(-1).masked_fill(~mask, float('-inf'))
            pooled = (x * scores.softmax(1).unsqueeze(-1)).sum(1)
        else:
            pooled = (x * mask.unsqueeze(-1)).sum(1) / lengths.to(x.device)[:, None]
        return self.head(pooled)
