"""Explicit architecture variants; no enforced performance ranking."""
import torch
from torch import nn
from light_models import LightModel, recurrent

class UniGEL(nn.Module):
    """GEL topology with the bidirectional recurrent layers replaced by UniLSTM64."""
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(81,64)
        self.drop=nn.Dropout(.3)
        self.rnn=nn.LSTM(64,64,2,batch_first=True,dropout=.3,bidirectional=False)
        self.norm=nn.LayerNorm(64)
        self.blocks=nn.ModuleList([self._block() for _ in range(2)])
        self.attention=nn.Linear(64,1)
        self.head=nn.Sequential(nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
                                nn.Dropout(.3),nn.Linear(64,32),nn.BatchNorm1d(32),
                                nn.ReLU(),nn.Dropout(.3),nn.Linear(32,3))

    def _block(self):
        return nn.ModuleDict({'lstm':nn.LSTM(64,64,1,batch_first=True),
                              'norm':nn.LayerNorm(64),'drop':nn.Dropout(.3)})

    def forward(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:] < lengths.to(x.device)[:,None]
        x=self.drop(torch.relu(self.projection(x)))
        x=self.norm(recurrent(self.rnn,x,lengths))
        for block in self.blocks:
            y=recurrent(block['lstm'],x,lengths)
            x=block['drop'](block['norm'](y+x))
        scores=self.attention(x).squeeze(-1).masked_fill(~mask,float('-inf'))
        pooled=(x*scores.softmax(1).unsqueeze(-1)).sum(1)
        return self.head(pooled)

class UniGELVariant(nn.Module):
    """Controlled UniGEL variants: attention and residual switches are explicit."""
    def __init__(self, attention=True, residual=True):
        super().__init__()
        self.use_attention=attention
        self.use_residual=residual
        self.projection=nn.Linear(81,64)
        self.drop=nn.Dropout(.3)
        self.rnn=nn.LSTM(64,64,2,batch_first=True,dropout=.3,bidirectional=False)
        self.norm=nn.LayerNorm(64)
        self.blocks=nn.ModuleList([self._block() for _ in range(2)])
        self.attention=nn.Linear(64,1) if attention else None
        self.head=nn.Sequential(nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
                                nn.Dropout(.3),nn.Linear(64,32),nn.BatchNorm1d(32),
                                nn.ReLU(),nn.Dropout(.3),nn.Linear(32,3))

    def _block(self):
        return nn.ModuleDict({'lstm':nn.LSTM(64,64,1,batch_first=True),
                              'norm':nn.LayerNorm(64),'drop':nn.Dropout(.3)})

    def forward(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:] < lengths.to(x.device)[:,None]
        x=self.drop(torch.relu(self.projection(x)))
        x=self.norm(recurrent(self.rnn,x,lengths))
        for block in self.blocks:
            y=recurrent(block['lstm'],x,lengths)
            if self.use_residual:
                y=y+x
            x=block['drop'](block['norm'](y))
        if self.attention is not None:
            scores=self.attention(x).squeeze(-1).masked_fill(~mask,float('-inf'))
            pooled=(x*scores.softmax(1).unsqueeze(-1)).sum(1)
        else:
            pooled=(x*mask.unsqueeze(-1)).sum(1)/lengths.to(x.device)[:,None]
        return self.head(pooled)

class UniLSTMBase(nn.Module):
    """Independent one-layer unidirectional LSTM baseline."""
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(81,64)
        self.drop=nn.Dropout(.3)
        self.rnn=nn.LSTM(64,64,1,batch_first=True,bidirectional=False)
        self.norm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,3))

    def forward(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:] < lengths.to(x.device)[:,None]
        x=self.drop(torch.relu(self.projection(x)))
        x=self.norm(recurrent(self.rnn,x,lengths))
        pooled=(x*mask.unsqueeze(-1)).sum(1)/lengths.to(x.device)[:,None]
        return self.head(pooled)

def revised(name):
    if name=='GEL-unidirectional':
        return UniGEL()
    if name=='GEL-no-attention-final':
        return UniGELVariant(attention=False,residual=True)
    if name=='GEL-no-residual-final':
        return UniGELVariant(attention=True,residual=False)
    if name=='LSTM-final':
        return UniLSTMBase()
    if name=='Transformer':
        return LightModel('Transformer-compact')
    if name in ('MSCNN-LSTM','GEL-MSCNN-attention'):
        m=LightModel('MSCNN-LSTM-compact')
        m.rnn=nn.LSTM(24,16,1,batch_first=True)
        m.norm=nn.LayerNorm(16)
        m.head=nn.Sequential(nn.Linear(16,8),nn.ReLU(),nn.Dropout(.3),nn.Linear(8,3))
        if name=='GEL-MSCNN-attention': m.gel_attention=nn.Linear(16,1)
        return m
    m=LightModel('LSTM')
    m.name='GRU' if name=='GRU' else 'UniLSTM-mean'
    recurrent=nn.GRU if name=='GRU' else nn.LSTM
    m.rnn=recurrent(64,64,2,batch_first=True,bidirectional=False,dropout=.3)
    m.norm=nn.LayerNorm(64)
    m.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,3))
    return m
