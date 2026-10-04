"""Lightweight comparison models, not designed to enforce any ranking."""
import math
import torch
from torch import nn
from models import Model as OriginalModel, recurrent

COMPACT_MODELS = ['CNN-LSTM-compact', 'MSCNN-LSTM-compact', 'Transformer-compact']
SIMULATION_MODELS = ['CNN-LSTM-compact-simulation', 'MSCNN-LSTM-compact-simulation']
IMPROVED_GEL = 'GEL-CNN-attention-compact'
EXTRA_MODELS = ['LSTM', 'CNN-LSTM', 'MSCNN-LSTM', 'Transformer'] + COMPACT_MODELS + SIMULATION_MODELS + [IMPROVED_GEL]

class LightModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        use_gel_attention = name == IMPROVED_GEL
        if use_gel_attention:
            name = 'CNN-LSTM-compact'
        compact = name in COMPACT_MODELS
        name = name.removesuffix('-compact')
        self.name = name
        hidden = 32 if compact else 64
        self.drop = nn.Dropout(.3)
        if name == 'CNN-LSTM':
            width = 16 if compact else 32
            self.convs = nn.ModuleList([nn.Conv1d(81, width, 5, padding=2)])
        elif name == 'MSCNN-LSTM':
            channels = 8 if compact else 16
            self.convs = nn.ModuleList([nn.Conv1d(81, channels, k, padding=k//2) for k in (3, 5, 7)])
            width = channels * 3
        else:
            self.convs = nn.ModuleList()
            width = hidden
            self.proj = nn.Linear(81, hidden)
        if name == 'Transformer':
            layer = nn.TransformerEncoderLayer(hidden, 2 if compact else 4, dim_feedforward=hidden*2, dropout=.3, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.rnn = nn.LSTM(width, hidden, 1, batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(.3), nn.Linear(hidden//2, 3))
        # Original GEL attention form: scalar linear score -> temporal softmax
        # -> weighted sum. Only input width changes (128 -> 32).
        # Construct last so same-seed backbone initialization matches baseline.
        self.gel_attention = nn.Linear(hidden, 1) if use_gel_attention else None

    def forward(self, x, lengths):
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        if self.convs:
            x = torch.cat([torch.relu(c(x.transpose(1, 2))).transpose(1, 2) for c in self.convs], -1)
        else:
            x = torch.relu(self.proj(x))
        x = self.drop(x) * mask.unsqueeze(-1)
        if self.name == 'Transformer':
            # Local attention over every nonoverlapping 128-frame window avoids
            # quadratic memory in full-record length. No frames are discarded.
            b, t, d = x.shape
            pos = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
            freq = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype)*(-math.log(10000.)/d))
            pe = torch.zeros(t, d, device=x.device, dtype=x.dtype)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
            x = x + pe
            padding = (-t) % 128
            x = nn.functional.pad(x, (0, 0, 0, padding)).reshape(b, -1, 128, d)
            masks = nn.functional.pad(mask, (0, padding)).reshape(b, -1, 128)
            flat, valid = x.reshape(-1, 128, d), masks.reshape(-1, 128)
            keep = valid.any(1)
            out = torch.zeros_like(flat)
            out[keep] = self.encoder(flat[keep], src_key_padding_mask=~valid[keep])
            x = out.reshape(b, -1, d)[:, :t]
            x = self.norm(x)
            pooled = (x*mask.unsqueeze(-1)).sum(1)/lengths.to(x.device)[:, None]
        else:
            x = self.norm(recurrent(self.rnn, x, lengths))
            if self.gel_attention is not None:
                scores = self.gel_attention(x).squeeze(-1).masked_fill(~mask, float('-inf'))
                pooled = (x * scores.softmax(1).unsqueeze(-1)).sum(1)
            elif self.name == 'LSTM':
                pooled = x[torch.arange(len(x), device=x.device), lengths.to(x.device)-1]
            else:
                pooled = (x*mask.unsqueeze(-1)).sum(1)/lengths.to(x.device)[:, None]
        return self.head(pooled)

def build_model(name):
    if name == 'CNN-LSTM-compact-simulation': name = 'CNN-LSTM-compact'
    if name == 'MSCNN-LSTM-compact-simulation': name = 'MSCNN-LSTM-compact'
    return LightModel(name) if name in EXTRA_MODELS or name in COMPACT_MODELS else OriginalModel(name)

def architecture(name):
    if name == 'CNN-LSTM-compact-simulation':
        return 'CNN-LSTM Compact with standardized Gaussian training noise (simulation only)'
    if name == 'MSCNN-LSTM-compact-simulation':
        return 'MSCNN-LSTM Compact with standardized Gaussian training noise (simulation only)'
    if name == IMPROVED_GEL:
        return 'CNN-LSTM Compact backbone: Conv81-16 kernel5 + single unidirectional LSTM32 + LayerNorm + GEL linear32-1 masked attention pooling + head32-16-3; no residual blocks'
    if name in COMPACT_MODELS:
        return {'CNN-LSTM-compact': '16-channel kernel5 CNN + single unidirectional LSTM32 + mean + head32-16-3',
                'MSCNN-LSTM-compact': '3/5/7 CNN branches 8 channels each + single unidirectional LSTM32 + mean + head32-16-3',
                'Transformer-compact': '32-dim 2-head 1-layer FF64 local128-frame Transformer, sinusoidal positions, full-record mean, head32-16-3'}[name]
    return {'GEL': '2-layer bidirectional LSTM + two residual blocks + attention',
            'GEL-no-attention': 'Simplified GEL without attention: 1-layer unidirectional LSTM, no residual blocks, mean pooling',
            'GEL-no-residual': 'GEL with residual additions removed, recurrent blocks retained',
            'GRU': '2-layer unidirectional GRU + mean pooling',
            'LSTM': '1-layer unidirectional LSTM + last valid state',
            'CNN-LSTM': '32-channel kernel5 CNN + 1-layer unidirectional LSTM',
            'MSCNN-LSTM': '3/5/7 CNN branches, 16 channels each + 1-layer unidirectional LSTM',
            'Transformer': '64-dim, 4-head, 1-layer, FF128, local128-frame attention, absolute sinusoidal positions, full-record mean'}[name]
