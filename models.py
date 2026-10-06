import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """Kim-style CNN: embedding -> parallel 1D convs (n-gram detectors) -> max-over-time pool -> linear."""

    def __init__(self, vocab_size, emb_dim=100, n_filters=64, kernel_sizes=(2, 3, 4, 5),
                 n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):                       # x: (B, T)
        e = self.emb(x).transpose(1, 2)         # (B, E, T)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class BiLSTM(nn.Module):
    """Embedding -> bidirectional LSTM -> concat final fwd/bwd hidden states -> linear."""

    def __init__(self, vocab_size, emb_dim=100, hidden=64, n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):                       # x: (B, T), padding id = 0 at the end
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)           # h: (2, B, hidden)
        h = torch.cat([h[-2], h[-1]], dim=1)    # forward + backward final states
        return self.fc(self.drop(h))


def build_model(name, vocab_size):
    if name == "cnn":
        return TextCNN(vocab_size)
    if name == "lstm":
        return BiLSTM(vocab_size)
    raise ValueError(name)
