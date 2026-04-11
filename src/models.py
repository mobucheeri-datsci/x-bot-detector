import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    glove_dim, rnn_hidden, rnn_layers, dropout,
    num_numeric_features, max_vocab_size,
    cnn_filters, cnn_num_filters,
)


class NumericNet(nn.Module):
    def __init__(self, num_features=num_numeric_features, hidden=128, dropout=dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BiGRU_LSTM(nn.Module):
    def __init__(self, vocab_size=max_vocab_size, embed_dim=glove_dim,
                 rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                 num_numeric=num_numeric_features, dropout=dropout,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32), requires_grad=False)

        self.bigru = nn.GRU(embed_dim, rnn_hidden, num_layers=rnn_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if rnn_layers > 1 else 0)
        self.lstm = nn.LSTM(rnn_hidden * 2, rnn_hidden, num_layers=rnn_layers,
                            batch_first=True, dropout=dropout if rnn_layers > 1 else 0)

        self.numeric_net = NumericNet(num_numeric, hidden=128, dropout=dropout)

        fused_dim = 128
        self.text_proj = nn.Linear(rnn_hidden, fused_dim)
        self.gate = nn.Linear(fused_dim * 2, fused_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, 1)

    def forward(self, input_ids, numeric):
        embedded = self.embedding(input_ids)
        gru_out, _ = self.bigru(embedded)
        _, (h_n, _) = self.lstm(gru_out)
        text_repr = self.text_proj(h_n.squeeze(0))
        num_repr = self.numeric_net(numeric)

        gate = torch.sigmoid(self.gate(torch.cat([text_repr, num_repr], dim=1)))
        fused = gate * text_repr + (1 - gate) * num_repr

        x = self.dropout(F.relu(fused))
        return self.classifier(x)


class CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size=max_vocab_size, embed_dim=glove_dim,
                 filter_sizes=cnn_filters, num_filters=cnn_num_filters,
                 rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                 num_numeric=num_numeric_features, dropout=dropout,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32), requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        cnn_out = num_filters * len(filter_sizes)

        self.bilstm = nn.LSTM(cnn_out, rnn_hidden, num_layers=rnn_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if rnn_layers > 1 else 0)

        self.numeric_net = NumericNet(num_numeric, hidden=128, dropout=dropout)

        fused_dim = 128
        self.text_proj = nn.Linear(rnn_hidden * 2, fused_dim)
        self.gate = nn.Linear(fused_dim * 2, fused_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, 1)

    def forward(self, input_ids, numeric):
        embedded = self.embedding(input_ids).permute(0, 2, 1)
        conv_outs = [F.max_pool1d(F.relu(conv(embedded)), conv(embedded).size(2)).squeeze(2)
                     for conv in self.convs]
        cnn_out = torch.cat(conv_outs, dim=1).unsqueeze(1)

        _, (h_n, _) = self.bilstm(cnn_out)
        text_repr = self.text_proj(torch.cat([h_n[-2], h_n[-1]], dim=1))
        num_repr = self.numeric_net(numeric)

        gate = torch.sigmoid(self.gate(torch.cat([text_repr, num_repr], dim=1)))
        fused = gate * text_repr + (1 - gate) * num_repr

        x = self.dropout(F.relu(fused))
        return self.classifier(x)
