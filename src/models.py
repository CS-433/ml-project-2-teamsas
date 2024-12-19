import torch
import torch.nn as nn
import torch.nn.functional as F


class TextModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_matrix,
        num_units,
        num_layers,
        dropout,
        num_classes,
    ):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        self.dropout = dropout if num_layers > 1 else 0
        self.bidirectional_gru = nn.GRU(
            embedding_dim,
            num_units,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc1 = nn.Linear(num_units * 2, num_units)
        self.fc2 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bidirectional_gru(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TextModelWithSideInfo(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_matrix,
        num_units,
        num_layers,
        dropout,
        side_info_dim,
        num_classes,
    ):
        super(TextModelWithSideInfo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        self.dropout = dropout if num_layers > 1 else 0
        self.bidirectional_gru = nn.GRU(
            embedding_dim,
            num_units,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.side_info_fc = nn.Linear(side_info_dim, num_units)
        self.fc1 = nn.Linear(num_units * 3, num_units)
        self.fc2 = nn.Linear(num_units, num_classes)

    def forward(self, x, side_info):
        x = self.embedding(x)
        x, _ = self.bidirectional_gru(x)
        x = x[:, -1, :]

        side_info = F.relu(self.side_info_fc(side_info))
        x = torch.cat([x, side_info], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
