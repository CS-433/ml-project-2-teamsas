import torch
import torch.nn as nn
import torch.nn.functional as F


class TextModel(nn.Module):
    """
    A simple text classification model with a GRU layer.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the word embeddings.
        embedding_matrix (np.ndarray): A matrix of shape (vocab_size, embedding_dim)
            containing the pre-trained word embeddings.
        num_units (int): The number of units in the GRU layer.
        num_layers (int): The number of layers in the GRU layer.
        dropout (float): The dropout rate.
        num_classes (int): The number of classes in the classification task.
    """
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
    """
    A text classification model with side information.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the word embeddings.
        embedding_matrix (np.ndarray): A matrix of shape (vocab_size, embedding_dim)
            containing the pre-trained word embeddings.
        num_units (int): The number of units in the GRU layer.
        num_layers (int): The number of layers in the GRU layer.
        dropout (float): The dropout rate.
        side_info_dim (int): The dimension of the side information.
        num_classes (int): The number of classes in the classification task.
    """
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
