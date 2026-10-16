from slr.models.base import BaseSequenceModel
from torch import nn
import torch

class RecurrentClassifier(BaseSequenceModel):
    """LSTM or GRU: reads the frame sequence and predicts the sign from the
    hidden state at the last frame."""

    def __init__(self, num_features: int, num_classes: int, rnn_type: str = "lstm",
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(num_features, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # out: (B, T, hidden)
        return self.head(out[:, -1])  # use the output at the last frame

class TemporalCNN(BaseSequenceModel):
    """1D convolutions along the time axis: detect characteristic motion patterns."""

    def __init__(self, num_features: int, num_classes: int,
                 channels: int = 128, dropout: float = 0.3) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                                 nn.BatchNorm1d(c_out), nn.ReLU())

        self.features = nn.Sequential(block(num_features, channels),
                                      block(channels, channels),
                                      block(channels, channels))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2))  # Conv1d expects (B, F, T)
        return self.head(x.mean(dim=2))  # average over time