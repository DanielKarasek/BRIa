from functorch import einops
from torch import nn


class Transformer1d(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples, n_classes)

    Pararmetes:

    """

    def __init__(self):
        super(Transformer1d, self).__init__()

        self.d_model = 32
        self.nhead = 4
        self.n_length = 16
        self.dim_feedforward = 256

        self.embedding_layer = nn.Sequential(nn.Linear(32, 32),
                                             nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, x):
        x = einops.rearrange(x,
                             'batch (window_size window_count) -> (batch window_count) window_size',
                             window_size=32)
        x = self.embedding_layer(x)
        x = einops.rearrange(x, '(batch window_count) embedding_dim -> batch embedding_dim window_count', window_count=16)
        x = einops.rearrange(x, 'batch embedding_dim window_count -> window_count batch embedding_dim')

        x = self.transformer_encoder(x)

        # x = x.mean(0)
        x = einops.rearrange(x, 'window_count batch embedding_dim -> batch (embedding_dim window_count)')
        x = self.batch_norm(x)
        return x
