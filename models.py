import torch
from torch import nn
import torch.nn.functional as F

class ConvBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # Input shape = (B, L, 1, H, W)
        #             = (B, L, 1, 32, 32)

        # conv-flattened shape = (B*L, 1, 32, 32)
        self.conv_layers = nn.Sequential(
            # (B*L, 1, 32, 32)
            nn.Conv2d(1, 64, (5, 5)),
            nn.ELU(),
            # (B*L, 64, 28, 28)
            nn.MaxPool2d((2, 2)),
            # (B*L, 64, 14, 14)

            nn.Conv2d(64, 128, (3, 3)),
            nn.ELU(),
            # (B*L, 128, 12, 12)
            nn.MaxPool2d((2, 2)),
            # (B*L, 128, 6, 6)

            nn.Conv2d(128, 128, (3, 3)),
            nn.ELU(),
            # (B*L, 128, 4, 4)
            nn.MaxPool2d((2, 2)),
            # (B*L, 128, 2, 2)
        )


        self.flat_layers = nn.Sequential(
            # (B*L, 128*2*2)            
            nn.Linear(128 * 2 * 2, 64),
            nn.ELU(),
            nn.Dropout(0.5)
            # (B*L, 64)
        )

        self.lstm = nn.LSTM(
            64, 128,
            num_layers=1, # only 1 layer per cell
            batch_first=True,
            bidirectional=True
        )
        # (B, L, 128*2) (output size doubls because of bidirectionality)

        self.output_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 2, 2),
        )

    def forward(self, x):
        # x shape (B?, L, 32, 32)

        is_batch = x.ndim == 4
        batch_size = x.shape[0] if is_batch else 1

        x = x.unsqueeze(-3)
        # (B?, L, 1, 32, 32)

        # index of first non-batch dimension
        i = 1 if is_batch else 0

        sql_len = x.shape[i]

        # flatten the B and L dimension
        if is_batch:
            x = x.flatten(-5, -4)

        # (B?*L, 1, 32, 32)

        x = self.conv_layers(x)
        # (B?*L, 128, 2, 2)
        x = x.flatten(-3, -1)
        # (B?*L, 128*2*2)
        x = self.flat_layers(x)
        # (B?*L, 64)

        if is_batch:
            x = x.view(batch_size, sql_len, -1)
        else:
            x = x.view(sql_len, -1)
        # (B?, L, 64)

        x, _ = self.lstm(x)
        # (B?, L, 256)
        logits = self.output_layers(x)
        # (B?, L, 2)

        return logits



