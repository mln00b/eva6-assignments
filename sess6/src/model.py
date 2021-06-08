import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, use_batchnorm=False, use_groupnorm=False, use_layernorm=False):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.use_groupnorm = use_groupnorm
        self.use_layernorm = use_layernorm

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self._get_norm((16, 26, 26)),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self._get_norm((16, 24, 24)),
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self._get_norm((16, 10, 10)),
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self._get_norm((16, 3, 3)),
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10,
                      kernel_size=(3, 3), padding=0, bias=False),
        )

        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)

    def _get_norm(self, inp_shape):
        if self.use_batchnorm:
            return nn.BatchNorm2d(inp_shape[0])
        elif self.use_groupnorm:
            return nn.GroupNorm(num_groups=2, num_channels=inp_shape[0])
        elif self.use_layernorm:
            return nn.LayerNorm(inp_shape)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.dropout_1(x)
        x = self.convblock3(x)
        x = self.pool2(x)
        x = self.dropout_2(x)
        x = self.convblock4(x)
        x = self.convblock5(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)